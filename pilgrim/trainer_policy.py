import torch
import os
import time
import pandas as pd
import schedulefree
from torch import nn
from torch.nn import functional as F

class TrainerPolicy:
    """
    Тренер для policy-сети в стиле "имитационного" обучения.
    Генерирует (state, action) пары через случайные блуждания
    (как в вашем value-тренере), а затем учит сеть воспроизводить
    эти действия (CrossEntropyLoss).
    """
    def __init__(
        self,
        net,                 # Ваша policy-сеть (PolicyNetwork)
        num_epochs, 
        device,
        batch_size=10000,
        lr=0.001,
        name="",
        K_min=1,
        K_max=55,
        all_moves=None,      # Аналогично вашему коду
        inverse_moves=None,  # Аналогично вашему коду
        V0=None,             # Тождественное состояние (или другое начальное)
        optimizer='Adam',    # 'Adam' или 'AdamSF'
    ):
        self.net = net.to(device)
        self.lr = lr
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        
        # Для классификации действий используем CrossEntropyLoss
        self.criterion = nn.CrossEntropyLoss()

        # Инициализация оптимизатора
        if optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        elif optimizer == 'AdamSF':
            import schedulefree
            self.optimizer = schedulefree.AdamWScheduleFree(self.net.parameters(), lr=lr)
        else:
            raise ValueError(f'Wrong optimizer value ({optimizer}). It can be "Adam" or "AdamSF".')

        self.epoch = 0
        self.id = int(time.time())
        self.log_dir = "logs_policy"
        self.weights_dir = "weights_policy"
        self.name = name
        self.K_min = K_min
        self.K_max = K_max

        # Аналогично вашему value-тренеру
        self.walkers_num = 1_000_000 // self.K_max
        self.all_moves = all_moves          # [n_gens, state_size], например
        self.n_gens = all_moves.size(0)     # кол-во генераторов (действий)
        self.state_size = all_moves.size(1)
        self.inverse_moves = inverse_moves  # массив, где inverse_moves[a] = действие, обратное a
        self.V0 = V0                        # начальное состояние

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.weights_dir, exist_ok=True)

    def do_random_step(self, states, last_moves):
        """
        Совершаем случайный шаг (то же, что у вас в value-тренере),
        но возвращаем ещё и 'actions' - какие именно действия были выбраны.
        """
        possible_moves = torch.ones((states.size(0), self.n_gens), dtype=torch.bool, device=self.device)
        # Запрещаем обратное действие:
        possible_moves[torch.arange(states.size(0)), self.inverse_moves[last_moves]] = False
        
        # Выбираем случайное действие из доступных
        next_moves = torch.multinomial(possible_moves.float(), 1).squeeze(-1)  # [batch_size]
        
        # Применяем его к состояниям
        new_states = torch.gather(states, 1, self.all_moves[next_moves])
        return new_states, next_moves

    def generate_random_walks(self, k=1000, K_min=1, K_max=30):
        """
        Генерируем датасет (X, A), где:
          X - это состояния,
          A - это действия, которые привели к этим состояниям.
        
        Логика:
          Для каждого K от K_min до K_max делаем K случайных шагов.
          Записываем все (state[t], action[t]) пары, КРОМЕ последнего state?
          Или включая?
          
          Вопрос: какое состояние считаете входом, а какое действие - меткой?
          Обычно (state[t], action[t]) -> state[t+1].
          Вы можете вернуть именно (state[t], action[t]).
        """
        # (K_max - K_min + 1) * k - общее число конечных состояний
        # Но нам нужны и действия...
        # Давайте будем сохранять все промежуточные шаги!
        # Итого: K шагов -> K пар (state, action).
        # Для упрощения, сохраним ТОЛЬКО последний шаг (как в value коде),
        # хотя обычно для policy стоит сохранять каждый шаг.
        # Но сделаем "аналогично, но для policy":
        
        # Будем для каждого K генерировать (state_K, last_action)
        # где last_action - действие, совершённое на K-том шаге.
        
        X = torch.zeros(((K_max - K_min + 1) * k, self.state_size), dtype=torch.int8, device=self.device)
        A = torch.zeros(((K_max - K_min + 1) * k,), dtype=torch.long, device=self.device)
        
        idx = 0
        for K in range(K_min, K_max + 1):
            # Начинаем с V0
            states = self.V0.repeat(k, 1)  # [k, state_size]
            last_moves = torch.full((k,), -1, dtype=torch.int64, device=self.device)
            chosen_actions = None
            
            for step_idx in range(K):
                states, chosen_actions = self.do_random_step(states, last_moves)
                last_moves = chosen_actions

            # После K шагов мы имеем "конечные" states и "последние" chosen_actions
            # Запомним их
            start_i = idx * k
            end_i   = (idx + 1) * k
            X[start_i:end_i] = states
            A[start_i:end_i] = chosen_actions
            idx += 1

        # Перемешаем (как в вашем коде)
        perm = torch.randperm(X.size(0), device=self.device)
        X = X[perm]
        A = A[perm]
        return X, A

    def _train_epoch(self, X, A):
        """
        Обучаем policy-сеть предсказывать 'A' (действие) по состоянию 'X'.
        """
        self.net.train()
        avg_loss = 0.0
        total_batches = X.size(0) // self.batch_size

        for i in range(0, X.size(0), self.batch_size):
            data = X[i:i + self.batch_size]      # [batch_size, state_size]
            target = A[i:i + self.batch_size]    # [batch_size] (действия)
            
            # Прогон через policy-сеть. Она возвращает logits: [batch_size, n_actions]
            logits = self.net(data.float())  
            
            # CrossEntropyLoss ждет (logits, target) где target - long [batch_size]
            loss = self.criterion(logits, target)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            avg_loss += loss.item()

        return avg_loss / total_batches if total_batches > 0 else avg_loss

    def run(self):
        for epoch in range(self.num_epochs):
            self.epoch += 1

            # Генерация данных
            data_gen_start = time.time()
            X, A = self.generate_random_walks(k=self.walkers_num, K_min=self.K_min, K_max=self.K_max)
            data_gen_time = time.time() - data_gen_start

            # Обучение (одна эпоха)
            epoch_start = time.time()
            train_loss = self._train_epoch(X, A)
            epoch_time = time.time() - epoch_start

            # Логирование
            log_file = f"{self.log_dir}/train_{self.name}_{self.id}.csv"
            log_data = pd.DataFrame([{
                'epoch': self.epoch,
                'train_loss': train_loss,
                'samples_seen': X.size(0),
                'data_gen_time': data_gen_time,
                'train_epoch_time': epoch_time
            }])
            log_data.to_csv(log_file, mode='a', header=not os.path.exists(log_file), index=False)

            # Сохраняем веса на степенях двойки (как у вас)
            if (self.epoch & (self.epoch - 1)) == 0:
                weights_file = f"{self.weights_dir}/{self.name}_{self.id}_e{self.epoch:05d}.pth"
                torch.save(self.net.state_dict(), weights_file)
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                print(f"[{timestamp}] Saved weights at epoch {self.epoch:5d}. Train Loss: {train_loss:.4f}")

            # Сохраняем на эпохах 10000, 50000
            if self.epoch in [10000, 50000]:
                weights_file = f"{self.weights_dir}/{self.name}_{self.id}_e{self.epoch:05d}.pth"
                torch.save(self.net.state_dict(), weights_file)
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                print(f"[{timestamp}] Saved weights at epoch {self.epoch:5d}. Train Loss: {train_loss:.4f}")

        # Сохраняем итоговую модель
        final_weights_file = f"{self.weights_dir}/{self.name}_{self.id}_e{self.epoch:05d}final.pth"
        torch.save(self.net.state_dict(), final_weights_file)
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        print(f"[{timestamp}] Finished. Saved final weights at epoch {self.epoch}. Train Loss: {train_loss:.4f}.")
