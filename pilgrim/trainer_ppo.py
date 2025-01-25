import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import os
import time

from .environment import PermutationEnv
from .model_policy import PilgrimPolicy
from .model_value import PilgrimValue


class PPOTrainer:
    def __init__(
        self,
        policy_net,                 # Actor: PolicyNetwork (PilgrimPolicy)
        value_net,                  # Critic: Pilgrim (PilgrimValue)
        n,                          # размер перестановки
        device='cpu',
        gamma=0.99,                 # Коэффициент дисконтирования
        lambd=0.95,                 # GAE-ламбда (или используйте 1.0 для простоты)
        eps_clip=0.2,               # Коэффициент клиппинга в PPO
        lr_policy=1e-4,            # Learning rate для policy
        lr_value=1e-4,             # Learning rate для value
        entropy_coef=0.01,         # Коэффициент при энтропии
        value_loss_coef=0.5,       # Коэффициент при MSE для value
        max_grad_norm=0.5,         # Опционально, если хотим clip grad
        rollout_size=2000,         # Число шагов (transition) на каждом сборе
        mini_batch_size=128,
        ppo_epochs=10,
        # Параметры для curriculum:
        max_steps_start=10,        # Начальная сложность (малое число шагов)
        max_steps_final=100,       # Финальная сложность (большее число шагов)
        curriculum_steps=10,       # За сколько итераций мы «доедем» от start до final
        # Папки для сохранения / загрузки
        save_dir="ppo_saves"
    ):
        self.policy_net = policy_net.to(device)
        self.value_net = value_net.to(device)

        # Изначально создаём среду с некоторым max_steps_start
        self.env = PermutationEnv(n=n, max_steps=max_steps_start, device=device)

        self.device = device
        self.n = n
        self.gamma = gamma
        self.lambd = lambd
        self.eps_clip = eps_clip
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm

        self.rollout_size = rollout_size
        self.mini_batch_size = mini_batch_size
        self.ppo_epochs = ppo_epochs

        # Curriculum
        self.max_steps_start = max_steps_start
        self.max_steps_final = max_steps_final
        self.curriculum_steps = curriculum_steps

        self.optimizer_policy = torch.optim.Adam(self.policy_net.parameters(), lr=lr_policy)
        self.optimizer_value  = torch.optim.Adam(self.value_net.parameters(), lr=lr_value)

        # Для сохранения весов
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # Счётчик итераций, чтобы двигаться по curriculum
        self.iteration_count = 0

    def load_pretrained_models(self, policy_path=None, value_path=None):
        """
        Загрузка предобученных весов для policy и value.
        Если путь не None и существует, загружаем оттуда.
        """
        if policy_path is not None and os.path.isfile(policy_path):
            print(f"Loading pretrained policy from {policy_path}")
            self.policy_net.load_state_dict(torch.load(policy_path, map_location=self.device))

        if value_path is not None and os.path.isfile(value_path):
            print(f"Loading pretrained value from {value_path}")
            self.value_net.load_state_dict(torch.load(value_path, map_location=self.device))

    def save_models(self, iteration):
        """
        Сохранение текущих весов policy и value.
        """
        policy_save_path = os.path.join(self.save_dir, f"policy_{iteration:06d}.pth")
        value_save_path  = os.path.join(self.save_dir, f"value_{iteration:06d}.pth")
        torch.save(self.policy_net.state_dict(), policy_save_path)
        torch.save(self.value_net.state_dict(),  value_save_path)
        print(f"[Iteration {iteration}] Saved policy to {policy_save_path} and value to {value_save_path}")

    def adjust_curriculum(self):
        """
        Простая версия curriculum:
        - увеличиваем max_steps в среде с итерациями
        - например, на (iteration_count / curriculum_steps) fraction 
          увеличиваем с max_steps_start до max_steps_final
        """
        fraction = min(1.0, self.iteration_count / max(self.curriculum_steps, 1))
        new_max_steps = int(self.max_steps_start + fraction * (self.max_steps_final - self.max_steps_start))
        self.env.max_steps = new_max_steps

    def collect_trajectories(self):
        """
        Сбор данных: на каждом шаге:
          - state -> logits -> dist = Categorical -> action
          - применяем action в среде -> (next_state, reward, done)
          - пишем всё в rollout-буфер.
        Возвращает список (или dict со списками) со всеми переходами.
        """

        self.policy_net.eval() 
        self.value_net.eval()
        rollout = []
        steps_collected = 0

        # Начинаем эпизод
        state = torch.tensor(self.env.reset(random_start=True), dtype=torch.float32, device=self.device)
        done = False

        while steps_collected < self.rollout_size:
            # Policy forward
            logits = self.policy_net(state.unsqueeze(0))  # [1, n_actions]
            dist = Categorical(logits=logits)

            action = dist.sample()        # [1]
            log_prob = dist.log_prob(action).detach()  # [1]
            entropy = dist.entropy().detach()          # [1]

            # Value forward
            value = self.value_net(state.unsqueeze(0)).detach()  # [1, 1]

            # Шаг в среде
            next_s, reward, done, _ = self.env.step(action.item())

            rollout.append({
                'state': state,               
                'action': action,             
                'log_prob': log_prob,         
                'entropy': entropy,           
                'value': value.squeeze(0),    
                'reward': torch.tensor([reward], dtype=torch.float32, device=self.device),
                'done': done
            })

            state = torch.tensor(next_s, dtype=torch.float32, device=self.device)
            steps_collected += 1

            if done:
                # Начинаем следующий эпизод, случайный старт
                state = torch.tensor(self.env.reset(random_start=True), dtype=torch.float32, device=self.device)
                done = False

        return rollout

    def compute_advantages(self, rollout):
        """
        Считает Returns и GAE-Advantages для каждого перехода в rollout.
        Дописываем в rollout поля 'advantage' и 'return'.
        """
        rewards = [r['reward'] for r in rollout]  # list of [1]-тензоров
        values  = [r['value']  for r in rollout]  # list of [1]
        dones   = [r['done']   for r in rollout]  # list of bool
        size    = len(rollout)

        # Будем считать next_values для delta
        next_values = values[1:] + [torch.zeros_like(values[0])]

        advantages = []
        gae = torch.zeros_like(values[0])  # Начинаем с нуля

        for i in reversed(range(size)):
            if i == size - 1:
                next_val = torch.zeros_like(values[i]) if dones[i] else values[i]
            else:
                next_val = values[i+1] if not dones[i] else torch.zeros_like(values[i])

            delta = rewards[i] + self.gamma * next_val - values[i]
            if i == size - 1 or dones[i]:
                gae = delta
            else:
                gae = delta + self.gamma * self.lambd * gae

            advantages.append(gae)

        advantages.reverse()

        # return = value + advantage
        for i in range(size):
            rollout[i]['advantage'] = advantages[i]
            rollout[i]['return']    = values[i] + advantages[i]

    def update(self, rollout):
        """
        PPO-обновление по собранному rollout.
        Делим rollout на мини-батчи и делаем K (self.ppo_epochs) проходов.
        """
        self.policy_net.train()
        self.value_net.train()

        states     = torch.stack([r['state'] for r in rollout])     
        actions    = torch.stack([r['action'] for r in rollout])    
        log_probs_old = torch.stack([r['log_prob'] for r in rollout])  
        advantages = torch.stack([r['advantage'] for r in rollout])    
        returns    = torch.stack([r['return'] for r in rollout])       
        entropies  = torch.stack([r['entropy'] for r in rollout])      

        # Нормализация advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = states.size(0)
        i_print = 0
        for _ in range(self.ppo_epochs):
            idxs = torch.randperm(dataset_size, device=self.device)
            for start in range(0, dataset_size, self.mini_batch_size):
                i_print += 1
                end = start + self.mini_batch_size
                batch_idx = idxs[start:end]

                states_b      = states[batch_idx]
                actions_b     = actions[batch_idx]
                old_log_probs_b = log_probs_old[batch_idx]
                adv_b         = advantages[batch_idx]
                ret_b         = returns[batch_idx]

                # Actor
                logits = self.policy_net(states_b)
                dist   = Categorical(logits=logits)
                log_probs_new = dist.log_prob(actions_b)
                entropy_b     = dist.entropy().mean()

                ratio = (log_probs_new - old_log_probs_b).exp()

                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                # Critic
                value_pred = self.value_net(states_b).squeeze(-1)
                value_loss = F.mse_loss(value_pred, ret_b)

                loss = policy_loss \
                       + self.value_loss_coef * value_loss \
                       - self.entropy_coef * entropy_b

                self.optimizer_policy.zero_grad()
                self.optimizer_value.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                self.optimizer_policy.step()
                self.optimizer_value.step()

                if i_print % 10 == 0:
                    print('loss', loss)

    def run(self, total_iterations=100, save_interval=10):
        """
        Основной цикл:
         1. Корректируем max_steps (curriculum learning)
         2. collect_trajectories
         3. compute_advantages
         4. update
         5. сохраняем модели
        """
        for iteration in range(total_iterations):
            print("iteration", iteration)
            self.iteration_count = iteration

            start_time = time.time()
            # 1) Корректируем curriculum
            self.adjust_curriculum()
            end_time = time.time()
            print("curriculum", end_time - start_time)
        

            # 2) Сбор trajectory
            start_time = time.time()
            rollout = self.collect_trajectories()
            end_time = time.time()
            print("collect_trajectories", end_time - start_time)

            # 3) Считаем advantage
            start_time = time.time()
            self.compute_advantages(rollout)
            end_time = time.time()
            print("compute_advantages", end_time - start_time)

            # 4) Обновляем
            start_time = time.time()
            self.update(rollout)
            end_time = time.time()
            print("update", end_time - start_time)

            # Сохраняем модели раз в несколько итераций
            if (iteration + 1) % save_interval == 0:
                self.save_models(iteration + 1)
