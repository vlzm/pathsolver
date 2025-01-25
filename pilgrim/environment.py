import torch
import random

class PermutationEnv:
    def __init__(self, n, goal=None, max_steps=100, reward_for_goal=1.0, device='cpu'):
        """
        :param n: размер перестановки
        :param goal: тензор или список из n элементов – целевая перестановка
        :param max_steps: ограничение на число шагов в эпизоде
        :param reward_for_goal: награда за достижение цели
        :param device: 'cpu' или 'cuda'
        """
        self.n = n
        self.device = device

        # Если goal не задана, пусть будет тождественная перестановка [0,1,2,...,n-1]
        if goal is None:
            goal = torch.arange(n, dtype=torch.long)
        else:
            goal = torch.tensor(goal, dtype=torch.long)
        self.goal = goal.to(self.device)

        self.max_steps = max_steps
        self.reward_for_goal = reward_for_goal

        # Текущее состояние (перестановка)
        self.state = None
        self.steps_count = 0

    def reset(self, random_start=False):
        """
        Сбрасываем окружение к начальному состоянию.
        :param random_start: если True, генерируем случайную перестановку;
                             если False, начинаем с тождественной [0,1,2,...,n-1].
        :return: начальная перестановка (тензор)
        """
        self.steps_count = 0

        if random_start:
            # Генерируем случайную перестановку
            perm = torch.randperm(self.n, device=self.device)
        else:
            # Тождественная перестановка [0,1,2,...,n-1]
            perm = torch.arange(self.n, device=self.device)

        self.state = perm
        return self.state.clone()  # на случай, если кто-то вне класса будет модифицировать state

    def step(self, action):
        """
        Применяем действие к текущей перестановке и возвращаем (next_state, reward, done, info).
        Допустим:
          action=0 -> циклический сдвиг вправо на 1,
          action=1 -> обмен первых двух элементов.
        """
        self.steps_count += 1

        # Клонируем state, чтобы редактировать
        next_state = self.state.clone()

        if action == 0:
            # Циклический сдвиг вправо на 1
            next_state = torch.roll(next_state, 1, dims=0)

        elif action == 1:
            # Транспозиция (обмен) первых двух элементов
            if self.n >= 2:
                tmp = next_state[0].item()
                next_state[0] = next_state[1]
                next_state[1] = tmp

        # Можете добавить другие действия:
        # elif action == 2:
        #     ...
        # elif action == 3:
        #     ...
        # и т.д.

        self.state = next_state

        # Проверяем, достигли ли мы цели
        done = torch.all(self.state == self.goal).item()  # True/False
        reward = 0.01

        if done:
            reward = self.reward_for_goal

        # Если превысили лимит шагов, тоже завершаем
        if self.steps_count >= self.max_steps and not done:
            done = True

        # В поле info обычно что-то дополнительное
        info = {}

        return self.state.clone(), float(reward), done, info

