import torch
from torch.distributions import Categorical
import random

def test_policy(policy_net, env, k=10, max_steps=200, device='cpu'):
    """
    Пример теста политики:
      1) Берём тождественную перестановку [0,1,2,...,n-1].
      2) Совершаем k случайных шагов в среде, получаем "перемешанное" состояние.
      3) Запускаем policy_net на этом состоянии, пытаемся вернуться к тождественному.
      4) Возвращаем tuple (done, steps_used).
         done=True, если достигли тождественной перестановки за <= max_steps.
    """
    policy_net.eval()

    # 1) Сбрасываем среду в тождественную (random_start=False, если вы так сделали)
    #    Предположим, что в env.reset(...) можно передать random_start=False
    #    и оно вернёт [0,1,2,...].
    state = env.reset(random_start=False)

    # 2) "Закрутим" k раз случайно
    #    Здесь мы вручную делаем k шагов с выбором случайного допустимого действия.
    for _ in range(k):
        # В env может быть несколько действий (например, 2). Выбираем случайное.
        # Либо, если в env есть метод do_random_step, используем его.
        action = random.randrange(env.n_actions)  # Или len(env.possible_actions)
        next_s, _, done, _ = env.step(action)
        state = next_s
        if done:
            # Если вдруг случайно уже вернулись в тождественное — "перемешивание" не особо удалось :)
            break

    # Теперь state — "запутанное" состояние.
    # 3) Запускаем policy, пытаемся вернуться к цели
    steps_used = 0
    done = False

    # Превращаем state в тензор
    state_t = torch.tensor(state, dtype=torch.float32, device=device)

    for _ in range(max_steps):
        steps_used += 1

        # Получаем logits из policy
        logits = policy_net(state_t.unsqueeze(0))  # [1, n_actions]
        dist = Categorical(logits=logits)
        action = dist.sample()  # [1]
        
        # Делаем шаг в среде
        next_s, reward, done, info = env.step(action.item())
        
        # Проверяем, не достигли ли цели
        if done:
            return True, steps_used
        
        # Обновляем state_t
        state_t = torch.tensor(next_s, dtype=torch.float32, device=device)

    # Если за max_steps не дошли до цели, возвращаем неудачу
    return False, steps_used

def test_policy_multiple_runs(
    policy_net,
    env,
    num_runs=10,
    random_steps=10,
    max_steps=200,
    device='cpu'
):
    """
    Запускаем test_policy несколько раз, чтобы посчитать статистику успеха.
    """
    successes = 0
    total_steps = 0

    for i in range(num_runs):
        done, steps_used = test_policy(
            policy_net, env,
            k=random_steps,
            max_steps=max_steps,
            device=device
        )
        total_steps += steps_used
        if done:
            successes += 1

    success_rate = successes / num_runs
    avg_steps = total_steps / num_runs
    print(f"Success rate: {success_rate*100:.1f}% ({successes}/{num_runs})")
    print(f"Average steps used (including failures): {avg_steps:.1f}")

    return success_rate, avg_steps
