import argparse
import torch
from pilgrim.trainer_ppo import PPOTrainer
from pilgrim.model_policy import PilgrimPolicy
from pilgrim.model_value import PilgrimValue

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train PPO Model")
    
    # Add arguments for training configuration
    parser.add_argument("--n", type=int, required=True, help="Size of the permutation")
    parser.add_argument("--total_iterations", type=int, default=100, help="Total number of training iterations")
    parser.add_argument("--save_interval", type=int, default=10, help="Interval for saving models")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for training")
    parser.add_argument("--policy_lr", type=float, default=1e-4, help="Learning rate for the policy network")
    parser.add_argument("--value_lr", type=float, default=1e-4, help="Learning rate for the value network")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lambd", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--eps_clip", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--value_loss_coef", type=float, default=0.5, help="Value loss coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Max gradient norm for clipping")
    parser.add_argument("--rollout_size", type=int, default=2000, help="Rollout size")
    parser.add_argument("--mini_batch_size", type=int, default=128, help="Mini-batch size")
    parser.add_argument("--ppo_epochs", type=int, default=10, help="Number of PPO epochs")
    parser.add_argument("--max_steps_start", type=int, default=10, help="Initial max steps for curriculum")
    parser.add_argument("--max_steps_final", type=int, default=100, help="Final max steps for curriculum")
    parser.add_argument("--curriculum_steps", type=int, default=10, help="Number of steps for curriculum")
    parser.add_argument("--save_dir", type=str, default="ppo_saves", help="Directory to save models")

    args = parser.parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Initialize policy and value networks
    policy_net = PilgrimPolicy(state_size=args.n, n_actions=args.n)
    value_net = PilgrimValue(state_size=args.n)

    # Initialize PPO trainer
    trainer = PPOTrainer(
        policy_net=policy_net,
        value_net=value_net,
        n=args.n,
        device=device,
        gamma=args.gamma,
        lambd=args.lambd,
        eps_clip=args.eps_clip,
        lr_policy=args.policy_lr,
        lr_value=args.value_lr,
        entropy_coef=args.entropy_coef,
        value_loss_coef=args.value_loss_coef,
        max_grad_norm=args.max_grad_norm,
        rollout_size=args.rollout_size,
        mini_batch_size=args.mini_batch_size,
        ppo_epochs=args.ppo_epochs,
        max_steps_start=args.max_steps_start,
        max_steps_final=args.max_steps_final,
        curriculum_steps=args.curriculum_steps,
        save_dir=args.save_dir
    )

    # Run training
    trainer.run(total_iterations=args.total_iterations, save_interval=args.save_interval)

if __name__ == "__main__":
    main()





