#import _init_paths
import argparse
import time
import datetime
import os
import torch
import torch_ac
import tensorboardX
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from collections import deque

import utils
from model import ACModel, QModel
from gym_minigrid.wrappers import FullyObsWrapper


# Parse arguments

parser = argparse.ArgumentParser()

## General parameters
parser.add_argument("--algo", required=True,
                    help="algorithm to use: a2c | ppo (REQUIRED)")
parser.add_argument("--env", required=True,
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--model", default=None,
                    help="name of the model (default: {ENV}_{ALGO}_{TIME})")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--save-interval", type=int, default=1,
                    help="number of updates between two saves (default: 10, 0 means no saving)")
parser.add_argument("--procs", type=int, default=16,
                     help="number of processes (default: 16)")
parser.add_argument("--frames", type=int, default=10**7,
                    help="number of frames of training (default: 1e7)")
parser.add_argument("--episodes", type=int, default=5000,
                    help="number of episodes of training")

## Parameters for main algorithm
parser.add_argument("--epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--batch-size", type=int, default=128,
                    help="batch size for PPO (default: 256)")
parser.add_argument("--frames-per-proc", type=int, default=None,
                    help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--lr", type=float, default=0.001,
                    help="learning rate (default: 0.001)")
parser.add_argument("--gae-lambda", type=float, default=0.95,
                    help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
parser.add_argument("--entropy-coef", type=float, default=0.01,
                    help="entropy term coefficient (default: 0.01)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--optim-eps", type=float, default=1e-8,
                    help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
parser.add_argument("--optim-alpha", type=float, default=0.99,
                    help="RMSprop optimizer alpha (default: 0.99)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--recurrence", type=int, default=1,
                    help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model to handle text input")
parser.add_argument("--max-memory", type=int, default=100000,
                    help="Maximum experiences stored (default: 100000)")
parser.add_argument("--update-interval", type=int, default=100,
                    help="update frequece of target network (default: 1000)")

args = parser.parse_args()

args.mem = args.recurrence > 1

# Set run dir

date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
default_model_name = f"{args.env}_{args.algo}_seed{args.seed}_{date}"

model_name = args.model or default_model_name
model_dir = utils.get_model_dir(model_name)

# Load loggers and Tensorboard writer

txt_logger = utils.get_txt_logger(model_dir)
csv_file, csv_logger = utils.get_csv_logger(model_dir)
tb_writer = tensorboardX.SummaryWriter(model_dir)

# Log command and all script arguments

txt_logger.info("{}\n".format(" ".join(sys.argv)))
txt_logger.info("{}\n".format(args))

# Set seed for all randomness sources

utils.seed(args.seed)

# Set device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
txt_logger.info(f"Device: {device}\n")

# Load environments
#
# envs = []
# for i in range(args.procs):
#     envs.append(utils.make_env(args.env, args.seed + 10000 * i))
envs = []
for i in range(args.procs):
    env = FullyObsWrapper(utils.make_env(args.env, args.seed + 10000 * i))
    envs.append(env)
txt_logger.info("Environments loaded\n")

# Load training status

try:
    status = utils.get_status(model_dir)
except OSError:
    status = {"num_frames": 0, "update": 0, 'num_episodes': 0}
txt_logger.info("Training status loaded\n")

# Load observations preprocessor

obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)
if "vocab" in status:
    preprocess_obss.vocab.load_vocab(status["vocab"])
txt_logger.info("Observations preprocessor loaded")

# Load model

if args.algo == 'dqn':
    policy_network = QModel(obs_space, envs[0].action_space).to(device)
    target_network = QModel(obs_space, envs[0].action_space).to(device)
    model = policy_network
else:
    model = ACModel(obs_space, envs[0].action_space, args.mem, args.text).to(device)
if "model_state" in status:
    model.load_state_dict(status["model_state"])
if "target_state" in status:
    target_network.load_state_dict(status["target_state"])
txt_logger.info("Model loaded\n")
txt_logger.info("{}\n".format(model))

# Load algo

if args.algo == "a2c":
    algo = torch_ac.A2CAlgo(envs, model, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                            args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                            args.optim_alpha, args.optim_eps, preprocess_obss)
elif args.algo == "ppo":
    algo = torch_ac.PPOAlgo(envs, model, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                            args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                            args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss)
elif args.algo == 'dqn':
    algo = torch_ac.DQNAlgo(envs[0], policy_network, target_network, device, args.max_memory,
              args.discount, args.lr, args.update_interval, args.batch_size,
              preprocess_obss)
else:
    raise ValueError("Incorrect algorithm name: {}".format(args.algo))

if "optimizer_state" in status:
    algo.optimizer.load_state_dict(status["optimizer_state"])
txt_logger.info("Optimizer loaded\n")

# Train model

num_frames = status["num_frames"]
num_episodes = status["num_episodes"]

update = status["update"]
start_time = time.time()

# Track all success rates for plotting
all_success_history = []
# Track recent 100 episodes for rolling success rate
recent_success_window = deque(maxlen=100)

while num_episodes < args.episodes:
    # Update model parameters

    update_start_time = time.time()
    if args.algo == "dqn":
        logs = algo.collect_experiences()
    else:
        exps, logs1 = algo.collect_experiences()
        logs2 = algo.update_parameters(exps)
        logs = {**logs1, **logs2}
    update_end_time = time.time()

    num_frames += logs["num_frames"]
    if args.algo == 'dqn':
        num_episodes += 1
    else:
        num_episodes += logs['done']
    update += 1

    # Print logs
    if update % args.log_interval == 0:
        fps = logs["num_frames"]/(update_end_time - update_start_time)
        duration = int(time.time() - start_time)

        if args.algo == 'dqn':
            return_per_episode = utils.synthesize(logs["rewards"])

            header = ["update", "episodes", "frames", "FPS", "duration"]
            data = [update, num_episodes, num_frames, fps, duration]
            header += ["return_" + key for key in return_per_episode.keys()]
            data += return_per_episode.values()
            header += ["policy_loss"]
            data += [np.mean(logs["loss"])]
            data += [logs['won']]

            txt_logger.info(
            "U {} | E {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | pL {:.3f} | W: {}"
            .format(*data)
            )
        else:
            return_per_episode = utils.synthesize(logs["return_per_episode"])
            rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
            num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

            # Update success tracking
            if "success_per_episode" in logs:
                for success in logs["success_per_episode"]:
                    recent_success_window.append(success)
                    all_success_history.append(success)

                # Calculate recent 100 episodes success rate
                success_rate_100 = np.mean(recent_success_window) if len(recent_success_window) > 0 else 0
                # Calculate success rate for current batch
                success_rate_batch = np.mean(logs["success_per_episode"]) if len(logs["success_per_episode"]) > 0 else 0
            else:
                success_rate_100 = 0
                success_rate_batch = 0

            header = ["update", "episodes", "frames", "FPS", "duration"]
            data = [update, num_episodes, num_frames, fps, duration]
            header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
            data += rreturn_per_episode.values()
            header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
            data += num_frames_per_episode.values()
            header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
            data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]
            header += ["success_rate_100", "success_rate_batch"]
            data += [success_rate_100, success_rate_batch]

            txt_logger.info(
                "U {} | E {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f} | SR100 {:.1%} | SRb {:.1%}"
                .format(*data))

            header += ["return_" + key for key in return_per_episode.keys()]
            data += return_per_episode.values()

        if status["num_frames"] == 0:
            csv_logger.writerow(header)
        csv_logger.writerow(data)
        csv_file.flush()

        for field, value in zip(header, data):
            tb_writer.add_scalar(field, value, num_frames)

    # Save status

    if args.save_interval > 0 and update % args.save_interval == 0:
        if args.algo == 'dqn':
            status = {"num_frames": num_frames, "num_episodes": num_episodes, "update": update,
                  "model_state": policy_network.state_dict(), "target_state": target_network.state_dict(),
                  "optimizer_state": algo.optimizer.state_dict()}
        else:
            status = {"num_frames": num_frames, "num_episodes":num_episodes,"update": update,
                      "model_state": model.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
        if hasattr(preprocess_obss, "vocab"):
            status["vocab"] = preprocess_obss.vocab.vocab
        utils.save_status(status, model_dir)
        txt_logger.info("Status saved")

# Training finished - plot success rate curve
if args.algo != 'dqn' and len(all_success_history) > 0:
    txt_logger.info("\n" + "="*60)
    txt_logger.info("Training completed! Generating success rate plot...")

    # Calculate rolling average success rate (window size 100)
    def calculate_rolling_avg(data, window=100):
        if len(data) < window:
            return list(range(1, len(data)+1)), [np.mean(data[:i+1]) for i in range(len(data))]
        rolling_avg = []
        for i in range(len(data)):
            if i < window:
                rolling_avg.append(np.mean(data[:i+1]))
            else:
                rolling_avg.append(np.mean(data[i-window+1:i+1]))
        return list(range(1, len(data)+1)), rolling_avg

    episodes, rolling_success_rate = calculate_rolling_avg(all_success_history, window=100)

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot 1: Rolling average success rate
    plt.subplot(1, 2, 1)
    plt.plot(episodes, rolling_success_rate, linewidth=2, color='#2E86AB')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Success Rate (Rolling Avg 100)', fontsize=12)
    plt.title('Success Rate over Training', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)

    # Add horizontal line at 0.9 for reference
    plt.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='90% Success')
    plt.legend()

    # Plot 2: Success rate by bins (every 100 episodes)
    plt.subplot(1, 2, 2)
    bin_size = 100
    num_bins = len(all_success_history) // bin_size
    if num_bins > 0:
        binned_success = [np.mean(all_success_history[i*bin_size:(i+1)*bin_size])
                         for i in range(num_bins)]
        bin_centers = [(i+0.5)*bin_size for i in range(num_bins)]
        plt.bar(bin_centers, binned_success, width=bin_size*0.8,
               color='#A23B72', alpha=0.7, edgecolor='black')
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Success Rate (per 100 episodes)', fontsize=12)
        plt.title('Success Rate Distribution', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        plt.ylim(0, 1.05)

    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(model_dir, 'success_rate_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Print summary statistics
    final_100_success = np.mean(all_success_history[-100:]) if len(all_success_history) >= 100 else np.mean(all_success_history)
    overall_success = np.mean(all_success_history)

    txt_logger.info(f"\n{'='*60}")
    txt_logger.info("SUCCESS RATE SUMMARY")
    txt_logger.info(f"{'='*60}")
    txt_logger.info(f"Total Episodes: {len(all_success_history)}")
    txt_logger.info(f"Overall Success Rate: {overall_success:.2%}")
    txt_logger.info(f"Final 100 Episodes Success Rate: {final_100_success:.2%}")
    txt_logger.info(f"Success rate plot saved to: {plot_path}")
    txt_logger.info(f"{'='*60}\n")
