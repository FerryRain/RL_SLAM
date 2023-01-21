import os

import numpy as np
import torch

from Policy.env import GazeboEnv
from Policy.TD3_Policy import TD3
from Policy.replay_buffer import ReplayBuffer
import argparse

def evaluate(network, epoch, eval_episodes=10):
    avg_reward = 0.0
    col = 0
    for _ in range(eval_episodes):
        count = 0
        state = env.reset()
        done = False
        while not done and count < 501:
            action = network.get_action(np.array(state))
            a_in = [(action[0] + 1) / 2, action[1]]
            state, reward, done, _ = env.step(a_in)
            avg_reward += reward
            count += 1
            if reward < -90:
                col += 1
    avg_reward /= eval_episodes
    avg_col = col / eval_episodes
    print("..............................................")
    print(
        "Average Reward over %i Evaluation Episodes, Epoch %i: %f, %f"
        % (eval_episodes, epoch, avg_reward, avg_col)
    )
    print("..............................................")
    return avg_reward

def args_set():
    """
    Initialization Parameter
    Returns:
        Args:
    """
    #Yolo Part
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='best.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', default='false', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')

    return parser.parse_args()

if __name__ == '__main__':
    args = args_set()
    env = GazeboEnv(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda or cpu
    eval_freq = 5e2  # After how many steps to perform the evaluation
    max_ep = 500  # maximum number of steps per episode
    eval_ep = 10  # number of episodes for evaluation
    max_timesteps = 5e6  # Maximum number of steps to perform
    expl_noise = 0.3  # Initial exploration noise starting value in range [expl_min ... 1]
    expl_decay_steps = (
        1500  # Number of steps over which the initial exploration noise will decay over
    )
    expl_min = 0.05  # Exploration noise after the decay in range [0...expl_noise]
    batch_size = 128  # Size of the mini-batch
    discount = 0.99999  # Discount factor to calculate the discounted future reward (should be close to 1)
    tau = 0.005  # Soft target update variable (should be close to 0)
    policy_noise = 1  # Added noise for exploration
    noise_clip = 5  # Maximum clamping values of the noise
    policy_freq = 4  # Frequency of Actor network updates
    buffer_size = 1e6  # Maximum size of the buffer
    file_name = "TD3_velodyne"  # name of the file to store the policy
    save_model = True  # Weather to save the model or not
    load_model = False  # Weather to load a stored model

    if not os.path.exists("../results"):
        os.makedirs("../results")
    if save_model and not os.path.exists("../pytorch_models"):
        os.makedirs("../pytorch_models")

    # Create the network
    state_dim = 5
    action_dim = 2
    max_action = 1
    network = TD3(state_dim, action_dim, max_action, device)
    replay_buffer = ReplayBuffer(buffer_size, args.seed)

    if load_model:
        try:
            network.load(file_name, "../pytorch_models")
            print("load sucessfully!")
        except:
            print(
                "Could not load the stored model parameters, initializing training with random parameters"
            )

    # TRAINING
    evaluations = []

    timestep = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True
    epoch = 1

    count_rand_actions = 0
    random_action = []

    # Begin training
    while timestep < max_timesteps:
        try:
            if done:
                if timestep != 0:
                    network.train(
                        replay_buffer,
                        episode_timesteps,
                        batch_size,
                        discount,
                        tau,
                        policy_noise,
                        noise_clip,
                        policy_freq,
                    )

                if timesteps_since_eval >= eval_freq:
                    print("Validating")
                    timesteps_since_eval %= eval_freq
                    evaluations.append(
                        evaluate(network=network, epoch=epoch, eval_episodes=eval_ep)
                    )
                    network.save(file_name, directory="../pytorch_models")
                    np.save("../results/%s" % (file_name), evaluations)
                    epoch += 1

                print("reset")
                state = env.reset()
                done = False

                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

            if expl_noise > expl_min:
                expl_noise = expl_noise - ((1 - expl_min) / expl_decay_steps)

            action = network.get_action(np.array(state))
            action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(
                -max_action, max_action
            )



            # Update action to fall in range [0,1] for linear velocity and [-1,1] for angular velocity
            # a_in = [action[0] * 5, action[1] * 5]
            next_state, reward, done, target = env.step(action)
            done_bool = 0 if episode_timesteps + 1 == max_ep else int(done)
            done = 1 if episode_timesteps + 1 == max_ep else int(done)
            episode_reward += reward

            # Save the tuple in replay buffer
            replay_buffer.add(state, action, reward, done_bool, next_state)

            # Update the counters
            state = next_state
            episode_timesteps += 1
            timestep += 1
            timesteps_since_eval += 1
        except:
            print("Error")
            pass

    if save_model:
        network.save("%s" % file_name, directory="../models")
    np.save("../results/%s" % file_name, evaluations)
    evaluations.append(evaluate(network=network, epoch=epoch, eval_episodes=eval_ep))

