import numpy as np
import torch
import argparse
import os

import td3.utils as utils
import td3.TD3 as TD3
from td3.environment import Environment
from settings.configuration import Configuration
from agents.bruteforce_agent import BruteforceAgent
from scipy.special import softmax

configuration = Configuration(rows=6, columns=7, inarow=4, actTimeout=10)
agent = BruteforceAgent(configuration=configuration, depth=1)
configuration.training_agent = agent.make_move

current_policy_win = 0
current_policy_loss = 0

def select_action_from_distribution(env, dist):
    return int(np.argmax(dist))

    #
    # possible = env.action_space()
    # probs = np.array([])
    # indices = np.array([])
    # for prob, index in reversed(sorted((v, i) for i, v in enumerate(dist))):
    #     if index in possible:
    #         probs = np.append(probs, prob)
    #         indices = np.append(indices, int(index))
    #
    # probs = softmax(probs)
    # if len(probs) == 0:
    #     raise ValueError("no action was possible. Was the game finished already?")
    # return int(np.random.choice(indices, p=probs))



# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
    global current_policy_win, current_policy_loss
    eval_env = Environment(configuration=configuration)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            action = select_action_from_distribution(eval_env, action)

            state, reward, done, _ = eval_env.step(action)
            if done and reward is None:
                reward = -1
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    if current_policy_loss + current_policy_win > 0:
        print(f"Won {current_policy_win} and lost {current_policy_loss} => {(current_policy_win / (current_policy_loss + current_policy_win))}% winrate")
    print("---------------------------------------")

    current_policy_loss = 0
    current_policy_win = 0

    return avg_reward


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="HalfCheetah-v2")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=25e3, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args()

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    env = Environment(configuration=configuration)

    # Set seeds
    env.seed(args.seed)
    # env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Now variables in environment
    state_dim = env.observation_space_dim
    action_dim = env.action_space_dim
    max_action = env.action_space_max_value

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }

    # Initialize policy
    if args.policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = TD3.TD3(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    # Evaluate untrained policy
    evaluations = [eval_policy(policy, args.env, args.seed)]

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(int(args.max_timesteps)):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space_sample()
        else:
            action = (
                    policy.select_action(np.array(state))
                    + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(0, max_action) #.clip(-max_action, max_action)
            action = select_action_from_distribution(env, action)

        # Perform action
        next_state, reward, done, _ = env.step(action)
        if done and reward is None:
            reward = -1
        done_bool = float(done) if episode_timesteps < env.max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            # print(
            #    f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            if episode_reward > 0:
                current_policy_win += 1
            elif episode_reward < 0:
                current_policy_loss += 1
            else:
                # pass
                print("wow a tie", episode_reward)

            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, args.env, args.seed))
            np.save(f"./results/{file_name}", evaluations)
            if args.save_model: policy.save(f"./models/{file_name}")