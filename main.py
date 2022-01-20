from kaggle_environments import make
from kaggle_environments import evaluate

from agents.rl_agent import *
from agents.human_agent import HumanAgent
from agents.bruteforce_agent import BruteforceAgent

from settings.configuration import Configuration
from itertools import count
from time import sleep
from tqdm import tqdm

# Configs
ROWS = 6
COLUMNS = 7
INAROW = 4
TIMEOUT = 10

configuration = Configuration(rows=ROWS, columns=COLUMNS, inarow=INAROW, actTimeout=TIMEOUT)
env = make("connectx", configuration=configuration)

def mean_reward(rewards):
    # return sum(r[0] if r[0] is not None else -1 for r in rewards) / len(rewards)
    return sum(rewards)/len(rewards)

def validate(agent, opponent):
    agent.eval()
    env.reset()
    print("Evaluate")
    rewards = evaluate('connectx', agents=[agent.make_move, opponent.make_move], configuration=configuration, num_episodes=100, debug=False)
    # print(rewards)
    # losses = len(r[0]<0 for r in rewards)
    rewards = np.array([r[0] for r in rewards])
    illegals = sum(rewards == None)
    rewards[rewards == None] = -1
    losses = sum(rewards == -1) - illegals
    ties = sum(rewards == 0)
    wins = sum(rewards == 1)
    # losses = sum([1 if r[0] is not None and r[0] < 0 else 0 for r in rewards])
    # ties = sum([1 if r[0] is not None and r[0] == 0 else 0 for r in rewards])
    # illegals = sum([1 if r[0] is None else 0 for r in rewards])
    # wins = sum([1 if r[0] is not None and r[0] > 0 else 0 for r in rewards])
    print(f'amount of wins, ties, illegals, losses: {(wins, ties, illegals, losses)}')
    print(f'winrate: {np.round(wins/(wins + ties + illegals + losses), decimals=2) * 100}%')
    print(f'mean reward: {mean_reward(rewards)}')


def play_against(agent):
    agent.eval()
    human = HumanAgent(configuration, validate_moves=True)
    while True:
        done = False
        trainer = env.train([None, agent.make_move])
        obs = trainer.reset()
        while not done:
            action = human.make_move(obs, configuration)
            obs, reward, done, info = trainer.step(action)
            if done:
                agent.renderer(obs.board)
                print(f'you {"lost" if reward == -1 else ("won" if reward == 1 else "tied")}')
                sleep(3)
                print()
                print()


def train(opponent):
    # Training agent in first position (player 1) against the default random agent.

    # player = Player()
    # agent = HumanAgent(configuration)
    agent = RLAgent(configuration)

    trainer = env.train([None, opponent.make_move])  # we might need to randomize the order
    obs = trainer.reset()
    done = False

    STEPS = 3000
    TARGET_UPDATE = 1000
    steps_in_episode = 0
    while True:
        for step in tqdm(range(STEPS)):
            if done:
                # print(f"Episode steps: {steps_in_episode}")
                # print(f'Reward: {reward}')
                # print(agent.renderer(obs.board))
                steps_in_episode = 0
                obs = trainer.reset()
            steps_in_episode += 1
            old_obs = obs
            action = agent.make_move(obs, configuration)
            obs, reward, done, info = trainer.step(action)
            state = obs.board
            old_state = old_obs.board
            if done:
                if reward is None:
                    reward = -1
                    state = None
            assert reward is not None
            agent.save(old_state, action, reward, state)
            agent.optimize()
            if step % TARGET_UPDATE == 0:
                agent.update_networks()
                # print('updating network')
        validate(agent, opponent)
    return agent

        # print(reward, doner)


if __name__ == '__main__':
    opponent = BruteforceAgent(configuration, depth=2)
    agent = train(opponent)
    validate(agent, opponent)
    # play_against(opponent)
