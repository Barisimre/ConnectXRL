from kaggle_environments import make

from agents.rl_agent import *
from agents.human_agent import HumanAgent

from settings.configuration import Configuration
from itertools import count

# Configs
ROWS = 6
COLUMNS = 7
INAROW = 4
TIMEOUT = 10

configuration = Configuration(rows=ROWS, columns=COLUMNS, inarow=INAROW, actTimeout=TIMEOUT)
env = make("connectx", configuration=configuration, debug=True)

def train():
    # Training agent in first position (player 1) against the default random agent.


    # player = Player()
    # player = BruteforceAgent(configuration, depth=2)
    agent = HumanAgent(configuration)
    # agent = RLAgent(COLUMNS, ROWS)

    trainer = env.train([None, 'random']) # we might need to randomize the order
    obs = trainer.reset()

    STEPS = 10000
    TARGET_UPDATE = 100
    steps_in_episode = 0
    for step in range(STEPS):
        if env.done:
            print(f"Episode steps: {steps_in_episode}")
            steps_in_episode = 0
            obs = trainer.reset()
        steps_in_episode += 1
        old_obs = obs
        action = agent.make_move(obs)
        obs, reward, done, info = trainer.step(action)
        print(old_obs==obs)



        reward = 0 if reward is None else reward
        agent.save(old_obs.board, action, reward, obs.board)
        agent.optimize()
        if step % TARGET_UPDATE == 0:
            agent.update_networks()

        #print(reward, done)

train()
    
