from kaggle_environments import make

from rl_player import *
from agents.bruteforce_agent import BruteforceAgent
from agents.human_agent import HumanAgent

from settings.configuration import Configuration

configuration = Configuration(rows=6, columns=7, inarow=4, actTimeout=10)
env = make("connectx", configuration=configuration, debug=True)

def train():
    # Training agent in first position (player 1) against the default random agent.
    trainer = env.train([None, "random"])
    obs = trainer.reset()

    # player = Player()
    # player = BruteforceAgent(configuration, depth=2)
    player = HumanAgent(configuration)

    for _ in range(100):
        # env.render()

        if env.done:
            obs = trainer.reset()
        old_obs = obs
        action = player.make_move(obs)
        obs, reward, done, info = trainer.step(action)
        player.save(old_obs, action, reward, obs)
        if done:
            print(reward)
        #print(reward, done)

train()
    
