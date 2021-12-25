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
    # agent = HumanAgent(configuration)
    agent = RLAgent(configuration)

    trainer = env.train([None, 'random']) # we might need to randomize the order
    obs = trainer.reset()
    state = obs.board

    STEPS = 10000
    TARGET_UPDATE = 100
    steps_in_episode = 0
    for step in range(STEPS):
        if env.done:
            print(f"Episode steps: {steps_in_episode}")
            print(f'Reward: {reward}')
            # print(agent.renderer(obs.board))
            steps_in_episode = 0
            obs = trainer.reset()
            state = obs.board
        steps_in_episode += 1
        old_state = state
        action = agent.make_move(obs)
        obs, reward, done, info = trainer.step(action)
        state = obs.board
        if state == old_state:
            reward = -1  # punish the rl agent for illegal moves.
            state = None
        agent.save(old_state, action, reward, state)
        agent.optimize()
        if step % TARGET_UPDATE == 0:
            agent.update_networks()
            print('updating network')

        #print(reward, done)


if __name__ == '__main__':
    train()
    
