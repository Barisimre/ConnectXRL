from kaggle_environments import make
from rl_player import *

configuration = {
    "rows": 6,
    "columns:": 7,
    "inarow": 4,
    "actTimeout": 10
}

env = make("connectx", configuration=configuration, debug=True)

def train():
    # Training agent in first position (player 1) against the default random agent.
    trainer = env.train([None, "random"])
    obs = trainer.reset()

    player = Player()

    for _ in range(1000):
        # env.render()

        if env.done:
            obs = trainer.reset()
        old_obs = obs
        action = player.make_move(obs)
        obs, reward, done, info = trainer.step(action)
        player.save(old_obs, action, reward, obs)
        print(reward, done)

train()
    
