from kaggle_environments import make
from rl_player import *

env = make("connectx", debug=True)

# Training agent in first position (player 1) against the default random agent.
trainer = env.train([None, "random"])
done = False
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
    
