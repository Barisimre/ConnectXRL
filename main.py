from kaggle_environments import make, agent

# System declaration
configuration = {
    "rows": 5,
    "columns":5,
    "inarow": 3,
    "actTimeout": 10
}

def step(env, custom_agent):
    trainer = env.train([None, "negamax"])

    obs = trainer.reset()
    for _ in range(1000):
        agent1_action = custom_agent.act(env.state[0].observation)
        obs, reward, done, info = trainer.step(agent1_action[0])
        print(reward)
        if done:
            obs = trainer.reset()

env = make("connectx", configuration=configuration)
custom_agent = agent.Agent("random", env)
step(env, custom_agent)


