from kaggle_environments import make
import random
from settings.configuration import Configuration

class Environment:
    def __init__(self, configuration, player_nr=0):
        self.config = configuration
        self.training_agent = configuration.training_agent
        self.env = make("connectx", configuration=configuration, debug=True)
        self.trainer = self.env.train([None, self.training_agent, None][player_nr:player_nr+2])

        # Initial values
        self._seed = -1
        self.rand = random.Random(self._seed)
        self.max_episode_steps = -1
        for i in range(len(self.env.state)-1, -1, -1):
            if 'board' in self.env.state[i]['observation']:
                self.obs = self.env.state[i]['observation']
                break
        else:
            print("using default observation")
            self.obs = {'board': [0]*(self.config.rows*self.config.columns)}

        self.action_space_max_value = self.config.columns - 1
        self.observation_space_dim = self.config.columns * self.config.rows
        self.action_space_dim = self.config.columns

    def reset(self):
        # print(self.board_to_string())
        self.trainer.reset()
        for i in range(len(self.env.state) - 1, -1, -1):
            if 'board' in self.env.state[i]['observation']:
                self.obs = self.env.state[i]['observation']
                break
        else:
            print("using default observation")
            self.obs = {'board': [0] * (self.config.rows * self.config.columns)}
        return self.obs['board']

    def step(self, action):
        """proceed"""
        obs, reward, done, info = self.trainer.step(action)
        self.obs = obs
        return obs['board'], reward, done, info

    def seed(self, seed):
        """Set the seed used for randomness"""
        self._seed = seed
        self.rand = random.Random(seed)

    def action_space(self):
        """An array containing all valid actions"""
        cols = self.config.columns
        board = self.obs['board']
        return [i for i in range(cols) if board[i] == 0]


    def action_space_sample(self):
        """return a random valid action"""
        # get valid actions
        # sample from array
        return self.rand.choice(self.action_space())

    #     action_dim = env.action_space.shape[0]
    def action_space_shape(self):
        """nr of valid actions at this time"""
        return self.action_space_dim

    #  state_dim = env.observation_space.shape[0]
    def observation_space_shape(self):
        """nr of observations """
        return self.observation_space_dim

    def action_space_max(self):
        """The maximum valid action"""
        return self.action_space_max_value

    def board_to_string(self):
        board2d = [self.obs['board'][i*self.config.columns:(i+1)*self.config.columns] for i in range(self.config.columns)]
        return "|"+"|\n|".join("".join("X" if a == 1 else ("O" if a == 2 else " ") for a in b) for b in board2d)[:-1]

configuration = Configuration(rows=6, columns=7, inarow=4, actTimeout=10)
env = Environment(configuration=configuration)
env.step(0)



