from traceback import print_tb
from kaggle_environments import make
from rl_model import Model

class Player():


    def __init__(self) -> None:
        self.model = Model()
        self.buffer = []
        self.TRESH = 100


    def make_move(self, observation):
        # print(observation)
        # move = model.get_move(observation, configuration)
        return 1  


    def save(self, old_obs, action, reward, new_obs):
        self.buffer.append([old_obs, action, reward, new_obs])
        
        if len(self.buffer) > self.TRESH:
            self.train()
            self.buffer = []


    def train(self):
        # Sample
        # Pass to network
        # Update
        pass


    
