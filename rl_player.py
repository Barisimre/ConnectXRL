from kaggle_environments import make
from rl_model import Model

class Player():


    def __init__(self) -> None:
        self.model = Model()
        self.training = True
        self.EMPTY = 0



    def make_move(self, observation):
        # print(observation)
        # move = model.get_move(observation, configuration)
        return 1  


    def save(self, old_obs, action, reward, new_obs):
        if self.training:
            self.model.save(old_obs, action, reward, new_obs)
        

    
