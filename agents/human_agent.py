from agents.agent import Agent


class HumanAgent(Agent):
    def __init__(self, configuration):
        super().__init__()
        self.validate_moves = False

    def make_move(self, observation):
        self.renderer(observation['board'])
        print(f"Playing for player {observation['mark']}")
        columns = self.configuration.columns
        rows = self.configuration.rows

        valid_move = False
        move = -1
        while not valid_move:
            move = int(input(f'Please provide a column between 0 and {self.configuration.columns - 1}\n'))
            if self.validate_moves:
                empty_in_column = [move + r * columns for r in range(rows) if observation['board'][move + r * columns] == self.EMPTY]
                if len(empty_in_column) != 0:
                    location = max(empty_in_column)
                    if 0 <= move < self.configuration.columns and observation['board'][location] == self.EMPTY:
                        valid_move = True
            else:
                valid_move = True
        return move

