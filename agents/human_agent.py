from agents.agent import Agent


class HumanAgent(Agent):
    def __init__(self, configuration):
        super().__init__()
        self.config = configuration

    def make_move(self, observation):
        self.renderer(observation['board'])
        print(f"Playing for player {observation['mark']}")
        columns = self.config.columns
        rows = self.config.rows

        valid_move = False
        move = -1
        while not valid_move:
            move = int(input(f'Please provide a column between 0 and {self.config.columns - 1}\n'))
            location = max([r for r in range(rows) if observation['board'][move + (r * columns)] == self.EMPTY])
            if 0 <= move < self.config.columns and observation['board'][location] == self.EMPTY:
                valid_move = True
        return move

    def renderer(self, board):
        columns = self.config.columns
        rows = self.config.rows

        def print_row(values, delim="|"):
            return f"{delim} " + f" {delim} ".join(str(v) for v in values) + f" {delim}\n"

        row_bar = "+" + "+".join(["---"] * columns) + "+\n"
        out = row_bar
        for r in range(rows):
            out = out + \
                  print_row(board[r * columns: r * columns + columns]) + row_bar
        print(out)