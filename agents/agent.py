
class Agent:
    def __init__(self, configuration):
        self.configuration = configuration
        self.EMPTY = 0
        pass

    def make_move(self, observation):
        pass

    def save(self, *args):
        pass

    def optimize(self):
        pass

    def update_networks(self):
        pass

    def renderer(self, board):
        columns = self.configuration.columns
        rows = self.configuration.rows

        def print_row(values, delim="|"):
            return f"{delim} " + f" {delim} ".join(str(v) for v in values) + f" {delim}\n"

        row_bar = "+" + "+".join(["---"] * columns) + "+\n"
        out = row_bar
        for r in range(rows):
            out = out + \
                  print_row(board[r * columns: r * columns + columns]) + row_bar
        print(out)