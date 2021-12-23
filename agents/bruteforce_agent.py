from rl_player import Player
from kaggle_environments.envs.connectx.connectx import is_win, play
from random import choice

class BruteforceAgent:
    def __init__(self, configuration, depth=4):
        # super().__init__()
        self.max_depth = depth
        self.config = configuration
        self.EMPTY = 0

    def make_move(self, observation, _):
        return self.negamax_agent(observation)

    def negamax_agent(self, obs):

        columns = self.config.columns
        rows = self.config.rows
        size = rows * columns

        def negamax(board, mark, depth):
            moves = sum(1 if cell != self.EMPTY else 0 for cell in board)

            # Tie Game
            if moves == size:
                return 0, None

            # Can win next.
            for column in range(columns):
                if board[column] == self.EMPTY and is_win(board, column, mark, self.config, False):
                    return (size + 1 - moves) / 2, column

            # Recursively check all columns.
            best_score = -size
            best_column = None
            for column in range(columns):
                if board[column] == self.EMPTY:
                    # Max depth reached. Score based on cell proximity for a clustering effect.
                    if depth <= 0:
                        row = max(
                            [
                                r
                                for r in range(rows)
                                if board[column + (r * columns)] == self.EMPTY
                            ]
                        )
                        score = (size + 1 - moves) / 2
                        if column > 0 and board[row * columns + column - 1] == mark:
                            score += 1
                        if (
                                column < columns - 1
                                and board[row * columns + column + 1] == mark
                        ):
                            score += 1
                        if row > 0 and board[(row - 1) * columns + column] == mark:
                            score += 1
                        if row < rows - 2 and board[(row + 1) * columns + column] == mark:
                            score += 1
                    else:
                        next_board = board[:]
                        play(next_board, column, mark, self.config)
                        (score, _) = negamax(next_board,
                                             1 if mark == 2 else 2, depth - 1)
                        score = score * -1
                    if score > best_score or (score == best_score and choice([True, False])):
                        best_score = score
                        best_column = column

            return best_score, best_column

        _, column = negamax(obs['board'][:], obs['mark'], self.max_depth)
        if column is None:
            column = choice([c for c in range(columns) if obs['board'][c] == self.EMPTY])
        return column
