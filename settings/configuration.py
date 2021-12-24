# can make this dataclass but need to update my python

class Configuration:
    def __init__(self, columns, rows, inarow, actTimeout, training_agent="random"):
        self.columns = columns
        self.rows = rows
        self.inarow = inarow
        self.actTimeout = actTimeout
        self.training_agent = training_agent