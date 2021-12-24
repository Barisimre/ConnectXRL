# can make this dataclass but need to update my python

class Configuration(dict):
    def __init__(self, columns, rows, inarow, actTimeout, training_agent="random"):
        super(Configuration, self).__init__(columns=columns, rows=rows, inarow=inarow, actTiemout=actTimeout, training_agent=training_agent)
        self.columns = columns
        self.rows = rows
        self.inarow = inarow
        self.actTimeout = actTimeout
        self.training_agent = training_agent
