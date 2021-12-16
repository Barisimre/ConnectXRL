from kaggle_environments import make

env = make("connectx", {"rows": 10, "columns": 8, "inarow": 5}, debug=True)

env.run([None, "random"])

