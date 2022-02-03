from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import os


def roll(xs, r):
    res = []
    ma = xs[0]
    for x in xs:
        ma = r*ma + (1-r)*x
        res.append(ma)
    return res


def make_winrate_graph(filename):

    time_name = filename.split("/")[2]
    treshold = filename.split('_')[1][:3]

    evals = np.load(filename, allow_pickle=True)
    winrates = [round(a[1]/(a[1]+a[2]+a[3])*100, 3) for a in evals[1:]]
    lines = [i*1000 for i, j in enumerate(winrates) if j>float(treshold)*100]

    winrates = roll(winrates, 0.5)
    
    plt.plot(range(0, len(winrates)*1000, 1000), winrates)
    for l in lines:
        plt.axvline(x=l, color='r')
    print(len(winrates))
    plt.title(f"Winrate over training steps, using TD3, treshold: {treshold}")
    plt.xlabel("Steps in training")
    plt.ylabel("Winrate")
    plt.legend(["winrate", "transition moment"])
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.savefig(f"Figures/{time_name}.png")
    plt.clf()

def make_all_winrate_graphs():
    ls = os.listdir("results/TD3")
    for l in ls:
        make_winrate_graph(f"results/TD3/{l}/evals.npy")

make_all_winrate_graphs()