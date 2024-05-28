import numpy as np # linear algebra
import struct
from array import array
from os.path  import join
import random
import matplotlib.pyplot as plt
import pandas as pd


#
# ABM data loader
#

def read_ABM_data(noise):
    ids = [[45,46,47,48,49],
           [270,271,272,273,274],
           [495,496,497,498,499],
           [720,721,722,723,724],
           [945,946,947,948,949]]
    experiments = []
    for experiment in ids:
        runs = []
        for run in experiment:
            sim_PCs = []
            for time in range(300, 404, 4):
                df = pd.read_csv(f"./data/ABM_Coordinates/simID{run}_time{time}_{noise}percentNoise.csv")
                X = df[np.logical_or(df["PointType"] == "Macrophage", df["PointType"] == "Noise")][["x", "y"]].values
                sim_PCs.append(X)
            runs.append(sim_PCs)
        experiments.append(runs)
    return experiments