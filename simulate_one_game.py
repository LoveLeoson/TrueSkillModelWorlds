import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import sys
import pickle 

teams = pickle.load(open("world_teams.pkl", "rb"))

team1 = "" # define first team
team2 = "" # define second team

sigma = 3
beta = 10

class Normal_Distribution():
    def __init__(self, mean, variance):
        self.mean = mean
        self.variance = variance

    def pdf(self, x):
        return 1/(np.sqrt(2*np.pi)*np.sqrt(self.variance))*np.exp(-(x-self.mean)**2/(2*self.variance))
    
    def generate_samples(self, num_samples = 1):
        if len(self.mean) == 1:
            return (np.random.normal(self.mean, np.sqrt(self.variance), num_samples)).tolist()

        else:
            return (np.random.multivariate_normal(self.mean, self.variance, num_samples)).tolist()

for i in range(1000):
    p_t = Normal_Distribution([teams[team1][0] - teams[team2][0]], beta + teams[team1][1] - teams[team2][1])
    t_sample = p_t.generate_samples()[0]
    if t_sample > 0:
        y_predict = 1
    elif t_sample < 0:
        y_predict = -1