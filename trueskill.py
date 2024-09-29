import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import sys
import pickle 

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
        
def truncGaussian(a, b, m0, s0): # generate a sample from a truncated Gaussian
    a_scaled , b_scaled = (a - m0)/np.sqrt(s0), (b - m0)/ np.sqrt(s0)
    s = sp.stats.truncnorm.rvs(a_scaled, b_scaled, loc = m0, scale = np.sqrt(s0), size=1)
    return s

def gibbs(s1, s2, y, score1, score2):
    mu1 = s1[0] # mean of player 1
    sigma1 = s1[1] # std of player 1
    mu2 = s2[0] # mean of player 2
    sigma2 = s2[1] # std of player 2
    mu_s1s2 = np.array([[mu1], [mu2]]) # joint mean of s1 and s2
    sigma_s1s2 = np.array([[sigma1, 0], [0, sigma2]]) # cov matrix of s1 and s2
    # Store samples
    s1_samples = []
    s2_samples = []

    # generate s1 and s2 for starting value for t
    s1 = Normal_Distribution([mu1],sigma1).generate_samples()[0]
    s2 = Normal_Distribution([mu2],sigma2).generate_samples()[0]
    t_given_s1_s2 = Normal_Distribution([s1 - s2], beta)
    t = t_given_s1_s2.generate_samples()[0]

    # calculate cov matrix for p(s1,s2|t)
    sigma_s1s2_given_t = inv(inv(sigma_s1s2) + A.T @ A * (1/beta))

    for i in range(n_iter):
        # calculate mean matrix for p(s1,s2|t)
        mu_s1s2_given_t = sigma_s1s2_given_t @ ( inv(sigma_s1s2) @ mu_s1s2 + A.T * 1/beta *(t - b))

        # sample s1 and s2 from p(s1,s2|t)
        [s1,s2] = Normal_Distribution(mu_s1s2_given_t.ravel(), sigma_s1s2_given_t).generate_samples()[0]   
        if y > 0:
            # if player one wins the game sample from N(t; s1-s2, beta) on 0 to inf
            [t] = truncGaussian(0, np.inf, s1-s2, beta) * sig(score1-score2)
        elif y < 0:
             # if player two wins the game sample from N(t; s1-s2, beta) on -inf to 0
            [t] = truncGaussian(np.NINF, 0, s1-s2, beta) * sig(score2-score1)

        # Store post burn-in samples
        if i >= burn_in:
            s1_samples.append(s1)
            s2_samples.append(s2)

    # calculate the means and std of s1 and s2 samples
    mean_s1, var_s1 = np.mean(s1_samples), np.var(s1_samples)
    mean_s2, var_s2 = np.mean(s2_samples), np.var(s2_samples)
    return [mean_s1, var_s1], [mean_s2, var_s2]

def sig(x):
    return abs(2*(1/(1+np.exp(1*-x))-0.5))+0.5


LEC = pd.read_csv("LEC.csv")
LCS = pd.read_csv("LCS.csv")
LCK = pd.read_csv("LCK.csv")
LPL = pd.read_csv("LPL.csv")
PCS = pd.read_csv("PCS.csv")
CBLOL = pd.read_csv("CBLOL.csv")
VCS = pd.read_csv("VCS.csv")
LLA = pd.read_csv("LLA.csv")
MSI = pd.read_csv("MSI.csv")
Worlds2023 = pd.read_csv("Worlds2023.csv")

LEC = LEC[::-1].reset_index(drop=True)
LCS = LCS[::-1].reset_index(drop=True)
LCK = LCK[::-1].reset_index(drop=True)
LPL = LPL[::-1].reset_index(drop=True)
PCS = PCS[::-1].reset_index(drop=True)
CBLOL = CBLOL[::-1].reset_index(drop=True)
VCS = VCS[::-1].reset_index(drop=True)
LLA = LLA[::-1].reset_index(drop=True)
MSI = MSI[::-1].reset_index(drop=True)
Worlds2023 = Worlds2023[::-1].reset_index(drop=True)

LEC_teams = {}
LCS_teams = {}
LCK_teams = {}
LPL_teams = {}
PCS_teams = {}
CBLOL_teams = {}
VCS_teams = {}
LLA_teams = {}
MSI_teams = {}
Worlds2023_teams = {}
regions = [Worlds2023, LEC, LCS, LCK, LPL, PCS, CBLOL, VCS, LLA, MSI]
region_teams = [Worlds2023_teams, LEC_teams, LCS_teams, LCK_teams, LPL_teams, PCS_teams,
                CBLOL_teams, VCS_teams, LLA_teams, MSI_teams]

mu = 0
sigma = 3
beta = 10
n_iter = 5000
burn_in = int(n_iter * 0.2)
inv = np.linalg.inv
A = np.array([[1], [-1]]).T
b = 0
for index, teams in enumerate(region_teams): 
    data = regions[index]
    length = len(data["team1"])
    for i in range(length):
        if (i*100//length)% 1 == 0:
            print(round(i*100/length), "%")
            sys.stdout.write("\033[F")

        # get the teams
        team1 = data["team1"][i]
        team2 = data["team2"][i]
        # if the teams not in the dictionairy add them with default values
        if team1 not in teams:
            teams[team1] = [mu, sigma]
        if team2 not in teams:
            teams[team2] = [mu, sigma]
    
    
        # if team1 wins the game
        if data["score1"][i] > data["score2"][i]:
            y = 1
    
        # if team2 wins the game
        elif data["score1"][i] < data["score2"][i]:
            y = -1

        # skip if draw
        else:
            continue
    
        # calculate new means and std for s1 and s2 and update the dictionary
        s1, s2 = gibbs(teams[team1], teams[team2], y, data["score1"][i], data["score2"][i])
        teams[team1] = [s1[0], s1[1]]
        teams[team2] = [s2[0], s2[1]]

names = ["Worlds2023", "LEC", "LCS", "LCK", "LPL", "PCS", "CBLOL", "VCS", "LLA", "MSI"]
for index, teams in enumerate(region_teams):
    f = open(names[index] + ".pkl", "wb")
    pickle.dump(teams, f)
    f.close()
    print(teams)
    sorted_teams= sorted(teams.items(), key=lambda x:x[1][0], reverse=True)
    fig, axs = plt.subplots(1, 5, figsize=(12, 6))  # 1 row, 3 columns
    colors = ["b", "r", "y", "g", "m"]

    print("The 5 teams with the highest skill level:")
    for i in range(5):
        print(f"{sorted_teams[i][0]}:  \t mean = {sorted_teams[i][1][0]}, var = {sorted_teams[i][1][1]}")
        samples = np.random.normal(sorted_teams[i][1][0], sorted_teams[i][1][1], 1000)
        axs[i].hist(samples, bins=50, label="s1", alpha=0.5, density=True, color=colors[i])
        axs[i].set_title(sorted_teams[i][0])

    plt.show()