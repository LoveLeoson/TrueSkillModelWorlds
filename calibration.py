import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import sys
import pickle 

def load_teams(filename):
    with open(filename, 'rb') as f:
        teams = pickle.load(f)
    return teams

def calibration_A(regions, MSI):
    for region in regions:
        team_counter = 0
        correction = 0
        for team in region.keys():
            if team in MSI.keys():
                difference = MSI[team][0] - region[team][0]
                team_counter += 1
                correction += difference
        correction /= team_counter
        for team in region.keys():
            region[team][0] += correction
    return regions

def calibration_B(regions, MSI):
    for region in regions:
        team_counter = 0
        correction = 0
        for team in region.keys():
            if team in MSI.keys():
                correction += MSI[team][0]
                team_counter += 1
        correction /= team_counter
        for team in region.keys():
            region[team][0] += correction
    return regions

def merge_regions(regions):
    merged_regions = {}
    for region in regions:
        for team in region.keys():
            if team not in merged_regions.keys():
                merged_regions[team] = region[team]
            else:
                merged_regions[team][0] += region[team][0]
                merged_regions[team][1][0] += region[team][1][0]
                merged_regions[team][1][1] += region[team][1][1]
    return merged_regions

def normalize_regions(regions):
    for region in regions:
        max_mean, min_mean = None, None
        for team in region.keys():
            if max_mean is None or region[team][0] > max_mean:
                max_mean = region[team][0]
            if min_mean is None or region[team][0] < min_mean:
                min_mean = region[team][0]
        nomalizing_value = (max_mean - min_mean)
        for team in region.keys():
            region[team][0] = 6*((region[team][0] - min_mean)/nomalizing_value)-3
            region[team][1] = 6*(region[team][1])/nomalizing_value
    return regions

def scale_dictionary_values(dictionary, weight):
    return {key: [value[0] * weight, value[1]] for key, value in dictionary.items()}


LEC_teams = load_teams("LEC.pkl")
LCS_teams = load_teams("LCS.pkl")
LCK_teams = load_teams("LCK.pkl")
LPL_teams = load_teams("LPL.pkl")
CBLOL_teams = load_teams("CBLOL.pkl")
LLA_teams = load_teams("LLA.pkl")
PCS_teams = load_teams("PCS.pkl")
VCS_teams = load_teams("VCS.pkl")
MSI_teams = load_teams("MSI.pkl")
Worlds2023_teams = load_teams("Worlds2023.pkl")
regions = [LEC_teams, LCS_teams, LCK_teams, LPL_teams,
           CBLOL_teams, LLA_teams, PCS_teams, VCS_teams]

weights = [0.8, 0.8, 1, 1, 0.6, 0.6, 0.7, 0.7] # different regions have different skill

# Iterate through regions and scale each dictionary using its corresponding weight
scaled_regions = []
for region, weight in zip(regions, weights):
    scaled_region = scale_dictionary_values(region, weight)
    scaled_regions.append(scaled_region)

regions = scaled_regions

regions = normalize_regions(regions)

updated_regions = calibration_A(regions, Worlds2023_teams)

updated_regions = calibration_A(regions, MSI_teams)

all_teams = merge_regions(updated_regions)

world_teams = ["G2 Esports", "Fnatic", "MAD Lions KOI", "Hanwha Life eSports",
                "Gen.G eSports", "Dplus KIA", "T1", "Bilibili Gaming", "Top Esports",
                "LNG Esports", "Weibo Gaming", "FlyQuest", "Team Liquid",
                "100 Thieves", "PSG Talon", "SoftBank Hawks Gaming",
                "GAM Esports", "Vikings Esports", "Movistar R7"]

max_length = max(len(team) for team in all_teams)

export_teams = {}
for team, values in sorted(all_teams.items(), key=lambda item: item[1][0], reverse=True):
    if team in world_teams:
        export_teams[team] = values
        print(f"{team}:{' ' * (max_length - len(team) + 1)} mean = {round(values[0], 2)}, var = {round(values[1], 2)}")

# Save export_teams as a pickle file
with open("world_teams.pkl", "wb") as f:
    pickle.dump(export_teams, f)