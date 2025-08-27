#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 17 19:04:51 2025

@author: geoff
"""
import numpy as np
import pandas as pd
import torch
from sqlalchemy import create_engine
from pathlib import Path
from three_state_network_solver import ThreeStateSolverNetwork

newthreestate = ThreeStateSolverNetwork("/home/geoff/Documents/SQL_Database/fleet1.sql",
                                        'trainingdata',1000,10)

newthreestate.trainmodel(20,(0,200))
#%%
# --- parameters ---
db_path = Path("/home/geoff/Documents/SQL_Database/fleet1.sql")
table_name = "trainingdata"
n = 100
k = 5
age_max = 25
bins = 20
bin_range = (0, 200)
rng = np.random.default_rng(42)

# assume we already have an instantiated network object
#   net = ThreeStateSolverNetwork(database_path=..., ...)

# --- read the database ---
engine = create_engine(f"sqlite:///{db_path.as_posix()}")
query = f"SELECT Result, Age, UnitID, Insult FROM {table_name};"
testingdataframe = pd.read_sql_query(query, engine)

# --- sample units and ages ---
unit_ids = testingdataframe["UnitID"].unique()
sampled_units = rng.choice(unit_ids, size=n, replace=False)

test_sequences = {}
for uid in sampled_units:
    g = testingdataframe[(testingdataframe["UnitID"] == uid) & (testingdataframe["Age"] < age_max)]
    ages = g["Age"].unique()
    if len(ages) < k:
        continue
    sampled_ages = rng.choice(ages, size=k, replace=False)
    sampled_ages.sort()

    feats = []
    for age in sampled_ages:
        timestep = g[g["Age"] == age]
        f, _ = newthreestate.encode_time_step(timestep, age, bins=bins, bin_range=bin_range)
        feats.append(torch.from_numpy(f).to(dtype=newthreestate.torch_dtype))
    test_sequences[uid] = torch.stack(feats, dim=0)
#%%
# now you can call:
preds = newthreestate.predict(test_sequences)