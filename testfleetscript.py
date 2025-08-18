#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 17:31:32 2025

@author: geoff
"""
import numpy as np
import matplotlib.pyplot as plt

from fleetgen import Fleet
from three_state_network_solver import ThreeStateSolverNetwork



testFleet=Fleet(1000, 50, 10,  
                25,.1,
                0,0,(0,200),100,
                (4,0.2), (2,0.1),
                (4.7,0.3),0.5, 0.5)
testFleet.create_fleet()
testFleet.to_sql("/home/geoff/Documents/SQL_Database/fleet1.sql")
#
#%%
count = 0

for testfleetnum in range(10):
    testFleet.create_fleet()

    count=testFleet.fleetfailure+count
    print(f"{testfleetnum}")


#%%


failureage=np.array([testFleet.unit[u]['failedage'] for 
                     u in range(0,len(testFleet.unit))])


plt.hist(failureage[~np.isnan(failureage)])