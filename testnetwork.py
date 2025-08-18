#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 17 19:04:51 2025

@author: geoff
"""

from three_state_network_solver import ThreeStateSolverNetwork

newthreestate = ThreeStateSolverNetwork("/home/geoff/Documents/SQL_Database/fleet1.sql",
                                        'trainingdata',1000,10)

newthreestate.trainmodel(20,(0,200))
