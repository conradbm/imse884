# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 15:05:49 2020

@author: blakeconrad
"""

import pandas as pd
import numpy as np

df = pd.read_csv("DraftKingsdata_Cleaner.csv")

import os
import pyomo.environ as pyo
from pyomo.environ import *

from pyomo.opt import SolverFactory

opt = pyo.SolverFactory('glpk')

model = ConcreteModel()
