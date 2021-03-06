#!/usr/bin/env python
# coding: utf-8

# # HW2 Problem 5 Vehicle Routing

# In[3]:


import pandas as pd
import numpy as np

D =[[0,2457,712,1433,66,2141,1616,635,2407,1104,644,1167,1057],                     
[2457,0,1752,1374,2409,365,851,1853,958,2339,1817,1688,1775],               
[712,1752,0,954,672,1452,906,275,1737,1195,167,838,778],            
[1433,1374,954,0,1368,1010,871,829,1891,967,878,336,445],            
[66,2409,672,1368,0,2090,1572,577,2383,1047,593,1101,991],               
[2141,365,1452,1010,2090,0,593,1522,1111,1974,1498,1324,1412],                  
[1616,851,906,871,1572,593,0,1039,1033,1710,987,1078,1124],               
[635,1853,275,829,577,1522,1039,0,1956,920,108,633,550],            
[2407,958,1737,1891,2383,1111,1033,1956,0,2732,1874,2110,2151],               
[1104,2339,1195,967,1047,1974,1710,920,2732,0,1028,654,587],
[644,1817,167,878,593,1498,987,108,1874,1028,0,713,640],                  
[1167,1688,838,336,1101,1324,1078,633,2110,654,713,0,117],                     
[1057,1775,778,445,991,1412,1124,550,2151,587,640,117,0]]
df = pd.DataFrame(D)
df


# In[50]:


import os
import pyomo.environ as pyo
from pyomo.environ import *

from pyomo.opt import SolverFactory

opt = pyo.SolverFactory('glpk')

model = ConcreteModel()

model.X = Var(range(df.shape[0]), range(df.shape[1]), within=Binary, initialize=0)

model.obj = Objective(expr = sum([model.X[i,j]*df.values[i,j] for i in model.X_index_0.data() for j in model.X_index_1.data()]), 
                      sense=minimize)
model.constraints = ConstraintList()

for j in model.X_index_1.data():
    if j > 0:
        model.constraints.add(sum([model.X[i,j] for i in model.X_index_0.data()]) == 1)
    
for i in model.X_index_0.data():
    if i > 0:
        model.constraints.add(sum([model.X[i,j] for j in model.X_index_1.data()]) == 1)
    
for j in model.X_index_1.data():
    for i in model.X_index_0.data():
        if i == j:
            model.constraints.add(model.X[i,j] == 0)

model.constraints.add(sum([model.X[i,0] for i in model.X_index_0.data()]) == 2)

model.constraints.add(sum([model.X[0,j] for j in model.X_index_1.data()]) == 2)


# # Solution

# In[51]:


instance = model.create_instance()
results = opt.solve(instance, tee=True)
#instance.display()

Z= np.zeros_like(df.values)
for k,v in instance.X.get_values().items():
    if v == 1:
        Z[k[0],k[1]] = 1
Z


# In[ ]:





# In[ ]:




