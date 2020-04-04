#!/usr/bin/env python
# coding: utf-8

# # HW2 Problem 4 Credit Card Debt

# In[23]:


import pandas as pd
import numpy as np

df = pd.DataFrame({"Rate":[0.75, 0.5, 0.01],
                   "MinPayment":[300, 50, 125],
                   "InitialDebt":[30000, 8000, 12000]})

df


# In[27]:


import os
import pyomo.environ as pyo
from pyomo.environ import *

from pyomo.opt import SolverFactory

opt = pyo.SolverFactory('glpk')

model = ConcreteModel()

T = [t for t in range(72)]
I = [i for i in range(3)]
D = np.zeros((len(I),len(T)))
D[:,0] = df.InitialDebt.values


model.Y = Var(range(3), range(72), within=Binary, initialize=0)
model.X = Var(I, T, within=NonNegativeReals, initialize=0)

model.obj = Objective(expr = sum([model.X[i,t] for i in model.X_index_0.data() for t in model.X_index_1.data()]), 
                      sense=minimize)
model.constraints = ConstraintList()

# Payoff all debt
model.constraints.add(sum([model.X[i,71] for i in model.X_index_0.data()]) == 0)

# Initialize all debt
for i in model.X_index_0.data():
    model.constraints.add(model.X[i,0] == 0)
    
# Max monthly payoff
for t in model.X_index_1.data():
    if t >= 2:
        model.constraints.add(sum([model.X[i,t] for i in model.X_index_0.data()]) <= 800)

# Min monthly payoff
for i in model.X_index_0.data():
    for t in model.X_index_1.data():
        if t >= 2:
             model.constraints.add(model.X[i,t] >= df.MinPayment.tolist()[i] + 50*model.Y[i,t])

# Must payoff two debts quick                
for t in model.X_index_1.data():
    if t >= 2:
        model.constraints.add(sum([model.Y[i,t] for i in model.X_index_0.data()]) >= 2)
        
# Rolling debt payoff
for i in model.X_index_0.data():
    for t in model.X_index_1.data():
        if t >= 2:
             model.constraints.add(D[i,t] == D[i,t-1] * (1 + df.Rate[i]) - model.X[i,t])       


# # Solution

# In[25]:


# Create a model instance and optimize
instance = model.create_instance()
results = opt.solve(instance, tee=True)
instance.display()


# In[ ]:




