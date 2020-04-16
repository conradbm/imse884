#!/usr/bin/env python
# coding: utf-8

# # IMSE 884 IP Midterm Takehome Project

# In[23]:


import pandas as pd
import numpy as np
import os
import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt

opt = pyo.SolverFactory('glpk')

stocks = [[60.81,59.07,68.43,66.21,60.85,66.54,70.2,64.29,61.86,68.53,65.49,72.09,74.33,75.05,79.18,76.2,77.58],
[174.52,188.47,203.13,200.72,175.07,198.78,205.21,193.34,206.49,220.7,224.4,243.29,266.29,270.77,297.43,308.95,319.0],
[172.51,165.87,178.78,188.65,177.47,191.14,203.91,181.73,180.36,188.08,177.75,189.31,199.32,202.26,208.67,214.87,217.8],
[98.26,97.21,103.16,99.54,101.44,111.13,114.98,105.82,111.99,116.51,117.58,117.15,119.89,119.0,117.89,115.86,119.63]]
names = ["Citibank", "Apple", "Facebook", "Walmart"]

df = pd.DataFrame(stocks, index=names)
df.T.plot()
plt.show()


# In[2]:


I = df.shape[0]
J = df.shape[1]


model = ConcreteModel()

# Number Bought i on time j
model.b = Var(range(I), range(J),
               within=NonNegativeIntegers)
# Number Sold i on time j
model.s = Var(range(I), range(J),
               within=NonNegativeIntegers)

# If Bought i on time j
model.x = Var(range(I), range(J),
               within=Binary)
# If Sold i on time j
model.y = Var(range(I), range(J),
               within=Binary)

# Total budget left at time j
model.B = Var(range(J),
               within=NonNegativeReals)


# In[3]:


model.obj = Objective(expr = model.B[J-1], 
                      sense=maximize)

#import pyomo.environ as aml
model.constraints = ConstraintList()


# # Cannot buy/sell on the same day

# In[4]:


for i in model.x_index_0.data():
    for j in model.x_index_1.data():
        model.constraints.add(model.y[i,j] <= (1-model.x[i,j]))


# # Cannot sell more than you own

# In[14]:


for i in model.x_index_0.data():
    for j in model.x_index_1.data():
        model.constraints.add(model.s[i,j] <= sum([model.b[i,t] - model.s[i,t] for t in range(j)]))

for i in model.x_index_0.data():
    model.constraints.add(sum([model.b[i,j] - model.s[i,j] for j in model.x_index_1.data()]) >= 0)


# # Initial Budget

# In[15]:


model.constraints.add(model.B[0] == 975)


# # Each buy/sell decrememnts and increments realized budget

# In[16]:


for j in model.x_index_1.data():
    if j == 0:
        continue
    model.constraints.add(model.B[j] == model.B[j-1] - sum(model.b[i,j]*df.iloc[i,j] for i in range(I)) + sum(model.s[i,j]*df.iloc[i,j] -7*model.y[i,j] for i in range(I)))


# # Integers and valid variables

# In[17]:


M=1000000
for i in model.x_index_0.data():
    for j in model.x_index_1.data():
        model.constraints.add(model.b[i,j] <= M*model.x[i,j])
        model.constraints.add(model.s[i,j] <= M*model.y[i,j])


# In[18]:


instance = model.create_instance()
results = opt.solve(instance, tee=True)
instance.display()


# # Buy Strategty

# In[19]:


import numpy as np
b = np.zeros((I,J), dtype=np.int)
for idx in instance.b.get_values():
    i,j = idx
    b[i,j] = instance.b.get_values()[idx]
dfb = pd.DataFrame(b, index=df.index)
dfb


# # Sell Strategy

# In[20]:


s = np.zeros((I,J), dtype=np.int)
for idx in instance.s.get_values():
    i,j = idx
    s[i,j] = instance.s.get_values()[idx]
dfs = pd.DataFrame(s, index=df.index)
dfs


# # Maximum Possible Profit

# In[28]:


instance.B.get_values()


# 

# In[ ]:




