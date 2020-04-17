#!/usr/bin/env python
# coding: utf-8

# # Final Project
# 
# ## Blake Conrad
# 
# ## TSP Subtour Elimination Via Cutting Planes

# In[28]:


import pandas as pd
import numpy as np
import os
import networkx as nx
import matplotlib.pyplot as plt
import itertools

os.chdir(r"C:\Users\blakeconrad\Desktop\imse884\term_project")
df = pd.read_csv("Nfldata.csv")
df = df.iloc[:,1:].iloc[:16,:16]
print(df.shape)
df


# # Solve TSP
# 
# ## Relaxed, allow subtours

# In[36]:


import os
import pyomo.environ as pyo
from pyomo.environ import *

from pyomo.opt import SolverFactory

opt = pyo.SolverFactory('glpk')

model = ConcreteModel()

#
# TSP DV
#
#Xij if we travel from i to j
#
model.X = Var(range(df.shape[0]), range(df.shape[1]), within=Binary, initialize=0)


#
# TSP Objective
#
# Min dij*xij
#
model.obj = Objective(expr = sum([model.X[i,j]*df.iloc[i,j] for i in model.X_index_0.data() for j in model.X_index_1.data()]), 
                      sense=minimize)
model.constraints = ConstraintList()

#
# TSP Constraints
#
# sum_j{xij} = 1 for all i
#
for i in model.X_index_0.data():
    model.constraints.add(sum([model.X[i,j] for j in model.X_index_1.data() if i != j]) ==1)

# sum_i{xij} = 1 for all j
#
for j in model.X_index_1.data():
    model.constraints.add(sum([model.X[i,j] for i in model.X_index_0.data() if i != j]) ==1)

# Remove 2 loops    
for i in model.X_index_0.data():
    for j in model.X_index_1.data():
        model.constraints.add(model.X[i,j] + model.X[j,i] <= 1)

# Create a model instance and optimize
instance = model.create_instance()
results = opt.solve(instance)
#instance.display()

results={}
D = np.zeros((df.shape[0],df.shape[1]), dtype=int)
for k,v in instance.X.get_values().items():
    if v == 1:
        results[k]=v
        D[k[0],k[1]]=1
        #print(k,v)
        
G = nx.from_numpy_matrix(D) 
nx.draw(G, with_labels=True)

#solve_model(plot=True)


# # Clearly subtours exist

# In[37]:


global model

def add_subtour_constraints2(nodes=[], verbose=False):
    
    tours = list(itertools.permutations(nodes))
    tours = [list(tour) + [tour[0]] for tour in tours]

    if verbose:
        print("Total number of subtours for {} is {}".format(nodes, len(tours)))
        if len(tours) < 8:
            print(tours)
    list_to_sum=[]
    for tour in tours:
        for i in range(len(tour)-1):
            list_to_sum.append(model.X[tour[i],tour[i+1]])
            list_to_sum.append(model.X[tour[i+1],tour[i]])
    print("Constraint added: {}".format(" ".join(['{}+'.format(i) for i in list_to_sum])[:-1] + "<=" + "{}".format(len(tours[0])-1)))
    model.constraints.add(sum(list_to_sum) <= len(tours[0])-1)


def add_subtour_constraints(nodes=[], verbose=False):
    list_to_sum=[]
    #nodes.append(nodes[0])
    for i in nodes:
        for j in nodes:
            list_to_sum.append(model.X[i,j])
            list_to_sum.append(model.X[j,i])
    print("Constraint added: {}".format(" ".join(['{}+'.format(i) for i in list_to_sum])[:-1] + "<=" + "{}".format(len(nodes)-1)))
    model.constraints.add(sum(list_to_sum) <= len(nodes)-1)

def solve_model(plot=False):
    
    # Create a model instance and optimize
    instance = model.create_instance()
    results = opt.solve(instance)
    #instance.display()

    results={}
    D = np.zeros((df.shape[0],df.shape[1]), dtype=int)
    for k,v in instance.X.get_values().items():
        if v == 1:
            results[k]=v
            D[k[0],k[1]]=1
            #print(k,v)

    if plot:
        G = nx.from_numpy_matrix(D) 
        nx.draw(G, with_labels=True)
        
    return D


# In[ ]:





# In[ ]:





# # (Feasibility Cut 1 -- Eliminate Subtours )

# In[38]:


add_subtour_constraints(nodes=[6,0,3], verbose=True)
add_subtour_constraints(nodes=[12,13,15], verbose=True)
add_subtour_constraints(nodes=[2,7,5], verbose=True)
add_subtour_constraints(nodes=[1,11,14,8,4,10,9], verbose=True)

solve_model(plot=True)


# # (Feasibility Cut 2)

# In[39]:


add_subtour_constraints(nodes=[0,2,3], verbose=True)
add_subtour_constraints(nodes=[12,14,15,13,11], verbose=True)

solve_model(plot=True)


# # (Feasibility Cut 3)

# In[40]:


add_subtour_constraints(nodes=[0,3,5], verbose=True)

solve_model(plot=True)


# # (Feasibility Cut 4)

# In[41]:


add_subtour_constraints(nodes=[6,2,5], verbose=True)

solve_model(plot=True)


# # Final Solution: 6704
# 
# 
# ## (0,5,4,7,14,8,12,10,15,13,11,1,9,3,6,2,0)
# 

# In[43]:


obj = instance.obj
print(obj.display())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




