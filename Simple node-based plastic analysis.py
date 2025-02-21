# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 15:49:46 2022

@author: hk3rr
"""
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# For supports, 0 means unsupported and 1 means supported
#               Position, Support, Force
#               x, y,      x, y,    x, y
nodes = np.array([[0,0 ,0,1 ,0,0], # A = 0
               [2000,4000 ,0,0 ,0,0], #B = 1
               [4000,0 ,0,0 ,0,-1], #C = 2
               [6000,4000 ,0,0 ,0,0], #D = 3
               [8000,0 ,0,0 ,0,-1], #E = 4
               [10000,4000 ,0,0 ,0,0], #F = 5
               [12000,0 ,0,0 ,0,-1], #G = 6
               [14000,4000 ,0,0 ,0,0], #H = 7
               [16000,0 ,1,1 ,0,0] #I = 8
               ])

#             Pt1, Pt2
barData = np.array([[0,  1, 20], # AB
              [1,  2, 20], # BC
              [0,  2, 20], # AC
              [1,3, 20], #BD
              [2,3, 20], #CD
              [2,4, 20], #CE
              [3,4, 20], #DE
              [3,5, 20], #DF
              [4,5, 20], #EF
              [4,6, 20], #EG
              [5,6, 20], #FG
              [5,7, 20], #FH
              [6,7, 20], #GH
              [6,8, 20], #GI
              [7,8, 20] #HI
              ]) # like ElmCon in stiffness matrix method

bars = barData[:, 0:2] # extract start/end node ids (first two columns)
bars = bars.astype(int) # ensure ids are of integer type (and use name 'bars' for compatibility with last weeks code)
a = barData[:,2] # extract areas (third column)

sigmaY = 1 # kN/mm2

num_of_nodes = nodes.shape[0] # automatically calculate how many nodes and bars there are
num_of_bars = bars.shape[0]

# note that there is no a = cp.Variable... here
q = cp.Variable(num_of_bars) # Axial forces, kN, tension +ve
load_factor = cp.Variable(nonneg=True) # Load factor, no unit

from numpy import sqrt

startPoint = np.array([nodes[id, 0:2] for id in bars[:,0]]) # extract positions based on start nodes of each bar
endPoint =  np.array([nodes[id, 0:2] for id in bars[:,1]]) # extract positions of end node of each bar

diffX = endPoint[:,0] - startPoint[:,0] # x2-x1 for all bars
diffY = endPoint[:,1] - startPoint[:,1] # y2-y1 for all bars

# Then use pythagoras to get lengths
length = np.array([sqrt(diffX[i]**2 + diffY[i]**2) for i in range(num_of_bars)])

print('Bar lengths:', length)

# Note: leave the lines which calculate length, as they are still needed 
#       in the equilibrium constraint calculations later
objective = cp.Maximize(load_factor)

yieldConstraints = [sigmaY * a >= q, 
                    sigmaY * a >= -q]

B=np.zeros([2*num_of_nodes, num_of_bars])
for i in range(num_of_bars):
    startNodeID = bars[i,0]
    startNodeRows = [startNodeID*2, startNodeID*2+1]
    endNodeID = bars[i,1]
    endNodeRows = [endNodeID*2, endNodeID*2+1]

    cosAlpha = diffX[i]/length[i] # cosAlpha = adjacent/hypotenuse
    sinAlpha = diffY[i]/length[i] # sinAlpha = opposite/hypotenuse

    B[startNodeRows, i] += [cosAlpha, sinAlpha]
    B[endNodeRows, i] += [-cosAlpha, -sinAlpha]

print("B = ")
print(B)

f = np.reshape(nodes[:,4:6], [-1]) # reshape into 1D array with length to fit

dofCondition = np.reshape(nodes[:,2:4], [-1]) # 1 for supported, 0 for unsupported
fixdof = np.nonzero(dofCondition) # list of supported row indices

f_reduced = np.delete(f, fixdof)
B_reduced = np.delete(B, fixdof, axis=0) # axis=0 deletes rows

print(B_reduced)

equilibriumConstraints = [B_reduced @ q == -f_reduced * load_factor]

prob = cp.Problem(objective, equilibriumConstraints + yieldConstraints)
prob.solve()

for i in range(num_of_bars):
  print('Bar', i, 'connects nodes', bars[i,0:2], 'has area', "%.2f"%a[i], 
        'mm^2 and force', "%.2f"%q.value[i], 'kN')

print('Load factor is', "%.3f"%prob.value)

scale = 0.2 # Arbitrary factor - for visual clarity only

for i in range(num_of_bars):
  start = startPoint[i] # coordinates of start point
  end = endPoint[i] # coordinates of end point

  col = 'k' # default to black if not yielding in either direction
  if  q.value[i]/a[i] >= 0.999*sigmaY: # stress = yield_stress, but with tolerance
      col='b' 
  if -q.value[i]/a[i] > 0.999*sigmaY: # -stress = yield_stress, but with tolerance  
      col='r'
  
  plt.plot([start[0], end[0]], [start[1], end[1]], col, linewidth=scale*a[i])
  plt.gca().set_aspect('equal')
