import gurobipy as gp
from gurobipy import GRB
import numpy as np

env = gp.Env(empty=True)
env.setParam("OutputFlag", 0)
env.start()
model = gp.Model(env=env)

x = model.addMVar(3, lb=0, name="x")
w = model.addMVar((4, 6), lb=0, name="w")
y = model.addMVar((2, 6), lb=0, name="y")

totalobj = 150*x[0]+230*x[1]+260*x[2]-0.1*(170*w[0,0]-238*y[0,0]+150*w[1,0]-210*y[1,0]+36*w[2,0]+10*w[3,0]) \
            -0.2*(170*w[0,1]-238*y[0,1]+150*w[1,1]-210*y[1,1]+36*w[2,1]+10*w[3,1]) \
            -0.15*(170*w[0,2]-238*y[0,2]+150*w[1,2]-210*y[1,2]+36*w[2,2]+10*w[3,2]) \
            -0.35*(170*w[0,3]-238*y[0,3]+150*w[1,3]-210*y[1,3]+36*w[2,3]+10*w[3,3]) \
            -0.15*(170*w[0,4]-238*y[0,4]+150*w[1,4]-210*y[1,4]+36*w[2,4]+10*w[3,4]) \
            -0.05*(170*w[0,5]-238*y[0,5]+150*w[1,5]-210*y[1,5]+36*w[2,5]+10*w[3,5]) \

model.setObjective(totalobj, GRB.MINIMIZE)

model.addConstr(x[0]+x[1]+x[2] <= 500)
model.addConstr(3*x[0]+y[0,0]-w[0,0] >= 200)
model.addConstr(3.6*x[1]+y[1,0]-w[1,0] >= 240)
model.addConstr(-24*x[2]+w[2,0]+w[3,0] <= 0)
model.addConstr(w[2,0] <= 6000)
model.addConstr(2.4*x[0]+y[0,1]-w[0,1] >= 200)
model.addConstr(3*x[1]+y[1,1]-w[1,1] >= 240)
model.addConstr(-20*x[2]+w[2,1]+w[3,1] <= 0)
model.addConstr(w[2,1] <= 6000)
model.addConstr(2*x[0]+y[0,2]-w[0,2] >= 200)
model.addConstr(2.4*x[1]+y[1,2]-w[1,2] >= 240)
model.addConstr(-16*x[2]+w[2,2]+w[3,2] <= 0)
model.addConstr(w[2,2] <= 6000)
model.addConstr(2.2*x[0]+y[0,3]-w[0,3] >= 200)
model.addConstr(3.4*x[1]+y[1,3]-w[1,3] >= 240)
model.addConstr(-17*x[2]+w[2,3]+w[3,3] <= 0)
model.addConstr(w[2,3] <= 6000)
model.addConstr(2.8*x[0]+y[0,4]-w[0,4] >= 200)
model.addConstr(2.6*x[1]+y[1,4]-w[1,4] >= 240)
model.addConstr(-19*x[2]+w[2,4]+w[3,4] <= 0)
model.addConstr(w[2,4] <= 6000)
model.addConstr(2.1*x[0]+y[0,5]-w[0,5] >= 200)
model.addConstr(2.5*x[1]+y[1,5]-w[1,5] >= 240)
model.addConstr(-22*x[2]+w[2,5]+w[3,5] <= 0)
model.addConstr(w[2,5] <= 6000)

model.optimize()

print(x.X)
print(y.X)
print(w.X)
print(model.objVal)