import numpy as np
import pandas as pd

def sim_vehicle_model(delta,v,time,initial_state):
m = 2000
I = 3000
lf = 1.51
lr = 1.49
Kf = 10**5
Kr = 10**5

a11 = -Kf/m - Kr/m
a12 = -Kf*lf/m + Kr*lr/m
a13 = Kf/m
a21 = -Kf*lf/I + Kr*lr/I
a22 = -Kf*lf**2/I - Kr*lr**2/I
a23 = Kf*lf/I

    return 