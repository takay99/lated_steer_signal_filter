import numpy as np
import pandas as pd
from scipy.integrate import odeint
import runge_kutta.py as rk

def sim_vehicle_model(delta,v,time,beta,dot_phai):
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

    dot_beta = (a11/v) * beta + (-1+(a12/v**2))*dot_phai + (a13/v)*delta
    ddot_phai = a21*beta + (a22/v)*dot_phai + a23*delta

    return dot_beta, ddot_phai



# ans = sim_vehicle_model(1,1,1,1,1)
# print(ans)
