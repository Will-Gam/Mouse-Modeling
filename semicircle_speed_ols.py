# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 22:39:21 2023

@author: Will
"""

import numpy as np
import seaborn as sea
import statsmodels.api as sm
import matplotlib.pyplot as plt

csvfile = "velocity.csv"
data = np.genfromtxt(csvfile, delimiter=',', skip_header=1, dtype=float)
#raw data, observations of position and time
N = data[:, 0]
Swipe = data[:, 1]
X = data[:, 2]  
Y = data[:, 3]  
T = data[:, 4]  
# final/ initial positions/time
Xi = data[:, 5]
Yi = data[:, 6]
Xf = data[:, 7]
Yf = data[:, 8]
Tf = data[:, 9]
# calculations of velocities/speed using the calculated delta deta
Vx = data[:,10]
Vy = data[:,11]
speed = data[:,12]
#local averages of velocities/speed
avx = data[:,13]
avy = data[:,14]
av_speed = data[:,15]
#parametric representations of time and distance. 0 when beginning, 1 when ending
PT = data[:,16]
Pdx = data[:,17]
Pdy = data[:,18]
Pd = data[:,19]
# the fraction of total dist that is x/y
frac_total_x = data[:,20]
frac_total_y = data[:,21]
#fraction of remaining dist that is x/y
frac_remain_x = data[:,22]
frac_remain_y = data[:,23]
#dist to the end
dist_end = data[:,24]
Xdist_end = data[:,25]
Ydist_end = data[:,26]
# actual angles calculated with local averages and direct angles, and their difference
pathed_angle = data[:,27]
direct_angle = data[:,28]
deviation = data[:,29]
# actual angles calculated with actual data and direct angles, and their difference
av_pathed_angle = data[:,30]
av_deviation = data[:,31]
#lags of speed
speed_lag1 = data[:,32]
speed_lag2 = data[:,33]
speed_lag3 = data[:,34]
speed_lag4 = data[:,35]
speed_lag5 = data[:,36]
speed_lag10 = data[:,37]
speed_lag15 = data[:,38]
#lags of local average speed
av_speed_lag1 = data[:,39]
av_speed_lag2 = data[:,40]
av_speed_lag3 = data[:,41]
av_speed_lag4 = data[:,42]
av_speed_lag5 = data[:,43]
av_speed_lag10 = data[:,44]
av_speed_lag15 = data[:,45]


Xdist = Xf - Xi
Ydist = Yf - Yi
dist = (Xdist**2 + Ydist**2)**.5

currentXdist = Xf - X
currentYdist = Yf - Y
currentdist = (currentXdist**2 + currentYdist**2)**.5

Tdis = Tf - T
p_cur_dis = -Tdis**2 + Tdis

#average speed to cover remaining dist
remain_spd = dist_end/Tdis

#the swipe i want to plot:
a = 337
#weird/bad: 
#good: 
    
mask = Swipe==a
#sea.scatterplot(x=currentdist[mask],y=av_speed[mask])

#parametric form of distance and time, transformed to functional form of semicircle
c_PT = (.25 - (PT-.5)**2)**.5
c_Pd = (.25 - (Pd-.5)**2)**.5

#parametric form of distance and time, transformed to functional form of parabola
p_PT = -PT**2 + PT
p_Pd = -Pd**2 + Pd

#create new parametric dist:
PPd = 1 - currentdist/dist
c_PPd = (.25 - (PPd-.5)**2)**.5
p_PPd = -PPd**2 + PPd

#sea.scatterplot(x=currentdist[mask],y=av_speed[mask])

#sea.scatterplot(x=p_PT[mask],y=av_speed[mask])

data = {'Tf': Tf,'PT': PT,'Pd':Pd,'av_speed':av_speed,'speed':speed,
        'c_Pd':c_Pd, 'c_PT':c_PT,
        'dist':dist, 'p_PT':p_PT,'p_Pd':p_Pd,
        'frac_total_x':frac_total_x,'frac_total_y':frac_total_y,
         'currentdist':currentdist,
        'Tdis':Tdis, 'p_cur_dis':p_cur_dis, 'remain_spd':remain_spd,
        'c_PPd':c_PPd,'p_PPd':p_PPd}

formula = 'speed ~ 0 + c_PPd:remain_spd +c_PPd:currentdist '
model = sm.OLS.from_formula(formula,
data=data, missing='drop')
results = model.fit()
print(results.summary())

res = results.resid
fit = results.fittedvalues

#histogram of residuals
#sea.histplot(res)

#looking at our model vs individual swipes, aggregate plots won't be helpful
actual_mask = ~np.isnan(speed) & ~np.isnan(currentdist) & ~np.isnan(remain_spd)
dist = currentdist[actual_mask]
Smask = Swipe[actual_mask]==a
#sea.scatterplot(x=dist[Smask],y=fit[Smask])


#residual plots
pt = 0
if pt == 1:
    sea.scatterplot(x=c_PT[actual_mask],y=res)
    
remain = 0
if remain ==1:
    sea.scatterplot(x=remain_spd[actual_mask],y=res)

cdist = 0
if cdist==1:
    sea.scatterplot(x=currentdist[actual_mask],y=res)
    
fitt = 0
if fitt ==1:
    sea.scatterplot(x=fit,y=res)


# Q-Q plot with residuals: checking for normality
qqnorm = 0
if qqnorm==1:
    fig, ax = plt.subplots(figsize=(8, 6))
    sm.qqplot(res, line='s', ax=ax)

    sea.set(style="whitegrid")
    ax.set_title("Q-Q Plot of Residuals")
    plt.show()

#plot order and residuals. looking for autocorrelation
auto = 0

if auto==1:
    sea.scatterplot(x=N[actual_mask],y=res)

