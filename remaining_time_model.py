# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 16:52:35 2023

@author: Will
"""
import numpy as np
import seaborn as sea
import statsmodels.api as sm
import matplotlib.pyplot as plt

csvfile = "velocity.csv"
data = np.genfromtxt(csvfile, delimiter=',', skip_header=1, dtype=float)

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


# fraction of time
PT = T/Tf
# time remaining
time_remain = Tf - T

#initial distance
init_dist = ((Xi-Xf)**2 + (Yi-Yf)**2)**.5
sqrt_init_dist = init_dist**.5

#initial estimate
initial_unit_x = abs(Xi - Xf)/ init_dist
est_time_f = 0.0286*sqrt_init_dist + -0.0155*initial_unit_x

# end oriented fraction of distance, call it progress
P_dist_end = dist_end/init_dist
progress = 1 - P_dist_end


rt_dist = dist_end**.6

est_PT = T/est_time_f

time_remain = Tf-T


# set replace to 1 if you want to keep 0 obs by adding a small value
replace = 0
if replace == 1:
    for i in range(len(time_remain)):
        if time_remain[i]==0:
            time_remain[i] =+ .00000001
        if dist_end[i]==0:
            dist_end[i] =+ .00000001
            
lnTR = np.log(time_remain)
ln_dist = np.log(dist_end)

# set drop to 1 if you want to drop 0 observations 
drop = 1
if drop == 1:
    for i in range(len(time_remain)):
        if np.isinf(lnTR[i]):
            lnTR[i] = np.nan
        if np.isinf(ln_dist[i]):
            ln_dist[i] = np.nan

unitx = Xdist_end/dist_end

for i in range(len(unitx)):
    if unitx[i]==0:
        unitx[i] = .00000001

ln_init_dist = np.log(init_dist)
ln_unitx = np.log(unitx)

ols = 1 
if ols ==1:
    data = {'lnTR':lnTR,'ln_dist':ln_dist, 'ln_unitx':ln_unitx, 'init_dist':init_dist }
   
    formula = 'lnTR ~  ln_dist'
    model = sm.OLS.from_formula(formula,data=data, missing='drop')
    results = model.fit()
    print(results.summary())
    
    res = results.resid
    fit = results.fittedvalues
    
    hist = 0
    if hist ==1:
        sea.histplot(res)

# Q-Q plot with residuals
    qq = 0
    if qq==1:
        fig, ax = plt.subplots(figsize=(8, 6))
        sm.qqplot(res, line='s', ax=ax)

        sea.set(style="whitegrid")
        ax.set_title("Q-Q Plot of Residuals")
        plt.show()
        
    heterosked_dist = 0
    if heterosked_dist ==1:
        mask = ~np.isnan(ln_dist) & ~np.isnan(lnTR)
        sea.scatterplot(x=ln_dist[mask], y=res)
        
    heterosked_fit = 0 
    if heterosked_fit ==1:
        sea.scatterplot(x=fit, y=res)

fit_visual = 1
if fit_visual == 1:
    
#let's transform fitted values to look at fits
    actual_mask = (~np.isnan(ln_dist)) & (~np.isnan(ln_init_dist)) & (~np.isnan(ln_unitx))
    OG_fit = np.exp(fit)
    OG_dist = np.exp(ln_dist[actual_mask])

    smask = Swipe[actual_mask] == 337

    TR = time_remain[actual_mask]

    sea.scatterplot(x=OG_dist[smask], y= OG_fit[smask])
    sea.scatterplot(x=OG_dist[smask], y=TR[smask])

