# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 15:24:17 2023

@author: Will
"""
import numpy as np
import statsmodels.api as sm
import seaborn as sea
import matplotlib.pyplot as plt

csv_file_path = "TfData.csv"
data = np.genfromtxt(csv_file_path, delimiter=',', skip_header=1, dtype=float)

dum_path = "Tfdummy.csv"
data2 = np.genfromtxt(dum_path,delimiter=',', skip_header=1, dtype =bool)

Xi = data[:, 0]
Yi = data[:, 1]
Xf = data[:, 2]
Yf = data[:, 3]
Tf = data[:, 4]
#dummies, true if the x,y comp of the unit vector is >=0
#recall down is positive for y and right is positive for x
dumx = data2[:,0]
dumy = data2[:,1]
#translate True to 1 and false to 0
Xdum =  dumx.astype(int)
Ydum = dumy.astype(int)

#find distance
dx = Xf - Xi
dy = Yf - Yi
dist = (dx**2 + dy**2)**.5

# trying to find a non-circular way to capture 
unit_x = abs(dx/dist)
unit_y = abs(dy/dist)

fracx = abs(dx)/(abs(dx)+ abs(dy))
fracy = abs(dy)/(abs(dx)+ abs(dy))

#folding unit vector x with absolute values
a_unit_x = abs(unit_x)
a_unit_y = abs(unit_y)

up_ux = a_unit_x*800
up_uy = a_unit_y*800

sqrtdist = dist**.5
sqrtdx = abs(dx)**.5
sqrtdy = abs(dy)**.5

xdir = []
ydir = []
#creating dummy for when changes in x,y is non-negative/ negative

#base case is positive or 0, that means right
for i in dx:
    if i>=0:
        xdir.append(0)
    else:
        xdir.append(1)
# base case is positive or 0, that means down
for i in dy:
    if i>=0:
        ydir.append(0)
    else:
        ydir.append(1)
data = {'Tf': Tf,'dist': dist,'Xdum':Xdum,'Ydum': Ydum,'up_ux': up_ux,'up_uy': up_uy,
        'sqrtdist':sqrtdist, 'dx':dx, 'dy':dy,'sqrtdx':sqrtdx ,'sqrtdy':sqrtdy,
        'xdir':xdir,'ydir':ydir, 'fracx':fracx, 'fracy':fracy, 'unit_x':unit_x, 'unit_y':unit_y }
#OLS model
oformula = 'Tf ~ 0 +sqrtdist + unit_x'
omodel = sm.OLS.from_formula(oformula,
data=data, missing='drop')
oresults = omodel.fit()
print(oresults.summary())
ofit = oresults.fittedvalues
ores = oresults.resid
aores = abs(ores)

#scatterplot of residuals
scatterOLS = 0
if scatterOLS == 1:
    sea.scatterplot(x=ofit,y=ores)


# now we need to find weights for WLS
weight_data = {'Tf': Tf,'dist': dist,'ofit':ofit,'aores':aores}
# option 1: model variance of errors with OLS of residuals and dist
distformula = 'aores ~ 0 + sqrtdist'
distmodel = sm.OLS.from_formula(distformula,data=weight_data, missing='drop')
distresults = distmodel.fit()
#print(distresults.summary())
distweight = distresults.fittedvalues

# option 2: model variance of errors with OLS of residuals and Tf
timeformula = 'aores ~ 0+ ofit'
timemodel = sm.OLS.from_formula(timeformula,data=weight_data, missing='drop')
timeresults = timemodel.fit()
#print(timeresults.summary())
timeweight = timeresults.fittedvalues

#option 3: just use sqrtdist
w3 = sqrtdist

#select the weight you want to use:
w = distweight


#WLS
w1model = sm.WLS.from_formula(oformula, data=data,weights=1/w, missing='drop')
w1results = w1model.fit()
print(w1results.summary())

w1fit = w1results.fittedvalues
w1res = w1results.resid
#standardized resids
stand_w1res = w1res*(w**-.5)

order = []
for i in range(1,len(w1res)+1):
    order.append(i)
    
#plot of residuals and the observation order
auto = 0
if auto==1:
    sea.scatterplot(x=order, y= w1res)


#scatterplot of standardized residuals and fitted values
scatter = 0
if scatter == 1:    
    sea.scatterplot(x=w1fit,y=w1res)


#hist of residuals
histnorm = 0
if histnorm == 1:
    sea.histplot(w1res, stat='density')  
 
    
#hist of standardized residuals
histstand = 0    
if histstand ==1:
    sea.histplot(stand_w1res, stat='density')


# Q-Q plot with normal residuals
qqnorm = 0
if qqnorm==1:
    fig, ax = plt.subplots(figsize=(8, 6))
    sm.qqplot(w1res, line='s', ax=ax)

    sea.set(style="whitegrid")
    ax.set_title("Q-Q Plot of Residuals")
    plt.show()


# heterskedasticity tests need a constant
X = sm.add_constant(unit_x)
Y = sm.add_constant(sqrtdist)

#breuschpagan test
BP = 0
if BP == 1:
    print(sm.stats.diagnostic.het_breuschpagan(stand_w1res,X))
#p= 0.776, F= 0.777 => fail to reject homoskedasticity
    print(sm.stats.diagnostic.het_breuschpagan(stand_w1res,Y))
#p= 0.486, F= 0.487 => fail to reject homoskedasticity


# white test for misspecification/heteroskedasticity
white = 0
if white == 1:
    print(sm.stats.diagnostic.het_white(stand_w1res,X))
#p=.781,F=.783 => fail to reject homosked/proper specification
    print(sm.stats.diagnostic.het_white(stand_w1res,Y))
#p= 0.463, F= 0.465 => fail to reject homosked/proper specification
