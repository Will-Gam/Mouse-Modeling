# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 16:19:16 2023

@author: Will
"""
import csv
import numpy as np
import pandas as pan

file = "mouse_data.csv"
data = np.genfromtxt(file, delimiter=',', skip_header=1, dtype=float)

N = data[:, 0]
Swipe = data[:, 1]
X = data[:, 2]  
Y = data[:, 3]  
T = data[:, 4]  
Xi = data[:, 5]
Yi = data[:, 6]
Xf = data[:, 7]
Yf = data[:, 8]
Tf = data[:, 9]
User = data[:, 10]

#dropping bad data, data found in raw data validation
file = 'dropswipes.csv'
drop = np.genfromtxt(file, delimiter=',', skip_header=1, dtype=int)

np.array(drop)
OGswipe = np.copy(Swipe)
dropmask = ~np.isin(OGswipe,drop)

mask = dropmask

N = N[dropmask]
Swipe = Swipe[dropmask]
X = X[dropmask]
Y = Y[dropmask]
T = T[dropmask]
Xi = Xi[dropmask]
Yi = Yi[dropmask]
Xf = Xf[dropmask]
Yf = Yf[dropmask]
Tf = Tf[dropmask]
User = User[dropmask]

#creating lags of positions

#create 2D arrays with Swipe counts
xs = np.column_stack((X,Swipe))
ys = np.column_stack((Y,Swipe))
ts = np.column_stack((T,Swipe))
#create dataframes
dfX = pan.DataFrame(xs, columns=['X','Swipe'])
dfY = pan.DataFrame(ys, columns=['Y', 'Swipe'])
dfT = pan.DataFrame(ts, columns=['T', 'Swipe'])

## Create lag variables
lag_intervals = [1 , 2 , 3, 4, 5, 10, 15]

for i in lag_intervals:
    string = 'lag'+str(i)
    
    dfX[string] = dfX.groupby('Swipe')['X'].shift(i)
    dfY[string] = dfY.groupby('Swipe')['Y'].shift(i)
    dfT[string] = dfT.groupby('Swipe')['T'].shift(i)

#remove redundant swipe columns
dfX = dfX.drop('X', axis=1)
dfX = dfX.drop('Swipe', axis=1)

dfY = dfY.drop('Y', axis=1)
dfY = dfY.drop('Swipe', axis=1)

dfT = dfT.drop('T', axis=1)
dfT = dfT.drop('Swipe', axis=1)

#now we create changes in position/T        
lagx = np.array(dfX['lag1'])
lagy = np.array(dfY['lag1'])
lagt = np.array(dfT['lag1'])

deltax = X - lagx
deltay = Y - lagy
deltat = T - lagt

# create lags in changes in positon/T
_x = np.column_stack((deltax,Swipe))
_y = np.column_stack((deltay,Swipe))
_t = np.column_stack((deltat,Swipe))

Dx = pan.DataFrame(_x, columns=['x','Swipe'])
Dy = pan.DataFrame(_y, columns=['y', 'Swipe'])
Dt = pan.DataFrame(_t, columns=['t', 'Swipe'])

## Create lag variables

for i in lag_intervals:
    string = 'lag'+str(i)
    
    Dx[string] = Dx.groupby('Swipe')['x'].shift(i)
    Dy[string] = Dy.groupby('Swipe')['y'].shift(i)
    Dt[string] = Dt.groupby('Swipe')['t'].shift(i)

                
#Now we're cleaning data for the final time model
# we'll have 1 observation per swipe here

#masking data we should keep. Doesn't check 1st element, always true here
# non-identical consecutive elements
mask = Tf[1:] != Tf[:-1]
mask = np.insert(mask, 0, True) 

#the first element is not included, we want it so make it true
c_Tf = Tf[mask]
c_Xf = Xf[mask]
c_Yf = Yf[mask]
c_Xi = Xi[mask]
c_Yi = Yi[mask]
        
file = 'TfData.csv'
with open(file, mode='w', newline='') as file:
    writer = csv.writer(file)
        
    header = ['Xi','Yi','Xf','Yf','Tf']
    writer.writerow(header)
        
    for i in range(len(c_Tf)):
            row = [c_Xi[i],c_Yi[i],c_Xf[i],c_Yf[i], c_Tf[i] ]
            writer.writerow(row)

#creating dummy variables for Tf regression
# dummy variable for directions of unit vector components
# we take absolute value of dx/dist and record 
dist_x = abs(c_Xf-c_Xi)  
dist_y = abs(c_Yf-c_Yi)

c_dx = c_Xf - c_Xi
c_dy = c_Yf - c_Yi

dist_almost = (dist_x*dist_x + dist_y*dist_y)
dist = np.power(dist_almost,.5)

unit_x = c_dx/dist
unit_y = c_dy/dist

#create array where true if unit_x is non-negative 
dumx = unit_x[:] >= 0

# same for unit_y, just recall down is positive!!
dumy = unit_y[:] >= 0

file = 'Tfdummy.csv'
with open(file,mode = 'w', newline='') as file:
    header = ['Unit_x_non_neg','Unit_y_non_neg']
    writer = csv.writer(file)
    writer.writerow(header)
    for i in range(len(dumy)):
        row = [dumx[i],dumy[i]]
        writer.writerow(row)


#creating local average velocities, velocity, speed

#total dist, start to end
totaldistX = Xf - Xi
totaldistY = Yf - Yi
totaldist = (totaldistX**2 + totaldistY**2)**.5

#current dist from end
Xdistfromend = Xf - X
Ydistfromend = Yf - Y
totaldistfromend = (Xdistfromend**2 + Ydistfromend**2)**.5

#current dist from start
Xdistfromstart = X - Xi
Ydistfromstart = Y - Yi
totaldistfromstart = (Xdistfromstart**2 + Ydistfromstart**2)**.5

#parametric dist: fraction of dist completed
DX = Xdistfromstart/totaldistX
DY = Ydistfromstart/totaldistY
D = totaldistfromstart/totaldist

#same with time
DT = T/Tf

#fraction of total dist that is X and Y
fractotalX = abs(totaldistX/totaldist)
fractotalY = abs(totaldistY/totaldist)

#fraction of remaining dist that is X and Y
fracX = abs(Xdistfromend/totaldistfromend)
fracY = abs(Ydistfromend/totaldistfromend)

#velocity components
Vx = deltax *np.power(deltat,-1)
Vy = deltay *np.power(deltat,-1)
speed = (Vx**2 + Vy**2)**.5


# now we fill velocity/speed as zero for first observation
# replacing initial nans for x vel, y velc, and speed
# with 0 for initial movement of swipes. the only nans are from position 1, they come from creating lag1
for i in range(len(Vx)):
    if np.isnan(Vx[i]):
        Vx[i] = 0        
for i in range(len(Vy)):
    if np.isnan(Vy[i]):
        Vy[i] = 0        
for i in range(len(speed)):
    if np.isnan(speed[i]):
        speed[i] = 0


#creating lags of speed
s = np.column_stack((speed,Swipe))
dfSp = pan.DataFrame(s,columns=['speed','Swipe'])

for i in lag_intervals:
    string = 'lag'+str(i)
    dfSp[string] = dfSp.groupby('Swipe')['speed'].shift(i)
    
speed_lag1 = dfSp['lag1']
speed_lag2 = dfSp['lag2']
speed_lag3 = dfSp['lag3']
speed_lag4 = dfSp['lag4']
speed_lag5 = dfSp['lag5']
speed_lag10 = dfSp['lag10']
speed_lag15 = dfSp['lag15']

#creating angle data with raw velocities
actual_angle = np.arctan2(Vy,Vx)
angle_to_dest = np.arctan2(Ydistfromend,Xdistfromend)
deviation = angle_to_dest - actual_angle

# creating moving averages of speed/velocity
def moving_averages(swipe):
    global Swipe
    global Vy
    global Vx
    vx = np.array([])
    vy = np.array([])
    mask = Swipe==swipe    
    q=0
    for i,x in enumerate(Vx[mask]):
        #just use raw value for initial velocity
        if i==1:
            vx = np.append(vx,x)
            vy = np.append(vy,Vy[mask][1])
            # the first elements of Vx/Vy are nans from constructing velocity
        elif np.isnan(x):
            vx = np.append(vx,np.nan)
            vy = np.append(vy,np.nan)
            #just use raw value for final velocity
        elif i==len(Vx[mask])-1:
           vx = np.append(vx,Vx[mask][i])
           vy = np.append(vy,Vy[mask][i])
           #average over relevant elements for 2nd and 2nd to last
        elif i ==2 or i == len(Vx[mask])-2:
            vx = np.append(vx,(x+Vx[mask][i-1]+Vx[mask][i+1])/3)
            vy = np.append(vy,(Vy[mask][i]+Vy[mask][i-1]+Vy[mask][i+1])/3)
            #average over relevant elements for 3rd and 3rd to last
        elif i==3 or i ==len(Vx[mask])-3:
            vx = np.append(vx,(x+Vx[mask][i-1]+Vx[mask][i+1]+Vx[mask][i-2]+Vx[mask][i+2])/5)
            vy = np.append(vy,(Vy[mask][i]+Vy[mask][i-1]+Vy[mask][i+1]+Vy[mask][i-2]+Vy[mask][i+2])/5)
            #average over relevant elements for 4th and 4th to last
        elif i==4 or i==len(Vx[mask])-4:
            vx = np.append(vx,(x+Vx[mask][i-1]+Vx[mask][i+1]+Vx[mask][i-2]+Vx[mask][i+2]+Vx[mask][i-3]+Vx[mask][i+3])/7)
            vy = np.append(vy,(Vy[mask][i]+Vy[mask][i-1]+Vy[mask][i+1]+Vy[mask][i-2]+Vy[mask][i+2]+Vy[mask][i-3]+Vy[mask][i+3])/7)
            # for the rest average over the 4 adjacent values each way
        else:
            vx = np.append(vx,(x+Vx[mask][i-1]+Vx[mask][i+1]+Vx[mask][i-2]+Vx[mask][i+2]+Vx[mask][i-3]+Vx[mask][i+3]+Vx[mask][i-4]+Vx[mask][i+4])/9)
            vy = np.append(vy,(Vy[mask][i]+Vy[mask][i-1]+Vy[mask][i+1]+Vy[mask][i-2]+Vy[mask][i+2]+Vy[mask][i-3]+Vy[mask][i+3]+Vy[mask][i-4]+Vy[mask][i+4])/9)
        q+=1
    return vx,vy

avx = np.array([])
avy = np.array([]) 
for i in range(1,int(Swipe[-1])+1):
        x,y = moving_averages(i)
        avx = np.append(avx,x)
        avy = np.append(avy,y)

av_speed = (avx**2 + avy**2)**.5

#create lags of average speed
#creating data frame of speed
av_s = np.column_stack((av_speed,Swipe))
df_av_Sp = pan.DataFrame(av_s,columns=['av_speed','Swipe'])

for i in lag_intervals:
    string = 'lag'+str(i)
    df_av_Sp[string] = df_av_Sp.groupby('Swipe')['av_speed'].shift(i)   
    
av_speed_lag1 = df_av_Sp['lag1']
av_speed_lag2 = df_av_Sp['lag2']
av_speed_lag3 = df_av_Sp['lag3']
av_speed_lag4 = df_av_Sp['lag4']
av_speed_lag5 = df_av_Sp['lag5']
av_speed_lag10 = df_av_Sp['lag10']
av_speed_lag15 = df_av_Sp['lag15']

#now we use the average velocities to calculate actual pathing
#angle to destination is the same as before, see line 203
av_vel_actual = np.arctan2(avy,avx)
av_vel_deviation = angle_to_dest - av_vel_actual

# find the angle between the current/final location.
# Does not rely on how velocity is calculated 
stack_dest_angle = np.column_stack((angle_to_dest,Swipe))

df_dest_angle = pan.DataFrame(stack_dest_angle,columns=['dest_angle','Swipe'])

#creating data frames for angles calculated with observed speeds
stack_actual_angle = np.column_stack((actual_angle,Swipe))
stack_deviation = np.column_stack((deviation,Swipe))

df_actual_angle = pan.DataFrame(stack_actual_angle,columns=['actual_angle','Swipe'])
df_deviation = pan.DataFrame(stack_deviation,columns=['deviation','Swipe'])

#creating data frames for angles calculated with local average speeds
av_stack_actual_angle = np.column_stack((av_vel_actual,Swipe))
av_stack_deviation = np.column_stack((av_vel_deviation,Swipe))

av_df_actual_angle = pan.DataFrame(av_stack_actual_angle,columns=['av_actual_angle','Swipe'])
av_df_deviation = pan.DataFrame(av_stack_deviation,columns=['av_deviation','Swipe'])

#create lags of angles
for i in lag_intervals:
    string = 'lag'+str(i)
    
    df_dest_angle[string] = df_dest_angle.groupby('Swipe')['dest_angle'].shift(i)
    df_actual_angle[string] = df_actual_angle.groupby('Swipe')['actual_angle'].shift(i)
    df_deviation[string] = df_deviation.groupby('Swipe')['deviation'].shift(i)
    
    av_df_actual_angle[string] = av_df_actual_angle.groupby('Swipe')['av_actual_angle'].shift(i)
    av_df_deviation[string] = av_df_deviation.groupby('Swipe')['av_deviation'].shift(i)

     
file = 'velocity.csv'
with open(file, mode='w', newline='') as file:
    writer = csv.writer(file)
        
    header = ['Obs',
              'Swipe',
              'X',
              'Y',
              'T',
              'Xi',
              'Yi',
              'Xf',
              'Yf',
              'Tf',
              'X_vel',
              'Y_vel',
              'speed',
              'Average_Vx',
              'Average_Vy',
              'Average_speed',
              'parametric_time',
              'parametric_Xdist',
              'parametric_Ydist',
              'parametric_dist',
              'frac_X_total_dist',
              'frac_Y_total_dist',
              'frac_X_remaining_dist',
              'frac_Y_remaining_dist',
              'dist_from_end',
              'Xdist_from_end',
              'Ydist_from_end',
              'actual_angle',
              'angle_to_dest',
              'deviation',
              'average_vel_actual_angle',
              'average_vel_deviation',
              'speed_lag1',
              'speed_lag2',
              'speed_lag3',
              'speed_lag4',
              'speed_lag5',
              'speed_lag10',
              'speed_lag15', 
              'av_speed_lag1',
              'av_speed_lag2',
              'av_speed_lag3',
              'av_speed_lag4',
              'av_speed_lag5',
              'av_speed_lag10',
              'av_speed_lag15',
              
              'lag1_dest_angle',
              'lag1_actual_angle',
              'lag1_deviation',  
              'lag1_av_actual_angle',
              'lag1_av_deviation',  
              'lag2_dest_angle',
              'lag2_actual_angle',
              'lag2_deviation',  
              'lag2_av_actual_angle',
              'lag2_av_deviation', 
              'lag3_dest_angle',
              'lag3_actual_angle',
              'lag3_deviation',  
              'lag3_av_actual_angle',
              'lag3_av_deviation', 
              'lag4_dest_angle',
              'lag4_actual_angle',
              'lag4_deviation',  
              'lag4_av_actual_angle',
              'lag4_av_deviation', 
              'lag5_dest_angle',
              'lag5_actual_angle',
              'lag5_deviation',  
              'lag5_av_actual_angle',
              'lag5_av_deviation', 
              'lag10_dest_angle',
              'lag10_actual_angle',
              'lag10_deviation',  
              'lag10_av_actual_angle',
              'lag10_av_deviation', 
              'lag15_dest_angle',
              'lag15_actual_angle',
              'lag15_deviation',  
              'lag15_av_actual_angle',
              'lag15_av_deviation'
              ]
    writer.writerow(header)
        
    for i in range(len(X)):
            row = [N[i],
                   Swipe[i],
                   X[i],
                   Y[i],
                   T[i],
                   Xi[i],
                   Yi[i],
                   Xf[i],
                   Yf[i],
                   Tf[i],
                   Vx[i],
                   Vy[i],                   
                   speed[i],
                   avx[i],
                   avy[i],
                   av_speed[i],
                   DT[i],
                   DX[i],
                   DY[i],
                   D[i],
                   fractotalX[i],
                   fractotalY[i],
                   fracX[i],
                   fracY[i],
                   totaldistfromend[i],
                   Xdistfromend[i],
                   Ydistfromend[i],
                   actual_angle[i],
                   angle_to_dest[i],
                   deviation[i],
                   av_vel_actual[i],
                   av_vel_deviation[i],
                   speed_lag1[i],
                   speed_lag2[i],
                   speed_lag3[i],
                   speed_lag4[i],
                   speed_lag5[i],
                   speed_lag10[i],
                   speed_lag15[i], 
                   av_speed_lag1[i],
                   av_speed_lag2[i],
                   av_speed_lag3[i],
                   av_speed_lag4[i],
                   av_speed_lag5[i],
                   av_speed_lag10[i],
                   av_speed_lag15[i],
                   
                   df_dest_angle['lag1'][i],
                   df_actual_angle['lag1'][i],
                   df_deviation['lag1'][i],  
                   av_df_actual_angle['lag1'][i],
                   av_df_deviation['lag1'][i],  

                   df_dest_angle['lag2'][i],
                   df_actual_angle['lag2'][i],
                   df_deviation['lag2'][i],    
                   av_df_actual_angle['lag2'][i],
                   av_df_deviation['lag2'][i], 

                   df_dest_angle['lag3'][i],
                   df_actual_angle['lag3'][i],
                   df_deviation['lag3'][i],    
                   av_df_actual_angle['lag3'][i],
                   av_df_deviation['lag3'][i], 

                   df_dest_angle['lag4'][i],
                   df_actual_angle['lag4'][i],
                   df_deviation['lag4'][i],    
                   av_df_actual_angle['lag4'][i],
                   av_df_deviation['lag4'][i],

                   df_dest_angle['lag5'][i],
                   df_actual_angle['lag5'][i],
                   df_deviation['lag5'][i],    
                   av_df_actual_angle['lag5'][i],
                   av_df_deviation['lag5'][i],

                   df_dest_angle['lag10'][i],
                   df_actual_angle['lag10'][i],
                   df_deviation['lag10'][i],    
                   av_df_actual_angle['lag10'][i],
                   av_df_deviation['lag10'][i],  

                   df_dest_angle['lag15'][i],
                   df_actual_angle['lag15'][i],
                   df_deviation['lag15'][i],    
                   av_df_actual_angle['lag15'][i],
                   av_df_deviation['lag15'][i] 
 
                   ]
            writer.writerow(row)
            


