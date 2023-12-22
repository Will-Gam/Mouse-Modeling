# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 12:34:37 2023

@author: Will
"""
import numpy as np
import seaborn as sea
import csv

csv_file_path = "mouse_data.csv"
data = np.genfromtxt(csv_file_path, delimiter=',', skip_header=1, dtype=float)
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

####manual tests##
#validate our data collection spans the possible angles, distances, locations, etc

#check how well i hit the whole screen at the end
#sea.scatterplot(x=Xf,y=Yf)

#check how well i hit the whole screen at the beginning
#sea.scatterplot(x=Xi,y=Yi)

#check to make sure we hit a range of angles and distances
Xdist = Xf - Xi
Ydist = Yf - Yi
dist = (Xdist**2 + Ydist**2)**.5
angle = np.arctan2(Ydist,Xdist)
angle = np.degrees(angle)

#sea.histplot(angle)
#sea.histplot(dist)
curr_X_dist = Xf - X
curr_Y_dist = Yf - Y
curr_dist = (curr_X_dist**2 + curr_Y_dist**2)**.5

#looking at the paths of mouse swipes, recall Y axis is upside down (origin is top left)

mask = Swipe == 12
#sea.scatterplot(x=X[mask],y=Y[mask])


####Automated tests ######
#switch auto to 1 if you want to run automatic tests
auto = 1
if auto == 1:


#here we validate our initial pre-processing and locate swipes where i make a mistake
#mistakes are important but they are a structural break from where i don't => new model
# some mistakes are in collection process. letting go of shift late/early. 
# not moving far enough for next swipe. no reason to model those

#find mouse swipes with discontinuities in observed time, suggests problem in reworked collection
    discont_list = []
    num1 = 0
    for i in range(1,int(Swipe[-1]+1)):
        mask = Swipe==i
        for j in range(len(T[mask])):
            if T[mask][j] > T[mask][j-1] + .1:
                discont_list.append(i)
                num1+=1
    a = len(discont_list)           

#make sure end points are correct
# check first/last entries of Xf/Yf

    final_list = []
    num2 = 0
    for i in range(1,int(Swipe[-1]+1)):
        mask = Swipe==i
        for j in range(len(T[mask])):
            if X[mask][-1]!=Xf[mask][-1] or X[mask][-1]!=Xf[mask][0] or Y[mask][-1]!=Yf[mask][-1] or Y[mask][-1]!=Yf[mask][0]:
                final_list.append(i)
                num2+=1
    b = len(final_list)

#make sure start points are correct
# check first/last entries of Xf/Yf
    start_list = []
    num3 = 0
    for i in range(1,int(Swipe[-1]+1)):
        mask = Swipe==i
        for j in range(len(T[mask])):
            if X[mask][0]!=Xi[mask][-1] or X[mask][0]!=Xi[mask][0] or Y[mask][0]!=Yi[mask][-1] or Y[mask][0]!=Yi[mask][0]:
                start_list.append(i)
                num3+=1
    c = len(start_list)   

#make sure all Times are smaller than or equal to their final time
    time_list = []
    num4 = 0
    for i in range(len(T)):
        if T[i] > Tf[i]:
            time_list.append(Swipe[i])
            num4+=1
    d = len(time_list)

#make sure all final times are larger than 0        
    Tf_list = []
    num5 = 0
    for i in range(len(T)):
        if Tf[i] <= 0 :
            Tf_list.append(Swipe[i])
            num5+=1
    e = len(Tf_list)

    failed = 0

    if num1!=0:
        failed+=1
    if num2!=0:
        failed+=1
    if num3!=0:
        failed+=1
    if num4!=0:
        failed+=1
    if num5!=0:
        failed+=1

    print(f'\nAutomated Test Results. {failed}/5 tests failed.')     
    print(f'\n{a} cases of discontinuities in time:\n' + str(discont_list))   
    print(f'\n{b} cases of wrong final positions:\n' + str(final_list))
    print(f'\n{c} cases of wrong start positions:\n' + str(start_list))
    print(f'\n{d} cases of times smaller than their final time:\n' + str(time_list))
    print(f'\n{e} cases of final times 0 or smaller:\n' + str(Tf_list)) 

#run this once, check for bad observations. inspect them, if you want to remove
# add the Swipe number to the drop list and run again

    drop = []
#manually fill in swipes you decide to keep, checked through 544:
# we're keeping swipes with one small gap in the last obs
    dont_drop = [110,148,169,374,390,437,438,441,525,531,544]


#automatically fill in drop with problematic swipes, still inspect them and decide whether to remove
    for i in Tf_list:
        if i not in drop and i not in dont_drop:
            drop.append(i)
        
    for i in time_list:
        if i not in drop and i not in dont_drop:
            drop.append(i)
        
    for i in start_list:
        if i not in drop and i not in dont_drop:
            drop.append(i)
        
    for i in final_list:
        if i not in drop and i not in dont_drop:
            drop.append(i)
        
    for i in discont_list:
        if i not in drop and i not in dont_drop:
            drop.append(i)
            
    print('\nWe drop the following swipes:' + f'{drop}')

    print(f'\nWe remove {len(drop)} swipes out of {int(Swipe[-1])} swipes.')
#turn to one when you want to write the csv
    switch = 0
    if switch ==1:   
        file = 'dropswipes.csv'

        with open(file,mode = 'w', newline='') as file:
            header = ['Dropped_Swipes']
            writer = csv.writer(file)
            writer.writerow(header)
            for i in drop:
                row = [i]
                writer.writerow(row)
