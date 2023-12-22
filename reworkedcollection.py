# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 13:32:06 2023

@author: Will
"""
import pyautogui as py
import keyboard as key
import csv
import numpy as np
import os
import threading
import timeit
###### Reworking collection by splitting into 2 threads. 
# one checks ctrl, the other checks shift and appends data
#replacing time module with timeit (better precision)
X = []
Y = []
T = []
Userlist=[]
Runlist = [1]
ctrl_down = threading.Event()

def coll(X=X,Y=Y,T=T):
    global ctrl_down
    #check if shift is pressed
    while not ctrl_down.is_set():
        if key.is_pressed('shift')==True:          
    #collecting raw data
            (x_, y_) = py.position()
            t_ = timeit.default_timer()
            X.append(x_)
            Y.append(y_)
            T.append(t_)
        if key.is_pressed('Ctrl')==True:
            ctrl_down.set() 
            print('ctrl detected, data processing')            
    return(X,Y,T)

def collectdata(Run_number, User='Will', X=X, Y=Y, Userlist=Userlist, Runlist=Runlist):
    # Create a thread to record mouse position when shift down
    key_thread = threading.Thread(target=coll, args=(X,Y,T))
    key_thread.start()
    # use another thread to check if ctrl has been pressed
    ctrl_down.wait()
    index = []
   # find indices where movement happens
    for i in range(len(T)):
        #if i == 0:
            #index.append(i)
        if i == len(T) - 1:
            index.append(i)
        elif i !=0:
            if X[i] != X[i-1] or Y[i] != Y[i-1]:
                index.append(i-1)     
    #creating new lists for unique positions       
    x_f = []
    y_f = []
    t_f = []
     #fill the new lists according to the indices list          
    for i in range(len(X)):        
        if i in index:
            x_f.append(X[i])
            y_f.append(Y[i])
            t_f.append(T[i])                               
       #increment obs  
    obs = 1    
    #initializing start/end lists
    Start_X = [x_f[0]]
    End_X = []    
    Start_Y = [y_f[0]]
    End_Y = []       
    #create lists storing indices of start/stop
    startindex = [0]
    endindex = []   
    delete_index = []    
    for i in range(0,len(x_f)):       
        if i != 0:            
            #creating ranges of X<Y values. if outside, assume new mouse swipe
            dx = 20 
            dy = 15
            Xrange = range(x_f[i-1]-dx,x_f[i-1]+dx)
            Yrange = range(y_f[i-1]-dy,y_f[i-1]+dy)
        # check if we leave the area of the last position. if we do, new swipe
            if x_f[i] not in Xrange or y_f[i] not in Yrange:
                delete_index.append(i-1)
                if i!=0:
                    obs+=1
                    Runlist.append(obs)
                #i can find start point for new swipe and end point
                #for previous swipe by dicont too
                    startx = x_f[i]
                    starty = y_f[i]
                    # -2 for end points, we drop last obs for each swipe
                    endx = x_f[i-2]
                    endy = y_f[i-2]
                #append start points
                    Start_X.append(startx)
                    Start_Y.append(starty)
                    startindex.append(i)              
                #insert into the first list index
                    End_X.append(endx)
                    End_Y.append(endy)
                    endindex.append(i-2)  
            else:
                    Runlist.append(obs)
    
    End_Y.append(y_f[-1])
    End_X.append(x_f[-1])    
    endindex.append(len(x_f)-1)
       
    # remove last observation of each Swipe. 
    #unavoidable delay between pressing/releasing shift key messes up data
    for i in delete_index:
        x_f.pop(i)  
        y_f.pop(i)       
        t_f.pop(i)  
        Runlist.pop(i)
        #the inices change as i remove elements. each time they go down by 1 (1 element is removed)
        for j in range(len(delete_index)):
            delete_index[j] -= 1

    #create user list. only matters if i diversify my data collection (another person, sens, hand, grip, etc)
    for i in range(len(y_f)):
        Userlist.append(User)
        
    #creating new lists for end/starts. 
    #these fill in the start/end for all indices of the same mouse swipe
    
    #startpoint lists
    fsx = []
    fsy = []
    #endpoint lists
    fex = []
    fey = []   
    #new variable recording the swipe end time to all observations
    finaltime = [] 
    
    # initialize with placeholders
    for i in range(len(x_f)):
        fsx.append("x")
        fsy.append("y")        
        fex.append("x2")
        fey.append("y2")       
        finaltime.append("fT")
                #manually remove last obs of last swipe
    x_f.pop()  
    y_f.pop()       
    t_f.pop()  
    Runlist.pop()  

    x_f = np.array(x_f)
    y_f = np.array(y_f)   
    Runlist = np.array(Runlist)    
    
    #finding first/last positions and filling them in for all swipe observations
    for i in range(0,Runlist[-1]+1):
        #no 0th run(Swipe), but start range at 0 so it runs loop if only 1 Swipe
        if i !=0:
            mask = Runlist==i
        
            xi = x_f[mask][0]
            yi = y_f[mask][0]
            
            xf = x_f[mask][-1]
            yf = y_f[mask][-1]
        
            for j in range(len(x_f)):
                if Runlist[j] == i:
                    fsx[j] = xi
                    fsy[j] = yi
                
                    fex[j] = xf
                    fey[j] = yf
    x_f = list(x_f)
    y_f = list(y_f)
    Runlist = list(Runlist)
#find values and indices for start and end times. we use end times for analysis
# we use start times to start T at zero and count from there                
    init = []
    final = []
    
    initdex = []
    finaldex = []
    
    for j in range(len(Runlist)):   
        if j == 0:
            init.append(t_f[j])
            initdex.append(0)
            
        elif Runlist[j] != Runlist[j-1] and j+1 < len(Runlist):
            final.append(t_f[j-1])
            init.append(t_f[j])
            
            finaldex.append(j-1)
            initdex.append(j)
            
    final.append(t_f[-1])
    
    finaldex.append(len(Runlist)-1)
  
    #make final times be counted from 0
    for i in range(len(final)):
        final[i] -= init[i]
        
     #subtract each T by T0.
    currentel = 0       
    for i in range(len(Runlist)):
        if i ==0:
            t_f[i] -= init[currentel]
            
        elif Runlist[i] == Runlist[i-1]:
            t_f[i] -= init[currentel]
            
        elif Runlist[i] != Runlist[i-1]:
            currentel += 1
            t_f[i] -= init[currentel]                            

    #fill in final times for every observation of the swipe
    currentel = 0       
    for i in range(len(Runlist)):
        if i == 0:
            finaltime[0] = final[0]
            currentel += 1
            
        elif Runlist[i] == Runlist[i-1]:
            finaltime[i] = finaltime[i-1]
            
        elif Runlist[i] != Runlist[i-1]:
            finaltime[i] = final[currentel]
            currentel += 1
                        
    return(x_f,y_f,t_f,Userlist,Runlist,fsx,fsy,fex,fey,finaltime)
        
(X,Y,t,Userlist,Runlist,fsx,fsy,fex,fey,finaltime) = collectdata(1)

#My csv file
path = "mouse_data.csv"

#if it doesn't exist, set up headers, count swipes and data points starting at 1
if os.path.exists(path) == False:
    
    with open(path, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        #Set header if recollecting data
        header = ['N', 'Swipe', 'X', 'Y', 'Time', 'Initial_X',
                  'Initial_Y', 'Final_X', 'Final_Y','Final_Time', 'User']
        writer.writerow(header)
        
        q = 1
        for i in range(len(X)):
            row = [q, Runlist[i], X[i], Y[i], t[i], fsx[i],
                   fsy[i],fex[i],fey[i],finaltime[i], Userlist[i] ]
            writer.writerow(row)
            q+=1
# if the file exists, append data, count N and swipes from previous last
else:
    #append rows
    data = np.genfromtxt(path, delimiter=',', skip_header=1, dtype=float)
    N = data[:, 0]
    Swipe = data[:, 1]
    pre_existing_N = N[-1]
    pre_existing_Swipes = Swipe[-1]
    
    with open(path, mode='a', newline='') as file:
        writer = csv.writer(file)
        # if it is not empty find how many swipes and how many rows

        q = 1
        for i in range(len(X)):
            row = [q+ pre_existing_N, Runlist[i] + pre_existing_Swipes,
                   X[i], Y[i], t[i], fsx[i], fsy[i],fex[i],
                   fey[i], finaltime[i],Userlist[i] ]
            writer.writerow(row)
            q+=1
        
print("You've collected an additional " + str(q) + " samples of movement across " + str(Runlist[-1])+ " mouse swipes")      