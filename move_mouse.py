# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 13:02:05 2024

@author: Will
"""
import numpy as np
import mouse
#model expected time of completion from onset
def finaltime(Xi,Xf,Yi,Yf):  
    #find X dist, Y dist, and total dist       
    Xdist = abs(Xf-Xi)
    Ydist = abs(Yf-Yi)
    dist = (Xdist**2 + Ydist**2)**.5
    #create unit vector components, absolute value
    unit_x = Xdist/dist
    sqrtdist = dist**.5

    #my model 
    my_pred = 0.0286 * sqrtdist + -0.0155*unit_x
    #create a range of values around pred. to est kde
    return my_pred
# model expected time remaining from any point in swipe
def remainingtime(X,Xf,Y,Yf):  
    #find X dist, Y dist, and total dist       
    Xdist = abs(Xf-X)
    Ydist = abs(Yf-Y)
    dist = (Xdist**2 + Ydist**2)**.5
    ln_dist = np.log(dist)
    ln_est_remain_time = -3.9328 + 0.5500*ln_dist
    TR = np.exp(ln_est_remain_time)

    return TR 
#model expected speed at any point in swipe
def est_speed(x,destx,y,desty, time, Tf):
    xdist = destx - x
    ydist = desty - y        
    dist = (xdist**2 + ydist**2)**.5
       
    time_dist = Tf - time
    PT = time/Tf
    
    remain_spd = dist/(time_dist)
    c_PT = (.25 - (PT-.5)**2)**.5
   
    speed = 3.9232*c_PT*remain_spd + 2.6059*c_PT*dist    
    T = speed**-1
      
    return T

#we use plan swipe to pre-plan swipes for performance reasons.
def planswipe(destx,desty):
    #when in use, the start will be the end of the previous swipe,
    #this allows testing for now
    x,y = mouse.get_position()
    xlist = []
    ylist = []
    Tlist = []
    
    # i have two models i could use here, they pretty much agree
    #this is est time remaining from start aka TF
    TR1 = remainingtime(x,destx,y,desty)
    TR2 = finaltime(x,destx,y,desty)
    TR = TR2
    
    #it's nice to see what the models say,
    #we also show time based on the speed values (sum of times to next pixel)
    #we want them to be very similar. 
    #with the bad pathing it is close but a lil fas
    print(f"initial model: {TR2}")
    print(f"remaining time model: {TR1}")
    
    #sumtime is where we calculate the amount of time passed,
    #by incrementing the time to move to next pixel
    #start after 0 bc speed is always 0 at first (get stuck)
    sumtime = .01
    
    #simple (bad) pathing, just move in direction of dest in each coordinate 
    while x!= destx or y!=desty:
        
        if x > destx:
            x-=1
        elif x < destx:
            x+=1
        if y > desty:
            y-=1
        elif y < desty:
            y+=1
            
        #call in est of remaining time  
        TR = remainingtime(x, destx, y, desty)
            
        #time of swipe is time remaining + time passed
        Tf = TR + sumtime
               
        #call in est of movement to next position
        T = est_speed(x, destx, y, desty, sumtime, Tf)
        
        #increment sumtime so it's right for next loop iteration
        sumtime +=  T 
                
        #add info to lists, we will run these lists
        xlist.append(x)
        ylist.append(y)
        Tlist.append(T)
        
    return(xlist,ylist,Tlist)

#last element of time is nan, we don't need it anyway
# easier to remove outside than make the function end before
def clean(X,Y,T):
    for i in range(len(T)):
        if np.isnan(T[i]) or np.isinf(T[i]):
            X.pop(i)
            Y.pop(i)
            T.pop(i)
                        
#simple move function using the pre-filled arrays. 
#next iteration should estimate the length of the loop to ensure accurate T's     
def move(X,Y,T):
    for i,j,k in zip(X,Y,T):
        mouse.move(i, j, duration=k)
    return



#plan the speed/path
X,Y,T = planswipe(20,20)

#remove last obs
clean(X,Y,T)

#move cursor
move(X,Y,T)

# sum T is simulated time TF (result of pathing + speed models)
print(sum(T))
        