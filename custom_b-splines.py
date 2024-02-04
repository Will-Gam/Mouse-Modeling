# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 13:49:08 2024

@author: Will
"""

import timeit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pan
import csv


#start timer
start = timeit.default_timer()

csv_file_path = "velocity.csv"
data = np.genfromtxt(csv_file_path, delimiter=',', skip_header=1, dtype=float)

Swipe = data[:, 1]
X = data[:, 2]  
Y = data[:, 3]  
    


def subsets(array,number):
#Takes an array and a number of subsets and returns relatively even intervals of an array's indices.
# It has the form: [a,b] [b,c] adjacent intervals share start/end values

#Use to pick exogenous control points (Turns out those are actually called knots)
    
#split into front and back sides until they overlap. middle elements are the first/last covered last
    
#we take arrays of x,y values and the number of subsets we want,
# and return indices for start/end values
           
    #we go forward with index1 and backward with index2
    length = len(array)
    index1 = 0 
    index2 = length - 1
    
    #initialize lists of front/back indices. we'll concatenate to finish
    indices1 = []
    indices2 = []
    
    #we increment indices by the number of elements per subset. calc outside loop:
    ele_per_subset = length/number
    
    # if we have an integer 
    header_range = number/2
   
    # we always round up 
    if ele_per_subset is not int:
        header_range = int(header_range) + 1
        ele_per_subset = int(ele_per_subset) +1

    for i in range(header_range):
            
        if i != header_range-1:
            indices1.append(index1)
            indices2.append(index2)
        
            index1 +=  ele_per_subset          
            index2 -= ele_per_subset  
        else:
            if index2 != indices1[-1]:
                indices2.append(index2)
                            
    #we append the elements of indices2 to indices1, backwards           
    q = len(indices2)            
    for i in indices2:
        q -= 1
        indices1.append(indices2[q])

# if we detect a range more than 1.5 times larger than normal, we even it out with previous range    
    for index in range(len(indices1)-1):
        if indices1[index +1] - indices1[index] >= 1.2*ele_per_subset:
            adjust = True
            needy_index1 = index
            needy_index2 = index+1           
            break
        
        else:
            adjust = False
    
    #while we have a gap bigger than 1.5*normal, we shrink the gap
    adjustments_count = 0       
    while adjust == True:
        
        adjustments_count += 1
                
        indices1[needy_index1] += 2 
        
        indices1[needy_index2] -= 2
        
        for index in range(len(indices1)-1):
            if indices1[index +1] - indices1[index] >= 1.5*ele_per_subset:
                adjust = True
                needy_index1 = indices1[index]
                needy_index2 = indices1[index+1]
            else:
                adjust = False
                
    number_of_subsets = len(indices1) - 1
    
    return indices1, number_of_subsets



def split_indices(Xarray, Yarray, number_of_splits, samples_for_tangent_lines):
    
# take arrays of x,y positions, the number of sub-curves for subsets(), and the number of points
# to sample for estimation of tangent lines

# returns lists of data  from rows(), 
# (list for coefficients, list for constant value) or constant, and the forward direction 
# for the first tangent line. Which allows us to frame them as tangent vectors

# this output will go into our next function which will solve systems of adjacent linear functions in batches    
    lines = []
    forward_directions = []
    
    indices, number_chunks = subsets(Xarray,number_of_splits)
    
    last_index = indices[-1]    
    index1 = 0 
    
    #Here, we have all exogenous control points, including first/last
    exogenousX = []
    exogenousY = []        
    #small: really good: (8,15,15)
    for i in indices:
        #if we are working on interior sections
        exogenousX.append(Xarray[i])
        exogenousY.append(Yarray[i])
        if i != index1 and i != last_index:
            center = i 
            lower_index = center - int(samples_for_tangent_lines/2)
            upper_index = center + int(samples_for_tangent_lines/2)
            
            x1 = Xarray[lower_index]
            x2 = Xarray[upper_index]
            
            y1 = Yarray[lower_index]
            y2 = Yarray[upper_index]
            
            point1 = (x1,y1)
            point2 = (x2,y2)
            
            line, forward_direction = row(point1,point2,Xarray[i],Yarray[i])    
            lines.append(line)
            forward_directions.append(forward_direction)
                        
        #if we are working on first part
        elif i == index1:
            lower_index = i
            upper_index = i + samples_for_tangent_lines
            
            x1 = Xarray[lower_index]
            x2 = Xarray[upper_index]
            
            y1 = Yarray[lower_index]
            y2 = Yarray[upper_index]
            
            point1 = (x1,y1)
            point2 = (x2,y2)
            
            line, forward_direction = row(point1,point2,Xarray[i],Yarray[i])    
            lines.append(line)
            forward_directions.append(forward_direction)
        # if we are working on last part   
        elif i == last_index:
            lower_index = i - samples_for_tangent_lines
            upper_index = i
            
            x1 = Xarray[lower_index]
            x2 = Xarray[upper_index]
            
            y1 = Yarray[lower_index]
            y2 = Yarray[upper_index]
            
            point1 = (x1,y1)
            point2 = (x2,y2)
            
            line, forward_direction = row(point1,point2,Xarray[i],Yarray[i])     
            lines.append(line)
            
            forward_directions.append(forward_direction)
            
    return (lines, exogenousX, exogenousY), number_chunks, forward_directions





def row(point1, point2, ctrlx, ctrly):
#takes two points, returns coefficients and constant from the augmented row of the linear function
# if a line is vertical, return the constant x value
# otherwise, return a tuple:  (list for coefficients, list for constant value)       
    x1 = point1[0]    
    x2 = point2[0]    
    y1 = point1[1]
    y2 = point2[1]
    
    deltax = x2 - x1    
    deltay = y2 - y1
        
    if deltax != 0:  
               
        slope = deltay/deltax
        constant = -slope*ctrlx + ctrly
        
        coefficients = [-slope, 1]
        constant_list = [constant]
         
        forward = np.array([deltax,deltay])
                                  
        return (coefficients, constant_list), forward
    
    else:                                     
       
        forward = np.array([0,deltay])

        return ctrlx, forward       




def solve(line1, line2):
#solves linear system given the linear equations specified in row()

# If we have a constant x (vertical line), the line will be int or float.
# We check data types of each line to decide how to solve 
    try:       
        if type(line1) == tuple and type(line2) == tuple:

            coeffs1 = line1[0]
            coeffs2 = line2[0]
            
            constant1 = line1[1]
            constant2 = line2[1]
        
            a = np.array([coeffs1,coeffs2])
            b = np.array([constant1,constant2])
        
            ctrl = (np.linalg.solve(a, b))
       
            ctrly = ctrl[1][0]
            ctrlx = ctrl[0][0]
      
        elif type(line2) != tuple and type(line1) != tuple:
# not meant to be a permanent solution to this. We assume a straight line
            ctrlx, ctrly = inf_or_none()

        elif type(line1) != tuple:
        
            coeffs2 = line2[0]
            constant2 = line2[1]
        
            ctrlx = line1 
            ctrly = round(ctrlx*coeffs2[0] + constant2[0])     
        
        elif type(line2) != tuple:
        
            coeffs1 = line1[0]
            constant1 = line1[1]
        
            ctrlx = line2
            ctrly = round(ctrlx*coeffs1[0] + constant1[0])

    except np.linalg.LinAlgError:
# exception comes when there is a singular matrix. Still working on how to handle this if 
# no solutions.                       
        ctrlx, ctrly = inf_or_none()
        
    return ctrlx, ctrly


     
def adjacent_batches(lines,exogenousX,exogenousY,forward_direction, Swipe, report = False):
# take lines, run solve() in batches of adjacent lines, return solutions which are control points
# The hub for checking for mis-specified sub-curves    
    list_ctrlx = []
    list_ctrly = []
    
    if report == True:
        print(f"\n\n***** Swipe {Swipe} Vector Info *****")
     
    for i in range(len(lines)-1):
        
        exog_point1 = np.array([exogenousX[i],exogenousY[i]])
        exog_point2 = np.array([exogenousX[i+1],exogenousY[i+1]])
        
        #augmented rows
        line1 = lines[i] 
        line2 = lines[i+1]
        
        ctrlx, ctrly = solve(line1,line2)

#if it's a string, we just replace it later, just skip them here (This is straight line case)      
        if type(ctrlx) != str:
            
#tangent vectors, with length of the secant line we calced with (Not a helpful magnitude)
# We use them to specify the directions of v1,v2
            forward1 = forward_direction[i]            
            forward2 = forward_direction[i+1]
            
#Array of the control point we're working on. Allows us to trivially operate on it with vectors.
            ctrl = np.array([ctrlx,ctrly])
            
# we guess the direction of v1,v2. vector of magnitude of dist between the fitted
# ctrls and exogenous ctrls, and direction of tangent vectors. We guess, check, then adjust if needed
            v1 = exog_point1 - ctrl 
            v2 = exog_point2 - ctrl
            
# use forward1/2 to make sure they're facing the right way. 
# theyre opposite if dotproduct is -1
            if np.dot(v1,forward1) < 0:
                v1 = ctrl - exog_point1
                                
            if np.dot(v2,forward2) < 0:
                v2 = ctrl - exog_point2
                             
            #find unit vectors:
            norm1 = np.linalg.norm(v1)
            uv1 = v1/norm1 
            
            norm2 = np.linalg.norm(v2)
            uv2 = v2/norm2 
            
# uv3, not really a unit vector but the product of them... 
#If the magnitude is less than 1 => they must be opposed
# => we turn around without any samples indicating it
            uv3 = uv1 + uv2
            norm = np.linalg.norm(uv3)
            
            if norm < 1 :            
                print("\n Case 1 detected: Undersampled turn")                                        
            #add our vectors. This gets us the diagonal
                v3 = v1 + v2                
            #new control points, shifted by v3:
                ctrlx += v3[0]
                ctrly += v3[1]
# Next cases are when these vectors converge to/diverge from the endogenous control point.
# Implies that we head that way from both sides, not what we want
# Well specified endog point (P2) is where you have a flow from P1 -> P2 -> P3

            elif np.allclose(exog_point1, ctrl + v1)  and np.allclose(exog_point2, ctrl + v2):
                ctrlx += (v1[0] + v2[0])
                ctrly += (v1[1] + v2[1])               
                if report == True:
                    print("\nCase 2: diverging vectors")
                
            elif np.allclose(exog_point1, ctrl - v1) and np.allclose(exog_point2,ctrl - v2):
                ctrlx -= (v1[0] + v2[0])
                ctrly -= (v1[1] + v2[1])               
                if report == True:
                    print("\nCase 3: converging vectors")
                    
            else:
                if report == True:
                    print("\nCorrect control point detected, no changes needed.")
                    
# now we check if our values are within a reasonable range defined by adjacent control points
# define the range as a square formed by adj ctrls. Need to check which defines upper/lower bounds:
            if exog_point2[0] > exog_point1[0]:
                xlow = exog_point1[0]
                xhigh = exog_point2[0]
            else:
                xlow = exog_point2[0]
                xhigh = exog_point1[0]
            
        
            if exog_point2[1] > exog_point1[1]:
                ylow = exog_point1[1]
                yhigh = exog_point2[1]
                
            else:
                ylow = exog_point2[1]
                yhigh = exog_point1[1]
            
# now we check if we leave these bounds and record if we overshoot or undershoot with X/Yincrement
# We don't really want this to turn into a piecewise linear fit, so we nudge the x,y coordinates 
# if they are on an adjacent control  
            Xincrement = 0
            Yincrement = 0
            if ctrlx > xhigh:
                ctrlx = xhigh
                Xincrement = -3
            elif ctrlx <xlow:
                ctrlx = xlow
                Xincrement = 3
            
            if ctrly > yhigh:
                ctrly = yhigh
                Yincrement = -3
            elif ctrly < ylow:
                ctrly = ylow
                Yincrement = 3
        
            ctrl = np.array([ctrlx,ctrly])
            
            if np.allclose(ctrl, exog_point1) or np.allclose(ctrl, exog_point2):
                ctrlx += Xincrement 
                ctrly += Yincrement
                          
        list_ctrlx.append(ctrlx)
        list_ctrly.append(ctrly)   
    
    return list_ctrlx, list_ctrly



def merge_exog_endog(exogX, exogY, endogX, endogY):
# We our fitted control points and place them inbetween the exogenous
# control points we used to find them

    #start with all exogenous coordinates in lists
    ctrlx = []
    ctrly = []
    
    new_len = len(exogX) + len(endogX)
    
    exog_index = 0 
    endog_index = 0
    for i in range(new_len):
        #if even (bc 1st element is 0):
        if i%2 == 0:
            ctrlx.append(exogX[exog_index])
            ctrly.append(exogY[exog_index])
            exog_index += 1
            #if odd (bc 2nd element is 1):
        elif i%2 ==1:
            ctrlx.append(endogX[endog_index])
            ctrly.append(endogY[endog_index])
            endog_index += 1

    return ctrlx, ctrly



def inf_or_none():
# We just assume linearity when we call this.
# See inf_or_none_future() for where I'm trying to take this
    ctrlx, ctrly = ("Xplaceholder","Yplaceholder")

    return ctrlx, ctrly 



def inf_or_none_future(coeffs1,coeffs2, constant1, constant2, line1, line2):
# in progress:
    
# using ranks of coeff matrix and augmented matrix to check number of solutions (complete)
# Solve systems with infinite solutions (complete)
# Re-work systems with no solutions (incomplete)

            
#if singular = > check rank of A and rank of A|b (augmented matrix)
        
# intuition: 
    
#rank(A) = dim(col(A)) (definition)

# if rank(A|b) = rank(A), then b is a linear combination of columns of A => solution exists
# we only check if det(A)=0 => infinite solutions
 
# if rank(A|b) > rank(A), then b is linearly independent from A columns 
# (bc it's inclusion allows it to span another dim),
#  => no solution
    
    coeff_matrix = np.row_stack((coeffs1,coeffs2))
            
    coeff_rank = np.linalg.matrix_rank(coeff_matrix)
            
    b = np.row_stack((constant1,constant2))
    augmented_matrix = np.column_stack((coeff_matrix,b))
            
    augmented_rank = np.linalg.matrix_rank(augmented_matrix)
            
#if infinite solutions, our linear assumption works
    if augmented_rank == coeff_rank:
        print("inf solutions")
        #infinite solutions, call it equal to next exogonous control point
        ctrlx, ctrly = ("Xplaceholder","Yplaceholder")
        
# if no solutions our linear assumption breaks. Not necessarily a big problem. 
# How far away are the || lines? Problems scale up with that distance
    else:
        print("No solutions")
        #no solution => we need an adjustment
                
        ctrlx, ctrly = str(line1), str(line2)

        # few options, unsure about best approach:
            # 1. alter number of points sampled for estimated tangent lines
            # 2. alter selection of indices for exogenous control points
        # both would force we to either re-work it all or detect them and re-start calculations
        # for all sub-curves for the swipe. In latter case I would need to alter previous work
        # to notice if we are re-running it
            
    return ctrlx, ctrly 



def process_placeholders(ctrlx,ctrly):
#we added placeholder strings when we assumed linearity. Now we set it equal to the next control
    
    # if an element of ctrlx is a string, we set it equal to the next value. 
    # Only can be string if, interior control point and not a knot. if x is str then y is str too
    for i in range(len(ctrlx)-1):        
        if type(ctrlx[i]) == str:
            
            ctrlx[i] = ctrlx[i+1]
            ctrly[i] = ctrly[i+1]
                
    return ctrlx, ctrly



def quadratic_bezier(p1,p2,p3):    
    #takes x,y coordinates of 3 control points and returns x,y values at 
    # formula: (1−t)^2 *P1 + 2(1−t)tP2 + t^2 * P3 
    
    x1 = p1[0]
    y1 = p1[1]
    
    x2 = p2[0]
    y2 = p2[1]
    
    x3 = p3[0]
    y3 = p3[1]
       
    #generate increments betweem 0 and 1 to calc positions at increments of paramter, t
    parameter = [i for i in range(21)]
    for i in range(len(parameter)):
        parameter[i] = parameter[i]/20
    
    x_bez = []
    y_bez = []
            
    for t in parameter:
        x = (1-t)**2*x1 + 2*(1-t)*t*x2 + t**2*x3
        y = (1-t)**2*y1 + 2*(1-t)*t*y2 + t**2*y3
                
        x_bez.append(x)
        y_bez.append(y)
                    
    x_bez = np.array(x_bez)
    y_bez = np.array(y_bez)
        
    return x_bez, y_bez



def bezier_batches(ctrlx,ctrly, number_chunks):
        
## takes array of all control points and uses quadratic bezier function to calc positions
#uses batches of 3 ctrl points
    
    # We find our linear cases and specify control points
    ctrlx, ctrly = process_placeholders(ctrlx, ctrly)

    xbez = []
    ybez = []
    
    middle_list = []
    q = 1
    
    # Only middle control points (endogenous) are 1-1 with subcurves.
    # we create a list of those indices
    for i in range(number_chunks):
        
        middle_list.append(q)
        q += 2
    
    # we loop through the middle control point indices and find adjacent points
    for index2 in middle_list:
        
        index1 = index2 -1 
        index3 = index2 + 1
        
        p1 = (ctrlx[index1], ctrly[index1])
        p2 = (ctrlx[index2], ctrly[index2])
        p3 = (ctrlx[index3], ctrly[index3])
       
        (xvals, yvals) = quadratic_bezier(p1, p2, p3)
               
        for index in range(len(xvals)):
            xbez.append(xvals[index])
            ybez.append(yvals[index])
       
    return xbez, ybez, ctrlx, ctrly



def plot_bezier(xbez, ybez, X, Y, swipe, number_chunks, straight_line=0):    
    #plot actual data, a straight line, and our generated bezier curve  
    
    #plotting actual data
    mask = Swipe == swipe
    plt.plot(X[mask],Y[mask], color = 'blue')
    plt.ylabel("Y Position")
    plt.xlabel("X Position")
    plt.gca().invert_yaxis()
    plt.title(f'Raw Data for Swipe {swipe}')
    plt.show()
    
    #formatting: 
    #screen pov:
        
    #ax = plt.gca()
    #ax.set_xlim([0, 1920])
    #ax.set_ylim([0, 1080])

    plt.ylabel("Y Position")
    plt.xlabel("X Position")
    
    #flip Y axis (origin is top left)
    plt.gca().invert_yaxis()
    plt.title(f'Fit of Swipe {swipe} with {number_chunks} sub-curves')
        
    if straight_line == 1:
    #plotting stright line
        xs = X[mask]
        x1 = xs[0]
        x2 = xs[-1]

        ys = Y[mask]
        y1 = ys[0]
        y2 = ys[-1]

        plt.plot([x1,x2],[y1,y2], color = 'black')
    
    #plot bezier curve           
    plt.plot(xbez,ybez,color = 'pink')
    plt.show()
    return



def sort(swipe,xarray,yarray,g1,g2,g3,g4,g5):
    #sorting swipes by distance between start and finish points
    #returns groups of swipes
    
    x1 = xarray[0]
    x2 = xarray[-1]
    
    y1 = yarray[0]
    y2 = yarray[-1]
    
    dist = ((x2 - x1)**2 + (y2 - y1)**2)**.5
    
    if dist <= 150:
        g1.append(swipe)
        
    elif dist > 150 and dist <= 450:
        g2.append(swipe)
        
    elif dist > 450 and dist <= 850:
        g3.append(swipe)

    elif dist > 850 and dist <= 1250:
        g4.append(swipe)
        
    elif dist > 1250:
        g5.append(swipe)

    return g1, g2, g3, g4, g5



def run_functions(group,dfx,dfy,number_subsets,plot,samples_for_tangent_lines,ctrlx_list,ctrly_list,swipe_list):   
    for swipe in group:
               
        xarray = np.array(dfx.loc[dfx['Swipe']==swipe]['X'])
        yarray = np.array(dfy.loc[dfx['Swipe']==swipe]['Y'])
        
        # find exogenous control points
        (lines, Xexog, Yexog), number_chunks, forward_direction = split_indices(xarray, yarray, number_subsets, samples_for_tangent_lines)
           
        # find endogenous control points
        endog_ctrlx, endog_ctrly = adjacent_batches(lines, Xexog, Yexog, forward_direction, swipe)
        
        # merge exog/endog ctrl points
        ctrlx, ctrly = merge_exog_endog(Xexog, Yexog, endog_ctrlx, endog_ctrly)
        
        #then, run quadratic bezier in chunks of three ctrl points
        xbez, ybez, ctrlx, ctrly = bezier_batches(ctrlx, ctrly, number_chunks)

        for i in range(len(ctrlx)):
            ctrlx_list.append(ctrlx[i])
            ctrly_list.append(ctrly[i])
            swipe_list.append(swipe)
       
        #plot it
        if plot == True:
            plot_bezier(xbez, ybez, X, Y, swipe, number_chunks)
            
    return ctrlx_list, ctrly_list, swipe_list
    


def runit(X,Y,plot=False,groups_run=(1,2,3,4,5),check_out=range(501)):        
    #using dataframes to group positions by swipe number
    stack_x = np.column_stack((X,Swipe))
    dfx = pan.DataFrame(stack_x, columns=['X','Swipe'])

    stack_y = np.column_stack((Y,Swipe))
    dfy = pan.DataFrame(stack_y, columns=['Y','Swipe'])
    
    g1 = []
    g2 = []
    g3 = []
    g4 = []
    g5 = []
    
    ctrlx_list = []
    ctrly_list = []
    swipe_list = []
    
    for swipe in check_out:
#only run code if swipe still present
# we removed observations with user error in data collection, but maintained OG numbering
        if swipe in Swipe:
            xarray = np.array(dfx.loc[dfx['Swipe']==swipe]['X'])
            yarray = np.array(dfy.loc[dfx['Swipe']==swipe]['Y'])
        
            g1,g2,g3,g4,g5 = sort(swipe,xarray,yarray,g1,g2,g3,g4,g5)
        
    length = 0
    
    if 1 in groups_run:
        ctrlx_list, ctrly_list, swipe_list = run_functions(g1, dfx, dfy, 4, plot,10,ctrlx_list,ctrly_list,swipe_list)       
        print("\nGroup 1 finished")
        length += len(g1)
        print("Group 1 length:",len(g1))
        
        
    if 2 in groups_run:
        ctrlx_list, ctrly_list, swipe_list = run_functions(g2, dfx, dfy, 6, plot,10,ctrlx_list,ctrly_list,swipe_list)        
        print("\nGroup 2 finished")
        length += len(g2)
        print("Group 2 length:",len(g2))
        

    if 3 in groups_run:
        ctrlx_list, ctrly_list, swipe_list = run_functions(g3, dfx, dfy, 8, plot,10,ctrlx_list,ctrly_list,swipe_list)        
        print("\nGroup 3 finished")
        length += len(g3)
        print("Group 3 length:",len(g3))        
        

    if 4 in groups_run:
        ctrlx_list, ctrly_list, swipe_list = run_functions(g4, dfx, dfy, 8, plot,10,ctrlx_list,ctrly_list,swipe_list)        
        print("\nGroup 4 finished")
        length += len(g4)
        print("Group 4 length:",len(g4))
        
          
    if 5 in groups_run:
        ctrlx_list, ctrly_list, swipe_list = run_functions(g5, dfx, dfy, 10, plot,10,ctrlx_list,ctrly_list,swipe_list)        
        print("\nGroup 5 finished")
        length += len(g5)
        print("Group 5 length:",len(g5))
        
                         
    return length, ctrlx_list, ctrly_list, swipe_list 



length, ctrlx_list, ctrly_list, swipe_list = runit(X,Y)

# stop timer       
stop = timeit.default_timer()
    
    #print performance results
print("\n***** Performance *****")
print(f'Run Time for {length} Swipes:', stop - start)
 
average_time = (stop - start)/length
print('Run Time Per Mouse Swipe', average_time)


#Now save to CSV file
save = False

#if it doesn't exist, set up headers, count swipes and data points starting at 1
if save == True:
    path = "controlpoints.csv"
    with open(path, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        #Set header if recollecting data
        header = ['Swipe','Control_Point_X','Control_Point_Y']
        writer.writerow(header)

        for i in range(len(ctrlx_list)):
            Row = [swipe_list[i],ctrlx_list[i],ctrly_list[i]]
            writer.writerow(Row)
     


