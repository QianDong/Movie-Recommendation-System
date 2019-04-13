# -*- coding: utf-8 -*-
"""
Spyder Editor

@author: Qian Dong
"""

import time
import scipy
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import KFold


data = np.loadtxt("ratings.csv", dtype='float', delimiter=",", skiprows = 1)
data = data[:,:3]

alluserID = np.array(set(data[:,0]))
allmovieID = sorted(list(set(data[:,1])))
allmovieID = np.array(allmovieID)

def ui_array_with_rating(train_set):
    # -------- train_set, in each loop of f_fold, it contains 1 test set -------------------  
    train_row_number = len( set(train_set[:,0]) )   # this X should be XT    
    train_rows = list( set( train_set[:,0]) )
    train_rows = sorted(train_rows)
    train_columns = list( set( train_set[:,1]) )  
    train_columns = sorted(train_columns)   
    ui_array = np.zeros(shape = (train_row_number+1,len(allmovieID)))  # +1 is because to have '0' as the begin number    
    # so movieID and userID start from 0 instead of 1, userID = index in that row 
    ui_array[:]= 0
    # train_set[0] = movieId in this set
    ui_array[0] = allmovieID  
    # -----------------------   
    for rowID in range(train_set.shape[0]):
        i = int(train_set[:,0][rowID])  # movieID = i   
        # row_index = i
        j = train_set[:,1][rowID]
        column_index = train_columns.index(j)  # column_index is the location(index) of a movieID in a row 
        k = train_set[:,2][rowID]    
        ui_array[i][column_index]=k    
    return ui_array


def find_Ru(ui_array):
    ''' calculate Ru of each user, 671 Ru in total'''
    all_Ru = []
    for row in ui_array:
        non_zero_rating = row[np.nonzero(row)]
        all_Ru.append( np.mean(non_zero_rating) )
    return all_Ru


 # all_ru_list has len=672, substracted_ui_array has shape=672,163950
def ui_array_substract(train_set, all_Ru_list):
    substract_ui_array = np.zeros(shape=(672, int(allmovieID[-1])+1))
    for row in train_set:
        u = int(row[0])
        i = int(row[1])
        r = row[2]
        substract_ui_array[u][i] = r - all_Ru_list[u]
    return substract_ui_array
 

def compute_sim(i,j, sim_array, substract_ui_array):
    '''compute similarity based on movie index i and movie index j'''  
    i_rowindex = np.searchsorted(allmovieID,i) 
    j_rowindex =np.searchsorted(allmovieID,j) 
    if sim_array[i_rowindex][j_rowindex] != 100 :
        return sim_array[i_rowindex][j_rowindex]
    elif sim_array[j_rowindex][i_rowindex] != 100:
        return sim_array[j_rowindex][i_rowindex]
    else:
        Rui_minus_Ru = substract_ui_array[:,i]  # access a row i in transpose matrix t, which is all rate of movies i 
        Ruj_minus_Ru = substract_ui_array[:,j]
        
        numerator =  np.dot(Rui_minus_Ru , Ruj_minus_Ru)    
        
        Ri_index = np.nonzero(Rui_minus_Ru)[0]   # return index of users, who rated i 
        Rj_index = np.nonzero(Ruj_minus_Ru)[0]
    
        Ri_index_set = set(Ri_index)
        Rj_index_set = set(Rj_index)
    
        u_ij = list(Ri_index_set & Rj_index_set)
    
        if len(u_ij) == 0:
            return 0
        else:
            Rui_minus_Ru = Rui_minus_Ru[u_ij]  # element 0 is movieID
            Ruj_minus_Ru = Ruj_minus_Ru[u_ij]                                                        
            de_num = np.linalg.norm(Rui_minus_Ru) * np.linalg.norm(Ruj_minus_Ru)
            if de_num == 0:
                return 1
            else:
                sim = numerator / de_num  
                sim_array[i_rowindex][j_rowindex] = sim
                sim_array[j_rowindex][i_rowindex] = sim
                return sim    


def all_train_test_index(fold, X):
    ''' divide all data into (fold) folds, f.x.0-8 is training fold, 9 is testing fold '''
    kf = KFold(n_splits = fold, shuffle = True)
    all_train_index = []
    all_test_index = []
    for train_index, test_index in kf.split(X):   # each index is a row of [userId, movieId] in X
        #print(train_index, test_index)
        all_train_index.append(train_index)
        all_test_index.append(test_index)
    return all_train_index, all_test_index
   
    
def all_user_movie_list(train_set, user_movie_dict):      
    user_movie_list = [0]
    start_row = 0
    end_row = 0
    for u in range(1, 672):
        end_row = user_movie_dict[u] + end_row 
        user_movie_list.append(train_set[start_row:end_row])
        start_row = end_row
    return user_movie_list
    

def calculate_p(u, i, sim_array, all_Ru_list, substract_ui_array, train_set, user_movie_array):          
    u_array = user_movie_array[int(u)]
    this_users_movie = u_array[:,1]

    R_u_n_list = u_array[:,2]
    sim_i_n_list = []
    for n in this_users_movie:
        sim = compute_sim(i,int(n),sim_array, substract_ui_array)
        sim_i_n_list.append( sim )
        
    sim_i_n_list = np.array(sim_i_n_list)
    abs_sim = list(map(abs,sim_i_n_list))

    numerator = np.dot( R_u_n_list, sim_i_n_list )
    denumerator = sum(abs_sim) 
    
    if denumerator == 0:
        p = all_Ru_list[u]
    else:
        p = numerator / denumerator
    return p
    

def normalize(array,range_min,range_max):
    array = np.array(array)
    mi = array.min()
    ma = array.max()
    numerator = np.dot( (array - mi ),(range_max- range_min) )
    denumerator = ma - mi
    norm = (numerator/denumerator + range_min) 
    return np.array(norm)


#def fold_10(X,y):   
X = np.vstack((data[:,0],data[:,1])).T
y = data[:,2]


def evaluate_system(fold):
    all_train_index, all_test_index = all_train_test_index(fold, X)
    MAElist = []
    RMSElist = []
    # --------- ----------------
    j = 0
    for f_fold in range(fold):   # f=10 is 10 folds, a is selected fold for testing-fold, rest b is used for training
        # each a is each cross-validation, 10 cross-validation in total        
        test_index = all_test_index[f_fold]
        test_set = data[:,:3][test_index]
        train_index = all_train_index[f_fold]
        train_set = data[:,:3][train_index]    
                      
        ui_array = ui_array_with_rating(train_set)
        
        all_Ru_list = find_Ru(ui_array)
        
        substract_ui_array = ui_array_substract(ui_array, all_Ru_list)
        
        sim_array = np.zeros(shape=(len(allmovieID),len(allmovieID) ))
        sim_array[:] = 100
        p_list = []
        
        # count how many movies each user rated --------------- 
        userID = train_set[:,0]   # NB !!!!!
        user_movie_dict = dict(Counter(userID))
        user_movie_array = all_user_movie_list(train_set,user_movie_dict)
        #------------------------------------ 
        print('trial:',j)
        for row in range(len(test_set)):
            print('round',j,'row', row)
            u = test_set[row][0]
            i = test_set[row][1]
            p = calculate_p( int(u), int(i), sim_array, all_Ru_list, substract_ui_array, train_set, user_movie_array)
            p_list.append(p)
        j += 1
        p_array = np.array([ np.nansum(np.array([p,0])) for p in p_list ] )
        p_array = normalize(np.array(p_list), 0.5, 5)
    
        r_array = test_set[:,2] 
    
        n = len(r_array)
        # MAE
        MAE = np.sum( np.absolute ( p_array -  r_array) ) / n   
        print('fold:',j, 'MAE:',MAE)
        MAElist.append(MAE)
        # RMSE
        RMSE = np.sqrt( np.sum(np.square (p_array -  r_array) ) / n ) 
        print('fold:',j, 'RMSE:',RMSE)
        RMSElist.append(RMSE)
    return MAElist, RMSElist

MAE10, RMSE10 = evaluate_system(10)

MAE3, RMSE3 = evaluate_system(3)
                 

# =================================
#      ANOVA
# =================================
fm,pm = scipy.stats.f_oneway (MAE10, MAE3)
print('p-value MAE is: ' + str(pm))

fr,pr = scipy.stats.f_oneway (RMSE10, RMSE3)
print('p-value of RMSE is: ' + str(pr))


