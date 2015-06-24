'''
Created on 25.02.2013

@author: psinger
'''

import csv
from random import choice
from random import shuffle


            
def stratified_kfold(paths, k=10):
    '''
    Performs stratified kfold on a set of paths
    paths = list of paths (each row is a numpy array of elements of a path)
    k = number of folds
    '''
    total_sum = sum([len(x) for x in paths])
    print "total nr. of paths: ", len(paths)
    print "total_sum= ", total_sum
    print "==========="
    wanted_sum = float(total_sum) / float(k)
    shuffle(paths)
    
    folds = []
    curr_fold = []
    curr_len = 0
    for line in paths:
        curr_fold.append(line)
        curr_len += len(line)
        if curr_len >= wanted_sum:
            folds.append(curr_fold)
            curr_fold =  []
            curr_len = 0
            
    if curr_len >= 0:
        folds.append(curr_fold)
        
    assert(len(folds) == k)
    
    return folds
                
