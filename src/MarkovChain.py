'''
Created on 14.01.2013

@author: psinger
'''

from __future__ import division

#import PathSim
#import csv
from collections import defaultdict, OrderedDict
import random
import collections
import operator
#import scipy.sparse as sp
import numpy as np
import sys
import math
#import operator
#from scipy import stats
from scipy.special import gammaln
from scipy.sparse import csr_matrix, coo_matrix
#from scipy.special import gamma
#import copy
#from random import choice
import itertools
import copy
import tables as tb
import warnings

RESET_STATE = "-1"

UNKNOWN_STATE = "1"

#we need this for k = 0
FAKE_ELEM = "-10"

#Prior
#PRIOR = 1.00

class MarkovChain():
    '''
    Class for fitting a Markov chain of arbitrary order
    '''

    def __init__(self, k=1, reverse=False, use_prior=False,  reset=True, prior=1., state_count = None, specific_prior = None, specific_prior_vocab = None, modus="mle"):
        '''
        Constructor
        :param modus: specifies the modus of the class, there are two possibilities: modus='mle' is focused on working
        with mle matrices representing probabilities
        :param  modus: 'bayes' focuses on working with bayesian evidence and only works with plain transition counts
        :param reverse: revert the paths
        :param use_prior: flag if script should use a prior
        :param reset: flag for using generic reset state
        :param prior: prior (pseudo count) for each single element (in case of MLE it is smoothing)
        :param state_count: set if you want to also cover states not observed in the data by setting the count manually
        :param specific_prior: sparse matrix of specific alpha configurations (can also be a hdf5 matrix)
        Note that usually the values should directly represent the additional alpha values for corresponding elements.
        :param specific_prior_vocab: dictionary of vocabulary that matches state names with indices
        of the specific_prior matrix
        '''
        self.k_ = k
        self.reset_ = reset

        self.state_count_ = state_count
        self.states_initial_ = []
        self.parameter_count_ = 0
        self.observation_count_ = 0


        self.paths_ = list()
        self.paths_test_ = list()

        #probabilities
        self.transition_dict_ = defaultdict(lambda : defaultdict(float))

        self.prediction_position_dict_ = dict()
        #self.states_ = dict()
        #self.states_reverse_ = dict()
        self.dtype_ = np.dtype(float)
        #self.reverse_ = reverse
        self.modus_ = modus

        self.use_prior_ = use_prior
        self.alpha_ = prior



        self.specific_prior_ = specific_prior
        self.specific_prior_vocab_ = specific_prior_vocab


        ##print self.specific_prior_
        if self.specific_prior_ is not None and k != 1:
            raise Exception("Using specific priors with higher orders not yet implemented!")
        if self.specific_prior_ is not None and self.specific_prior_vocab_ is None:
            raise Exception("Can't work with a specific alpha without vocabulary information!")
        if self.specific_prior_ is not None and self.modus_ != "bayes":
            raise Exception("Specific alpha only works mit Bayes modus!")
        if self.specific_prior_ is not None and isinstance(self.specific_prior_, csr_matrix):
            if self.specific_prior_.shape[0] != self.specific_prior_.shape[1]:
                warnings.warn("Specific alpha dimensions are not the same. Only appropriate if one the matrix is 1xN for setting each row the same! Only works for csr_matrix!")


        self.proba_from_unknown_ = 0
        self.proba_to_unknown_ = dict()

    def _dict_divider(self, d):
        '''
        Internal function for dict divider and smoothing
        '''

        if self.use_prior_ == True:
            smoothing_divider = float(self.state_count_ * self.alpha_)
            #print "smoothing divider: ", smoothing_divider
            self.proba_from_unknown_ = self.alpha_ / smoothing_divider
            #print "proba_from_unknown_: ", self.proba_from_unknown_

            for k, v in d.iteritems():
                s = float(sum(v.values()))
                #smoothing_divider = float(sum([round(x*self.alpha_)+self.alpha_ for x in self.specific_prior_[k].values()]))
                #smoothing_divider += float((self.state_count_ - len(self.specific_prior_[k].values())) * self.alpha_)

                divider = s + smoothing_divider
                self.observation_count_ += divider
                for i, j in v.iteritems():
                    v[i] = (j + self.alpha_) / divider
                self.proba_to_unknown_[k] = self.alpha_ / divider
                ##print "row sum: ", (float(sum(v.values())) + ((self.state_count_ - len(v)) * self.proba_to_unknown_[k]))
        else:
            for k, v in d.iteritems():
                s = float(sum(v.values()))
                self.observation_count_ += s
                for i, j in v.iteritems():
                    v[i] = j / s

                ##print "row sum: ", float(sum(v.values()))

    def _dict_ranker(self, d):
        '''
        Apply ranks to a dict according to the values
        Averages ties
        '''
        my_d = collections.defaultdict(list)
        for key, val in d.items():
            my_d[val].append(key)

        ranked_key_dict = {}
        n = v = 1
        for _, my_list in sorted(my_d.items(), reverse=True):
            #v = n + (len(my_list)-1)/2.
            v = n + len(my_list)-1
            for e in my_list:
                n += 1
                ranked_key_dict[e] = v

        #little hack for storing the other unobserved average ranks
        #this is wanted so that we do not have to calculate it all the time again
        #ranked_key_dict[FAKE_ELEM] = n + ((self.state_count_-len(ranked_key_dict)-1)/2.)
        ranked_key_dict[FAKE_ELEM] = self.state_count_

        return ranked_key_dict

    def _distr_chips_row(self, matrix, chips):
        '''
        Helper class!
        Do not use outside.
        See: https://github.com/psinger/HypTrails
        '''

        matrix = (matrix / matrix.sum()) * chips

        floored = matrix.floor()
        rest_sum = int(chips - floored.sum())

        matrix = matrix - floored

        idx = matrix.data.argpartition(-rest_sum)[-rest_sum:]

        i, j = matrix.nonzero()

        i_idx = i[idx]
        j_idx = j[idx]

        if len(i_idx) > 0:
            floored[i_idx, j_idx] += 1

        floored.eliminate_zeros()

        del matrix
        return floored

    def prepare_data(self, paths):
        '''
        preparing data
        ALWAYS CALL FIRST
        '''
        states = set()
        if self.reset_:
            states.add(RESET_STATE)
            if self.state_count_ is not None:
               self.state_count_ += 1

        for line in paths:
            for ele in line:
                states.add(ele)
        ##print self.state_distr_

        self.states_initial_ = frozenset(states)


        #self.state_count_ = math.pow(float(len(states)), self.k_)
        if self.state_count_ is None:
            self.state_count_ = float(len(states))

        if self.state_count_ < float(len(states)):
            raise Exception("You set the state_count too low!")

        self.parameter_count_ = pow(self.state_count_, self.k_) * (self.state_count_ - 1)
        #print "initial state count", self.state_count_
        ##print self.states_initial_

    def fit(self, paths, ret=False):
        '''
        fitting the data and constructing MLE
        ret = flag for returning the transition matrix
        '''
        #print "====================="
        #print "K: ", self.k_
        #print "prior: ", self.alpha_

        for line in paths:
            if self.reset_:
                self.paths_.append(self.k_*[RESET_STATE] + [x for x in line] + [RESET_STATE])
            else:
                self.paths_.append([x for x in line])

        for path in self.paths_:
            i = 0
            for j in xrange(self.k_, len(path)):
                elemA = tuple(path[i:j])
                i += 1
                elemB = path[j]
                if self.k_ == 0:
                    self.transition_dict_[FAKE_ELEM][elemB] += 1
                else:
                    self.transition_dict_[elemA][elemB] += 1

        ##print self.transition_dict_


        if self.modus_ == "mle":
            self._dict_divider(self.transition_dict_)

        if ret:
            return self.transition_dict_

    def loglikelihood(self):
        '''
        Calculating the log likelihood of the fitted MLE
        '''

        if self.modus_ != "mle":
            raise Exception("Loglikelihood calculation does not work with modus='bayes'")

        likelihood = 0
        prop_counter = 0

        for path in self.paths_:
            i = 0
            for j in xrange(self.k_, len(path)):
                elemA = tuple(path[i:j])
                i += 1
                elemB = path[j]
                if self.k_ == 0:
                    prop = self.transition_dict_[FAKE_ELEM][elemB]
                else:
                    prop = self.transition_dict_[elemA][elemB]
                likelihood += math.log(prop)
                prop_counter += 1

        #print "likelihood", likelihood
        #print "prop_counter", prop_counter
        return likelihood


    #@profile
    def bayesian_evidence(self):
        '''
        Calculating the bayesian evidence
        Not every single exception case is tackled in the code of this function.
        It is the responsibility of the user that---if used---the specific prior matrix is appropriately shaped and set.
        :return: Bayesian evidence (marginal likelihood)
        '''
        if self.modus_ != "bayes":
            raise Exception("Bayesian evidence does not work with modus='mle'")


        single_row = False
        is_hdf5 = False
        is_csr = False
        if self.specific_prior_ is not None:
            if isinstance(self.specific_prior_, csr_matrix):
                is_csr = True
                if self.specific_prior_.shape[0] == 1:
                    single_row = True
                if self.reset_:
                    if self.specific_prior_.shape[1] < self.state_count_ - 1:
                        raise Exception("your specific prior needs to at least cover all states in the trails, shape mismatch")
                else:
                    if self.specific_prior_.shape[1] < self.state_count_:
                        raise Exception("your specific prior needs to at least cover all states in the trails, shape mismatch")
            elif isinstance(self.specific_prior_, tb.group.RootGroup):
                is_hdf5 = True
            else:
                raise Exception("wrong specific prior format")



        evidence = 0
        counter = 0
        i = 0

        #only works for order 1 atm
        # if self.reset_ == False:
        #     allkeys = frozenset(self.transition_dict_.keys())
        #     for s in self.states_initial_:
        #         if (s,) not in allkeys:
        #             self.transition_dict_[(s,)] = {}

        tmp = 0

        for k,v in self.transition_dict_.iteritems():
            tmp += 1

            first_term_denom = 0
            second_term_enum = 0

            #start with combining prior knowledge with real data
            cx = None
            if self.specific_prior_ is not None:
                if single_row:
                    if k[0] != RESET_STATE:
                        cx = self.specific_prior_
                else:
                    if k[0] != RESET_STATE:
                        if is_csr:
                            cx = self.specific_prior_.getrow(self.specific_prior_vocab_[k[0]])
                        elif is_hdf5:
                            row = self.specific_prior_vocab_[k[0]]
                            indptr_first = self.specific_prior_.indptr[row]
                            indptr_second = self.specific_prior_.indptr[row+1]
                            data = self.specific_prior_.data[indptr_first:indptr_second]
                            indices = self.specific_prior_.indices[indptr_first:indptr_second]
                            indptr = np.array([0,indices.shape[0]])
                            if self.reset_:
                                shape = (1, self.state_count_-1)
                            else:
                                shape = (1, self.state_count_)
                            cx = csr_matrix((data, indices, indptr), shape=shape)


            n_sum = sum(v.values())

            if n_sum == 0.:
                raise Exception("The row sum should not be zero, something went wrong here!")

            prior_sum = 0

            if cx is not None:
                prior_sum += cx.sum()

            prior_sum += int(self.state_count_) * self.alpha_
            for x, c in v.iteritems():
                prior = self.alpha_

                # if empirical_prior > 0:
                #     prior += empirical_prior
                if cx is not None and x != RESET_STATE:
                    idx = self.specific_prior_vocab_[x]
                    prior += cx[0, idx]

                cp = c + prior

                first_term_denom += gammaln(prior)
                
                second_term_enum += gammaln(cp)

                counter += prior

            #do the final calculation
            first_term_enum = gammaln(prior_sum)
            first_term = first_term_enum - first_term_denom
            
            second_term_denom = gammaln(n_sum + prior_sum)
            second_term = second_term_enum - second_term_denom

            evidence += (first_term + second_term)

        #print "evidence", evidence
        ##print self.alpha_, empirical_prior, wrong_prior
        ##print "pseudo counts: ", counter
        return evidence

    
    def predict_eval(self, test, eval="rank"):
        '''
        Evaluating via predicting sequencies using MLE
        eval = choice between several evaluation metrics, "rank" is a ranked based approach and "top" checks whether
                true state is in the top K ranks
        ''' 
        
        if self.modus_ != 'mle':
            raise Exception("Prediction only works with MLE mode!")
        
        if self.use_prior_ != True:
            raise Exception("Prediction only works with smoothing on!")

        if eval == "rank":
            for k,v in self.transition_dict_.iteritems():
                #print v
                self.prediction_position_dict_[k] = self._dict_ranker(v)

        known_states = frozenset(self.transition_dict_.keys())
        
        for line in test:
            #if self.k
            self.paths_test_.append(self.k_*[RESET_STATE] + [x for x in line] + [RESET_STATE])

        topx = 5
        position = 0.
        counter = 0.
        #print "clicks test", len(self.paths_test_)
        
        for path in self.paths_test_:
            i = 0
            for j in xrange(self.k_, len(path)):
                elem = tuple(path[i:j])
                i += 1
                true_elem = path[j]
                
                if self.k_ == 0:
                    if eval == "rank":
                        p = self.prediction_position_dict_[FAKE_ELEM].get(true_elem,
                                                                          self.prediction_position_dict_[FAKE_ELEM][
                                                                              FAKE_ELEM])
                    elif eval == "top":
                        row = self.transition_dict_[FAKE_ELEM]
                        items = row.items()
                        random.shuffle(items)
                        row = OrderedDict(items)
                        top = dict(sorted(row.iteritems(), key=operator.itemgetter(1), reverse=True)[:topx]).keys()
                        if true_elem in top:
                            p = 1
                        else:
                            p = 0
                else:
                    #We go from an unknown state to some other
                    #We come up with an uniform prob distribution
                    if elem not in known_states:
                        if eval == "rank":
                            #p = self.state_count_ / 2.
                            p = self.state_count_
                        elif eval == "top":
                            prob = topx / self.state_count_
                            if random.uniform <= prob:
                                p = 1
                            else:
                                p = 0
                    #We go from a known/learned state to some other
                    else:
                        if eval == "rank":
                            p = self.prediction_position_dict_[elem].get(true_elem,
                                                                         self.prediction_position_dict_[elem][
                                                                             FAKE_ELEM])
                        elif eval == "top":
                            row = self.transition_dict_[elem]
                            items = row.items()
                            random.shuffle(items)
                            row = OrderedDict(items)
                            top = dict(sorted(row.iteritems(), key=operator.itemgetter(1), reverse=True)[:topx]).keys()
                            if true_elem in top:
                                p = 1
                            else:
                                p = 0

                position += p
                counter += 1
                

        average_pos = position / counter 
        ##print "unknown elem counter", unknown_elem_counter       
        #print "counter", counter
        #print "average position", average_pos
        return average_pos
       

        
            

                    
        