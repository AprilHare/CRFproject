# Generates examples for the label bias problem

import numpy as np
import math

def HMMgenerator(hiddenCDF, observationCDF, numNodes):
    #generates examples from a hidden markov model
    #hidden CDF gives p(a_t | a_t-1)
    #observation CDF gives p(o_t | a_t)

    hidden = np.zeros( numNodes )
    observation = np.zeros( numNodes )

    for index in range(len(hidden)- 1):
        hidden[index+ 1] = hiddenCDF( hidden[index] )
        observation[index] = observationCDF(hidden[index] )

    # last observation
    observation[ index + 1] = observationCDF( hidden[index+1])

    return hidden, observation

def hiddenCDF(number):
    #outputs a random integer
    return np.random.randint( number -1 , number + 4)

def observationCDF(number):
    #outputs a random integer
    return np.random.randint( number - 2, number + 2) 
