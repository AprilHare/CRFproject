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


def ribGen():
    state = 0

    transMat = np.zeros((6, 6))
    transMat[1, 0] = 0.7
    transMat[2, 0] = 0.3

    transMat[3, 1] = 1
    transMat[4, 2] = 1

    transMat[5, 3] = 1
    transMat[5, 4] = 1


    emits = {1: 'r', 2:'r', 3: 'i', 4: 'o', 5:'b' }
    other = ['r', 'i', 'o', 'b']

    code = {'b': 0, 'i': 1, 'o':2, 'r':3}

    state = 0
    toEmits = ''
    actual = ''

    encodedEmits = []
    encodedActual = []

    while state < 5:
        nextState = np.random.choice(range(0, 6), p = transMat[:, state])
        state = nextState

        if np.random.rand() < (28./ 32):
            toEmits += emits[nextState]
            encodedEmits.append( code[emits[nextState]] )
        else:
            out = np.random.choice(other)
            toEmits += out
            encodedEmits.append( code[out] )


        actual += emits[nextState]
        encodedActual.append( code[emits[nextState]] ) 

    return (toEmits, actual), (encodedEmits, encodedActual)  #, np.array( [ encodedEmits, encodedActual ] )

train = [ ribGen()[1] for i in range(2000)] 
test = [ribGen()[1] for i in range(500)]





