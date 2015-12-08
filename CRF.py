# Implementation of a conditional random field model

# we use the convention that the 'underlying' data is in the first row, and the observations are in the second'
#
#     -- y1 --- y2 --- ...
#         |     |
#        x1     x2
# becomes:
# [ [y1, y2,...]
#   [x1, x2,...] ]

import numpy as np


class CRF(object):
    """ Implements a graphical model as a list of transition matrices conditioned on x"""
 
    def __init__(self, params, basisFns):
        self.params = params
        self.basisFns = basisFns


    def findProb(self, data):
        #calculates the (conditional) probability of the given sequence
        ys = data[0,:]
        xs = data[1,:]

        matList = makeMats(x)
        Zmat = findZ(xs, matList)
        Z = Zmat[ys[0], ys[-1]]

        running = 1
        for index in range(1, len(self)+1):
            running *= self.matList[index](xs)[ ys[index-1], ys[index] ]

        return running / Z


    def updateWeights(self, data, delta):
        #updates the weights based on the given data
        pass


    ################# utility functions ####################
    def makeMats(x):
        #makes matrix list from the current parameters and 

        #Basis functions are assumed to return matrices of the correct size.
        outsMats = [0]*len(x)

        for iMat in range( len(x) ):
            currMatrix = 0
            for k in range(len(params)):
                currMatrix = currMatrix + params[k] * basisFns[k](iMat, x)
            outsMats[iMat] = np.exp( currMatrix )


    def findZ(x, matList):
        #finds the normalization for the pdf conditioned on x
        running = matList[0](x)
        for entry in matList[1:]:
            # update the running matrix
            running = np.dot(running, entry(x) )

        return running


