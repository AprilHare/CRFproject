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

        matList = self.makeMats(xs)

        Zmat = self.findZ(matList)
        Z = Zmat[ys[0], ys[-1]]

        running = 1
        for index in range(1, len(matList)):
            running *= matList[index][ ys[index-1], ys[index] ]

        return running / Z


    def updateWeights(self, data, delta):
        #updates the weights based on the given data
        pass


    ################# utility functions ####################
    def makeMats(self, x):
        #makes matrix list from the current parameters and 

        #Basis functions are assumed to return matrices of the correct size.
        outsMats = [0]*len(x)

        for iMat in range( len(x) ):
            currMatrix = 0
            for k in range(len(self.params)):
                currMatrix = currMatrix + self.params[k] * self.basisFns[k](iMat, x)
            outsMats[iMat] = np.exp( currMatrix )

        return outsMats


    def findZ(self, matList):
        #finds the normalization for the pdf conditioned on x
        running = matList[0]
        for entry in matList[1:]:
            # update the running matrix
            running = np.dot(running, entry )

        return running


