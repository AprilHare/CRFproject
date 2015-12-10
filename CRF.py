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
 
    def __init__(self, params, basisFns, fsgs):
        self.params = params
        self.basisFns = basisFns
        self.fsgs = fsgs      # number of f and g basis functions

    def findProb(self, data):
        #calculates the (conditional) probability of the given sequence
        ys = data[0,:]
        xs = data[1,:]

        matList = self.makeMats(xs)

        Zmat = self.findZ(matList)
        Z = Zmat[ys[0], ys[-1]] # this index doesn't actually matter thanks to symmetries of first and last matrices

        running = 1
        for index in range(0, len(matList)):
            prevY = ys[max( 0, index-1)]
            nextY = ys[index % len(ys)]
            running *= matList[index][ prevY, nextY ]

        return running / Z


    def updateWeights(self, data):
        #updates the weights based on the given data
        #the data is assumed to be a list of data matrices of the form above 

        # this could cause problems if the size of the data is not constant

        # we need a variety of supporting functions
        def findMessages(matList):
            #calculates the messages passed in either direction
            forward = [0]*len(matList)
            backward = [0]*len(matList)


            return [foreward, backward]


        def empiricalExp(data, funToExp):
            #calculates the empirical expectation of the function given.
            norm = float( len(data) )

            runningTotal = 0
            for entry in data:
                runningTotal += funToExp(entry)

            return runningTotal / norm


        def edgeCount(dataPoint, basisFun):
            #returns edge count associated with the given basis function
            ys = dataPoint[0,:]
            xs = dataPoint[1,:]

            count = 0
            for edge in range( len(ys) + 1):
                basisMat = basisFun(edge, xs)
                #accounts for starts and stops
                count += basisMat[ ys[ (edge % len(ys) ), ( (edge + 1) % len(ys) ) ] ]
            return count


        def predictedCount(dataPoint, basisFun):
            #returns the predicted edge count associated with the given basis function
            xs = dataPoint[1, :]

            matList = self.makeMats(xs)
            Z = self.findZ(matList)
            [foreward, backward] = findMessages( matList )

            matSize = matList[0].shape

            running = 0
            for edge in range( len(ys) + 1 ):
                # iterate through the possible values of y and y'
                for row in range(matSize[0]):
                    for column in range(matSize[1]):

                        message = foreward[edge][row] * matList


        #the empirical expected edge counts
        # ........................................................................................................
        empiricalF = [0] * fsgs[0]
        empiricalG = [0] * fsgs[1]

        for Findex in range(fsgs[0]):
            #these are the f basis vectors
            empiricalF[Findex] = empiricalExp(data, lambda x: edgeCount( x, self.basisFns[Findex]) )

        for Gindex in range(fsgs[0], sum(fsgs)):
            #these are the g basis vectors
            empiricalG[ Gindex-fsgs[0] ] = empiricalExp(data, lambda x: edgeCount( x, self.basisFns[Gindex]) )


        #the expected edge counts base on x data
        # .........................................................................................................
        expectedF = [0] * fsgs[0]
        expectedG = [0] * fsgs[1]

        for Findex in range(fsgs[0]):
            #these are the f basis vectors
            expectedF[Findex] = empiricalExp(data, lambda x: predictedCount( x, self.basisFns[Findex]) )

        for Gindex in range(fsgs[0], sum(fsgs)):
            #these are the g basis vectors
            expectedG[ Gindex-fsgs[0] ] = empiricalExp(data, lambda x: predictedCount( x, self.basisFns[Gindex]) )




    ################# utility functions ####################
    def makeMats(self, x):
        #makes matrix list from the current parameters and 

        #Basis functions are assumed to return matrices of the correct size.
        outsMats = [0]*( len(x)+1 )

        for iMat in range( len(x) + 1):
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


