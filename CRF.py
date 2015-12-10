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
import math

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
        def empiricalExp(data, funToExp):
            #calculates the empirical expectation of the function given.
            norm = float( len(data) )

            runningTotal = 0
            for entry in data:
                runningTotal += funToExp(entry)

            return runningTotal / norm


        def fEdgeCount(dataPoint, basisFun):
            #returns edge count associated with the given basis function
            ys = dataPoint[0,:]
            xs = dataPoint[1,:]

            count = 0
            for edge in range( len(ys) + 1):
                basisMat = basisFun(edge, xs)
                #accounts for starts and stops
                #print 'edge', edge, basisMat.shape

                if edge == 0:
                    rowIndex = 0
                    colIndex = ys[edge]
                elif edge == len(ys):
                    rowIndex = ys[edge - 1]
                    colIndex = 0
                else:
                    rowIndex = ys[ edge - 1 ]
                    colIndex = ys[ edge ]

                count += basisMat[ rowIndex, colIndex ]
            return count


        def gEdgeCount(dataPoint, basisFun):
            #returns edge count associated with the given basis function
            ys = dataPoint[0,:]
            xs = dataPoint[1,:]

            count = 0
            for edge in range( len(ys)):
                basisMat = basisFun(edge, xs)
                #accounts for starts and stops
                #print 'edge', edge, basisMat.shape

                count += basisMat[ ys[edge] ]
            return count


        def findMessages(matList):
            #calculates the messages passed in either direction
            forward = [0]* ( len(matList) + 1)
            backward = [0]* ( len(matList) + 1)

            for index in range( len(matList) +1 ):
                #print'for', index
                if index == 0:
                    #watch out for dimension changing weirdness
                    nextFor = np.ones(matList[0].shape[1])
                else:
                    currFor = nextFor
                    nextFor = np.dot( currFor, matList[index-1])
                forward[index] = nextFor

            for index in range(len(matList), -1, -1):
                #print 'back', index
                if index == len(matList):
                    nextBack = np.ones(matList[0].shape[0])
                else:
                    currBack = nextBack
                    nextBack = np.dot( matList[index], currBack)
                backward[index-1] = nextBack

            return [forward, backward]



        def fPredictedCount(dataPoint, basisFun):
            #returns the predicted edge count associated with the given basis function
            xs = dataPoint[1, :]

            matList = self.makeMats(xs)
            Z = self.findZ(matList)
            [foreward, backward] = findMessages( matList )

            matSize = matList[0].shape

            running = 0
            for edge in range( len(xs) + 1 ):
                # iterate through the possible values of y and y'
                for row in range(matSize[0]):
                    for column in range(matSize[1]):
                        #print row, column, basisFun(edge, xs).shape

                        if edge == 0:
                            rowIndex = 0
                            columnIndex = column
                        elif edge == len(xs):
                            rowIndex = row
                            columnIndex = 0
                        else:
                            rowIndex = row
                            columnIndex = column

                        message = (foreward[edge][rowIndex] * matList[edge][rowIndex, columnIndex] * backward[edge + 1][columnIndex] ) / Z[0,0]
                        basis = basisFun(edge, xs)[rowIndex, columnIndex]
                        running += message * basis

            return running



        def gPredictedCount(dataPoint, basisFun):
            #returns the predicted edge count associated with the given basis function
            xs = dataPoint[1, :]

            matList = self.makeMats(xs)
            Z = self.findZ(matList)
            [foreward, backward] = findMessages( matList )

            matSize = matList[0].shape

            running = 0
            for edge in range( len(xs) ):
                # iterate through the possible values of y and y'
                for column in range(matSize[0]):
                    #print row, column, basisFun(edge, xs).shape
                    message = (foreward[edge + 1][column] * backward[edge + 1][column] ) / Z[0, 0]
                    basis = basisFun(edge, xs)[column]
                    running += message * basis


            return running


        numFs = self.fsgs[0]
        numGs = self.fsgs[1]
        totalBases = sum( self.fsgs)
        #the empirical expected edge counts
        # .........................................................................................................
        empiricalF = [0] * numFs
        empiricalG = [0] * numGs

        for Findex in range(numFs):
            #these are the f basis vectors
            #print 'f', Findex
            empiricalF[Findex] = empiricalExp(data, lambda x: fEdgeCount( x, self.basisFns[Findex]) )

        for Gindex in range(numFs, totalBases):
            #these are the g basis vectors
            #print 'g', Gindex
            empiricalG[ Gindex-numFs ] = empiricalExp(data, lambda x: gEdgeCount( x, self.basisFns[Gindex]) )

        empirical = empiricalF + empiricalG

        print empirical

        #the expected edge counts base on x data
        # .........................................................................................................
        expectedF = [0] * numFs
        expectedG = [0] * numGs

        for Findex in range(numFs):
            #these are the f basis vectors
            #print 'f', Findex
            expectedF[Findex] = empiricalExp(data, lambda x: fPredictedCount( x, self.basisFns[Findex]) )

        for Gindex in range(numFs, totalBases):
            #print 'g', Gindex
            #these are the g basis vectors
            expectedG[ Gindex-numFs ] = empiricalExp(data, lambda x: gPredictedCount( x, self.basisFns[Gindex]) )

        expected = expectedF + expectedG


        S = len(self.basisFns)

        #update parameters
        for index in range( len (self.params) ):
            #print empirical[index], expected[index]
            #print float( empirical[index] ) / expected[index] 
            self.params[index] += math.log( (empirical[index] + 1E-4 ) / expected[index] ) / S



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


