# Implementation of a conditional random field model

# we use the convention that the 'underlying' data is in the first row
# and the observations are in the second
#
#        0     1     2     3
# start -- y0 -- y1 -- y2 -- stop
#          | 0   | 1   | 2
#          x0    x1    x2 
#
# becomes:
# [ [y1, y2,...]
#   [x1, x2,...] ]

import numpy as np
import math

class CRF(object):
    """ Implements a graphical model as a list of transition matrices conditioned on x"""
 
    def __init__(self, params, basisFns, fsgs, numYnumX):
        self.params = params
        self.basisFns = basisFns
        self.fsgs = fsgs      # number of f and g basis functions  (numf, numg)
        self.numYnumX = numYnumX        #num of values for y and x  (numY, numX)

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

    def findLabels(self, xdata):
        # implements a Viterbi algorithm to determine the most likely Y sequence
        #  given current parameters and the X sequence
        Yoptions = self.numYnumX[0]
        matList = self.makeMats(xdata)

        #There is one path for each Y option
        paths = [ (i,) for i in range(Yoptions) ]
        nextPaths = [ (i,) for i in range(Yoptions) ]

        probs = matList[0][0,:]  # the first matrix has row symmetry
        nextProbs = matList[0][0,:]

        for transMat in matList[1:-1]:
            # We want the probabilities of paths that end in each value y.
            transProbs = probs * transMat.T

            bestPath = np.argmax(transProbs, 1)
            bestProbs = np.max( transProbs, 1)
            for terminal in range(Yoptions):
                #print paths, probs
                currBest = bestPath[terminal] # this is the best path to the terminal value
                pathToTerminal = paths[currBest] + (terminal, )

                nextPaths[terminal] = pathToTerminal
                nextProbs[terminal] = bestProbs[terminal]

            paths[:] = nextPaths[:]
            probs[:] = nextProbs[:]

            #print paths, probs
        #choose a path'
        print paths
        finalProbs = probs * matList[-1].T
        finalIndex = np.argmax( finalProbs, 1)[0] # there should be symmetry

        return list( paths[finalIndex] )


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

                #print basisMat, rowIndex, colIndex
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


        matMemo = {}
        def fPredictedCount(dataPoint, basisFun):
            #returns the predicted edge count associated with the given basis function
            xs = dataPoint[1, :]

            # attempt to memoize for speed
            if tuple(xs) in matMemo:
                (matList, Z, [forward, backward]) = matMemo[tuple(xs)]

            else:
                matList = self.makeMats(xs)
                Z = self.findZ(matList)
                [forward, backward] = findMessages( matList )

                matMemo[tuple(xs)] = (matList, Z, [forward, backward])


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

                        message = (forward[edge][rowIndex] * matList[edge][rowIndex, columnIndex] * backward[edge + 1][columnIndex] ) / Z[0,0]
                        basis = basisFun(edge, xs)[rowIndex, columnIndex]
                        running += message * basis

            return running


        def gPredictedCount(dataPoint, basisFun):
            #returns the predicted edge count associated with the given basis function
            xs = dataPoint[1, :]

            # attempt to memoize for speed
            if tuple(xs) in matMemo:
                (matList, Z, [forward, backward]) = matMemo[tuple(xs)]

            else:
                matList = self.makeMats(xs)
                Z = self.findZ(matList)
                [forward, backward] = findMessages( matList )

                matMemo[tuple(xs)] = (matList, Z, [forward, backward])


            matSize = matList[0].shape

            running = 0
            for edge in range( len(xs) ):
                # iterate through the possible values of y and y'
                for column in range(matSize[0]):
                    #print row, column, basisFun(edge, xs).shape
                    message = (forward[edge + 1][column] * backward[edge + 1][column] ) / Z[0, 0]
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
            #print 'f emp', Findex
            empiricalF[Findex] = empiricalExp(data, lambda x: fEdgeCount( x, self.basisFns[Findex]) )

        for Gindex in range(numFs, totalBases):
            #these are the g basis vectors
            #print 'g emp', Gindex
            empiricalG[ Gindex-numFs ] = empiricalExp(data, lambda x: gEdgeCount( x, self.basisFns[Gindex]) )

        empirical = empiricalF + empiricalG

        #print empirical

        #the expected edge counts base on x data
        # .........................................................................................................
        expectedF = [0] * numFs
        expectedG = [0] * numGs

        for Findex in range(numFs):
            #these are the f basis vectors
            #print 'f exp', Findex
            expectedF[Findex] = empiricalExp(data, lambda x: fPredictedCount( x, self.basisFns[Findex]) )

        for Gindex in range(numFs, totalBases):
            #these are the g basis vectors
            #print 'g exp', Gindex
            expectedG[ Gindex-numFs ] = empiricalExp(data, lambda x: gPredictedCount( x, self.basisFns[Gindex]) )


        expected = expectedF + expectedG

        #print expected

        S = len(self.basisFns)

        #update parameters
        for index in range( len (self.params) ):
            #print empirical[index], expected[index]
            #print float( empirical[index] ) / expected[index] 
            self.params[index] += math.log( (empirical[index] + 1E-4 ) / (expected[index] + 1E-4) ) / S



    ################# utility functions ####################
    def makeMats(self, x):
        #makes matrix list from the current parameters and x
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


