# runs the CRF

import numpy as np

import CRF
reload(CRF)
from CRF import *

#        0     1     2     3
# start -- y0 -- y1 -- y2 -- stop
#          |0    |1    |2
#          x0    x1    x2 



def generateBasis(yvals, xvals):
    #generates the HMM like feature functions

    #Each feature vector corresponds a matrix with a single one in one location

    #generate lists of the matrices that we want to output
    fmatList = []
    for i1 in range(yvals):
        for i2 in range(yvals):
            matTemplate = np.zeros( (yvals, yvals) )
            matTemplate[i1, i2] = 1
            fmatList.append(matTemplate)


    #the pdfs for transitions from and to the initial and final states
    initialList = []
    for i1 in range(yvals):
        matTemplate = np.zeros( (1, yvals) )
        matTemplate[0, i1 ] = 1
        initialList.append( matTemplate)

    finalList = []
    for i1 in range(yvals):
        matTemplate = np.zeros( (yvals, 1) )
        matTemplate[i1, 0 ] = 1
        finalList.append( matTemplate)


    gmatList = []
    for i1 in range(xvals):
        for i2 in range(yvals):
            matTemplate = np.zeros( (xvals, yvals) )
            matTemplate[i1, i2] = 1
            gmatList.append( matTemplate )


    #functions to generate the functions we want:
    def fmaster(e, i, x):
        if (e == 0) and ( i < len(initialList) ):
            #it is the first edge and one of the basis vectors
            return initialList[i]

        elif (e >= len(x) ) and (i >= len(initialList) + len(fmatList) ):
            #it is the last edge
            index = i - len(initialList) - len(fmatList)
            return finalList[index]

        elif (e > 0) and ( e < len(x) ) and ( i >= len( initialList)) and ( i < (len(initialList) + len(fmatList) ) ):
            #it is an internal edge
            index = i - len(initialList)
            return fmatList[index]

        else:
            #failure case
            return 0*fmatList[0]

    def gmaster(e, i, x):
        if e >= len(x):
            return 0 * gmatList[0][0,:]

        index = x[e]
        return gmatList[i][ index, : ] 

    fgen = lambda i : ( lambda e, x : fmaster(e,i,x) )
    ggen = lambda i : ( lambda e, x : gmaster(e,i,x) )

    flist = [fgen(i) for i in range(yvals + yvals*yvals + yvals)]
    glist = [ggen(i) for i in range(xvals*yvals)]


    return flist + glist, (len(flist), len(glist))



basisFns, fsgs = generateBasis(2,2)
initialParams = 0*np.random.rand(len(basisFns))

model = CRF( initialParams, basisFns, fsgs )
data = np.array( [[0,1], [0,1]])



