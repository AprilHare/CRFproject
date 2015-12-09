# runs the CRF

import numpy as np

import CRF
reload(CRF)
from CRF import *

#        0     1     2     3
# start -- y0 -- y1 -- y2 -- stop
#          |0    |1    |2
#         x0    x1    x2 



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

    gmatList = []
    for i1 in range(xvals):
        for i2 in range(yvals):
            matTemplate = np.zeros( (xvals, yvals) )
            matTemplate[i1, i2] = 1
            gmatList.append( matTemplate )


    #functions to generate the functions we want:
    def fmaster(e, i, x): 
        return fmatList[i]

    def gmaster(e, i, x):
        index = x[e]
        return gmatList[i][ index, : ] 

    fgen = lambda i : ( lambda e, x : fmaster(e,i,x) )
    ggen = lambda i : ( lambda e, x : gmaster(e,i,x) )

    flist = [fgen(i) for i in range(yvals*yvals)]
    glist = [ggen(i) for i in range(xvals*yvals)]


    #the pdfs for transfers from and to the initial and final states


    return flist + glist



basisFns = generateBasis(2,2)
initialParams = 0*np.random.rand(len(basisFns))

model = CRF( initialParams, basisFns )
data = np.array( [[0,1], [0,1]])



