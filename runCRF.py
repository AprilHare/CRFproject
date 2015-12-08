# runs the CRF

import numpy as np

import CRF
reload(CRF)
from CRF import *

#       0     1
#   y0 -- y1 -- y2
#   |1    |2    |3
#   x0    x1    x2 


def generateBasis(yvals, xvals):
    #generates the HMM like feature functions

    #Each feature vector corresponds a matrix with a single one in one location
    ffuncs = []
    for i1 in range(yvals):
        for i2 in range(yvals):
            matTemplate = np.zeros( (yvals, yvals) )
            matTemplate[i1, i2] = 1
            ffuncs.append( lambda e, x: matTemplate)


    gfuncs = []
    for i1 in range(xvals):
        for i2 in range(yvals):
            matTemplate = np.zeros( (xvals, yvals) )
            matTemplate[i1, i2] = 1
            print matTemplate
            gfuncs.append( lambda e, x: matTemplate[x, :] )

    return ffuncs + gfuncs

