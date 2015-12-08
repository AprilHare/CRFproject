#!!!!!!!!!!!!! old file!!!!!!!!!!!!!!!!!!!!

#implements graphical model classes

# we use the convention that the 'underlying' data is in the first row, and the observations are in the second'
#
#     - y1 --- y2 --- ...
#         |     |
#        x1     x2
# becomes:
# [ [y1, y2,...]
#   [x1, x2,...] ]

import numpy as np



class CRF(list):
    """ Implements a graphical model as a list of transition matrices conditioned on x"""
 
    def __init__(self, params, basisFns):
        # calculate the initial matrix entries


        super(graphicalModel, self).__init__(initialMats)
        self.params = params
        self.basisFns = basisFns


    def findProb(self, data):
        #calculates the (conditional) probability of the given sequence
        ys = data[0,:]

        Zmat = self.findZ(data[1,:])
        Z = Zmat[ys[0], ys[-1]]

        for index in range(len(self)):




    def updateWeights(self, data):
        #updates the weights based on the given data
        pass


    ################# utility functions ####################
    def findZ(self, x):
        #finds the normalization for the pdf conditioned on x
        running = self[0](x)
        for entry in self[1:]:
            # update the running matrix
            running = np.dot(running, entry(x) )

        return running



