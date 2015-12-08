# Implementation of a conditional random field model






def conditionalProbs(ysize, xdata, params, funcs):
    #makes conditional probailities of the transition y -> y' given the xdata (sequence)
    #each of the conditional probabilities is a ysize x ysize matrix

    #for now, assumes a chain of dependencies
    lambdaVals = params[0, :]  #the last lambda value never comes into play, and should always be fixed
    muVals = params[1, :]

    ffuncs = funcs[0]
    gfuncs = funcs[1]

    # params should be a 2xN matrix of values
    if params.shape[1] != xdata.shape[1]:
        raise Exception('Wrong number of parameters')
    if (len(ffuncs) != len(gfuncs)) or (len( ffuncs ) != params.shape[1]:
        raise Exception('Wrong number of functions')


    #calculate a matrix for each position in the xdata sequence:
    transitions = [0] * xdata.shape[1]
    for index = 

    energy = 


