#implements graphical model classes

# we use the convention that the 'underlying' data is in the first row, and the observations are in the second'
#
#     - y1 --- y2 --- ...
#         |     |
#        x1     x2
# becomes:
# [[y1, y2,...]
#   [x1, x2,...]]






class sequenceData(object):
    """sequenceData an individual data point for training HMMs MEMMs, and CRFs"""
    def __init__(self, seen, underlying):
        # intuitively seen is the structure that we are observing
        # underlying is the structure that we are trying to learn
        if len(seen) != len(underlying):
            raise Exception("Sequence lengths don't match")
        self.seen = seen
        self.underlying = underlying


