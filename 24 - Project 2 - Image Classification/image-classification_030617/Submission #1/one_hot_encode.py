# The instructios say I  need to save mappings OUTSIDE of the function, like so:
listOfEncodings = [0,1,2,3,4,5,6,7,8,9]
# However, it seems possible to implement the one_hot_encode function that does NOT rely on this external mapping.
# See alternative lines commented out below.

# The implementation bloew is very most probably inefficient (I am still a Python newbie.)
# Probably wasting GPU credits...
# There is probably a more efficient "vectorized" way to do this in Python.

def one_hot_encode(x):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    # TODO: Implement Function
    n = len(x)
    one_hot = np.ndarray((n,10),dtype=int)
    # There is probably a more efficient way to do this...
    for i in range(n):
        for j in range(10):
            #if ( x[i] == j ): # Alternative Line
            if ( x[i] == listOfEncodings[j] ):
                one_hot[i][j] = 1
            else:
                one_hot[i][j] = 0
            #one_hot[i][j] = ( x[i] == j ).astype(int) # Alternative Line
            #one_hot[i][j] = ( x[i] == listOfEncodings[j] ).astype(int)
    return one_hot


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_one_hot_encode(one_hot_encode)