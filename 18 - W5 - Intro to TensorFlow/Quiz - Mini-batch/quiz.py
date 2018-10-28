import math
def batches(batch_size, features, labels):
    """
    Create batches of features and labels
    :param batch_size: The batch size
    :param features: List of features
    :param labels: List of labels
    :return: Batches of (Features, Labels)
    """
    assert len(features) == len(labels)
    # TODO: Implement batching
    num_batches    = math.ceil(len(features)/batch_size)
    num_items_left = len(features)
    for i in range(num_batches):
        # Check whether we have enough items left for a full batch size.
        if ( num_items_left > batch_size ):
            next_batch_size = batch_size
        else:
            next_batch_size = num_items_left # last batch
            
        # Update number of items left
        num_items_left = num_items_left - next_batch_size

        # Grab next batch of features.
        startIndex = ( i * batch_size )
        endIndex   = startIndex + next_batch_size
        #print([startIndex,endIndex])
        next_features = features[startIndex:endIndex]
        next_labels   = labels[startIndex:endIndex]
        if ( i == 0 ):
            batch_list = [[next_features,next_labels]]
        else:
            batch_list.append([next_features,next_labels])
    return(batch_list)