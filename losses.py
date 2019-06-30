from tensorflow import abs, reduce_mean, log


def log_mae(labels, predictions):
    """
    Log of mae loss function

    :param labels: tensor of labels (type: tf.Tensor)
    :param predictions: tensor of predictions (type: tf.Tensor)
    """

    result = abs(labels - predictions)
    result = reduce_mean(result)
    result = log(result)

    return result
