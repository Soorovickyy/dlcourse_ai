def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    tp = sum((prediction == True) & (ground_truth == True))
    fp = sum((prediction == True) & (ground_truth == False))
    fn = sum((prediction == False) & (ground_truth == True))
    tn = sum((prediction == False) & (ground_truth == False))
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    tp = sum((prediction == True) & (ground_truth == True))
    fp = sum((prediction == True) & (ground_truth == False))
    fn = sum((prediction == False) & (ground_truth == True))
    tn = sum((prediction == False) & (ground_truth == False))
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return accuracy
