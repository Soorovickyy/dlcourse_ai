import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    #print(probs)
    return np.exp(predictions - np.max(predictions)) / np.sum(np.exp(predictions - np.max(predictions)))
    
    #print(np.finfo(float).eps)
    # TODO implement softmax
    # Your final implementation shouldn't have any loops
    raise Exception("Not implemented!")


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    
    
    return (- np.log(probs[target_index]))
    
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    raise Exception("Not implemented!")

def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    target = np.zeros(predictions.shape)
    
    for j in range(predictions.shape[0]):
        target[j, target_index[j]]=1
            
    
    sm = np.zeros(predictions.shape)
    for k in range(predictions.shape[0]):
        sm[k] = softmax(predictions[k])
       
    
    loss = np.zeros(target_index.shape)
    for k in range(target_index.shape[0]):
        loss[k] = cross_entropy_loss(sm[k], target_index[k])
        
    #loss = sum(loss) / len(loss)
    
    #loss = cross_entropy_loss(softmax(predictions), target_index)
        
    dprediction = np.zeros(predictions.shape)   
    for k in range(predictions.shape[0]):
        dprediction[k] = sm[k] - target[k]
        
    #print(sm)
   # print(loss)
    #print(dprediction)
    
    # TODO implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops
    
    return loss, dprediction
    
    raise Exception("Not implemented!")
    
    


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    
    
    loss = reg_strength * np.sum(W**2) / 2
    #grad  = 0.5 * reg_strength * np.sum(W**2) * W / 3
    grad = reg_strength * W
    #print(loss)
    #print(grad)
    #print(W.shape)
    #print(W)
    #print(grad.shape)
    #print(grad)
    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    
    
    return loss, grad
    #raise Exception("Not implemented!")
    
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W)
    target = np.zeros(predictions.shape)
    for j in range(predictions.shape[0]):
        target[j, target_index[j]]=1
    
    sm = np.zeros(predictions.shape)
    for k in range(predictions.shape[0]):
        sm[k] = softmax(predictions[k])
    
    loss = np.zeros(target_index.shape)
    for k in range(target_index.shape[0]):
        loss[k] = cross_entropy_loss(sm[k], target_index[k])
    
    dp = np.zeros(predictions.shape)   
    for k in range(predictions.shape[0]):
        dp[k] = sm[k] - target[k]
    
    dW = np.dot(X.T, dp)
    #print(dW)
    # TODO implement prediction and gradient over W
    # Your final implementation shouldn't have any loops
    #print(loss)
    #print(dW)
    #raise Exception("Not implemented!")

    return loss, dW
    

class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''
        
        num_train = X.shape[0]
        print(num_train)
        num_features = X.shape[1]
        print(num_features)
        num_classes = np.max(y)+1
        print(num_classes)
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)
        
        
        loss_history = []
        
        for epoch in range(epochs):
            pred = 0
            sm = np.zeros((batch_size, num_classes))
            loss_epoch = 0
            
            
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)
            batches_indices = np.array(batches_indices)
            
            for j in range(batches_indices.shape[0]):
                
                loss_history_epoch = []
                pred = np.dot(X[batches_indices[j]], self.W)
                sm = np.zeros((batch_size, num_classes))
                for i in range(sm.shape[0]):
                    sm[i] = softmax(pred[i])
                loss = np.zeros((batch_size, 1))
                grad = np.zeros((batch_size, num_features, num_classes))
                
                for k in range(loss.shape[0]):
                    target = np.zeros((1, num_classes))
                    target[:, y[batches_indices[j]][k]] = 1
                    loss[k] = cross_entropy_loss(sm[k], y[batches_indices[j]][k]) +  reg * np.sum(self.W**2)
                    grad[k, :, :] = np.dot((X[batches_indices[j]][k].T).reshape(num_features, 1), (sm[k] - target)) + 2 * reg * self.W
                grad_epoch = np.zeros((num_features, num_classes))
                grad_epoch = np.sum(grad, axis=0)
                self.W -= learning_rate * (grad_epoch)
                loss_history_epoch.append(np.sum(loss) / loss.shape[0])
            loss_epoch = sum(loss_history_epoch) / len(loss_history_epoch)
            loss_history.append(loss_epoch)
        
        #grad = np.dot(X.T, sm[k] - target[k]) + 2 * reg_strength * W
            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
            #raise Exception("Not implemented!")
        #print(loss.shape)
        
        print(grad.shape)
        print(grad)
        print(grad_epoch.shape)
        print(grad_epoch)
        #print(X.shape)
        #print(X)
        #print(sm.shape)
        #print(sm)
        print(self.W.shape)
        print(self.W)
        print(loss_history)
        #print(loss_epoch)
        #print(X[batches_indices[j]][k].T.shape)
        #print(y[batches_indices[j]][k])
        #print(target)
        #print(sm[k])
        #print((sm[k] - target).shape)
        #print(sm[k] - target)
        #print(X[batches_indices[j]][k].T)
            # end
            #print("Epoch %i, loss: %f" % (epoch, loss_epoch))

        #return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)
        print(y_pred.shape)
        # TODO Implement class prediction
        # Your final implementation shouldn't have any loops
        #raise Exception("Not implemented!")

        #return y_pred



                
                                                          

            

                
