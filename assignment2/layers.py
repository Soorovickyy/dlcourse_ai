import numpy as np


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # TODO: Copy from the previous assignment
    loss = reg_strength * np.sum(W**2) / 2
    grad = reg_strength * W
    return loss, grad

def softmax(predictions):
    return np.exp(predictions - np.max(predictions)) / np.sum(np.exp(predictions - np.max(predictions)))

def cross_entropy_loss(probs, target_index):
    return (- np.log(probs[target_index]))


def softmax_with_cross_entropy(preds, target_index):
    """
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
    """
    # TODO: Copy from the previous assignment
    target = np.zeros(preds.shape)
    
    for j in range(preds.shape[0]):
        target[j, target_index[j]]=1
            
    
    sm = np.zeros(preds.shape)
    for k in range(preds.shape[0]):
        sm[k] = softmax(preds[k])
       
    
    loss = np.zeros(target_index.shape)
    for k in range(target_index.shape[0]):
        loss[k] = cross_entropy_loss(sm[k], target_index[k])
        
    #loss = sum(loss) / len(loss)
    
    #loss = cross_entropy_loss(softmax(predictions), target_index)
        
    dprediction = np.zeros(preds.shape)   
    for k in range(preds.shape[0]):
        dprediction[k] = sm[k] - target[k]
        
    #print(sm)
   # print(loss)
    #print(dprediction)
    
    # TODO implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops
    
    

    return loss, dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """
    
    def __init__(self, value):
        
        self.value = value
        self.grad = np.zeros_like(value)
    def reset_grad(self):
        self.grad = np.zeros_like(self.value)
    

class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        #raise Exception("Not implemented!")
        
        result = np.maximum(0,X)
        self.X = X
        
        return result
        
                
    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        #raise Exception("Not implemented!")
        #dw = np.zeros(self.X.T.shape)
        
        d_result = np.array(d_out, copy=True)
        d_result[self.X < 0] = 0
        
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    
    def __init__(self, n_input, n_output):
        
        self.W = Param(0.001 * np.random.randn(n_input, n_output))   
        self.B = Param(0.001 * np.random.randn(1, n_output))   
        self.X = None
        
        
    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        #raise Exception("Not implemented!")
        #super().__init__(n_input, n_output)
        #print(X)
        #print(self.W.value)
        #print(self.B.value)
        
        result = np.dot(X, self.W.value) + self.B.value
        self.X = X
        
        
        return result
       
        

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment

        #raise Exception("Not implemented!")
        
        d_input = np.dot(d_out, self.W.value.T)
        
        self.W.grad += np.dot(self.X.T, d_out)
        self.B.grad += np.sum(d_out, axis=0, keepdims=True)
        
        
        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}
