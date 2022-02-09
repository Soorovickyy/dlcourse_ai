import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization, softmax


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        self.FullyConnected1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.ReLU = ReLULayer()
        self.FullyConnected2 = FullyConnectedLayer(hidden_layer_size, n_output)
        
        # TODO Create necessary layers
        #raise Exception("Not implemented!")

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        #raise Exception("Not implemented!")
        
        for param in self.params().values(): 
            param.grad = np.zeros_like(param.grad)
            
            
        
        layer_output = self.FullyConnected1.forward(X)
        layer_output = self.ReLU.forward(layer_output)
        layer_output = self.FullyConnected2.forward(layer_output)
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        loss, grad = softmax_with_cross_entropy(layer_output, y)
        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        #raise Exception("Not implemented!")
        grad = self.FullyConnected2.backward(grad)
        
        grad = self.ReLU.backward(grad)
        
        grad = self.FullyConnected1.backward(grad)
        
        for param in self.params().values():
            loss_l2, grad_l2 = l2_regularization(param.value, self.reg)
            loss += loss_l2
            param.grad += grad_l2
        #print(loss)
        #print(grad)
        
        return loss
        

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        #pred = np.zeros(X.shape[0], np.int)
        pred = self.FullyConnected1.forward(X)
        pred = self.ReLU.forward(pred)
        pred = self.FullyConnected2.forward(pred)
        pred = softmax(pred)
        pred = np.argmax(pred, axis=1)

        #raise Exception("Not implemented!")
        return pred

    def params(self):
        result = {'W1': self.FullyConnected1.W, 'B1': self.FullyConnected1.B, 'W2': self.FullyConnected2.W, 'B2': self.FullyConnected2.B}
        
        # TODO Implement aggregating all of the params

        #raise Exception("Not implemented!")

        return result
