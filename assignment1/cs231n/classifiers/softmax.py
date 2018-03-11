import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength
    
    N: number of training example
    C: number of class
    D: dimension of 1 training example
  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # you can find the derivative here: https://cs231n.github.io/neural-networks-case-study/#linear
  # using the chain rule, we can further: d(L)/d(W) = (pk- 1{yi = k})xi
  loss_matrix = np.zeros_like(W)
  for i in xrange(num_train):
      scores = X[i].dot(W) #(1,C) array
      scores -= np.max(scores) #normalized scores, to avoid instability
      # calculate loss of each training elements
      p = -np.log(np.exp(scores[y[i]])/np.sum(np.exp(scores))) 
      loss += p
      for k in range(num_classes):
          p_k = np.exp(scores[k])/np.sum(np.exp(scores))
          if k == y[i]:
            dW[:, k] += (p_k - 1) * X[i]
          else:
            dW[:, k] += p_k *X[i]
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  
  ### why do we have to divide by num_train here???
  dW /= num_train
  dW += reg*W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  unlog_scores = X.dot(W) #(N,C) array
  unlog_scores = (unlog_scores.transpose() - np.max(unlog_scores, axis = 1)).transpose() #normalized score
  scores = np.exp(unlog_scores) #(N,C) array
  sum_scores_each_elements = np.sum(scores, axis = 1) #(N, 1) array
  loss = -np.log(scores[np.arange(num_train),y]/sum_scores_each_elements)

  #return a matrix with a 1 in correct class correspond to the example training
  
  p_k = np.divide(scores[np.arange(num_train), :].transpose(),sum_scores_each_elements).transpose() #(N,C) array
  p_k[np.arange(num_train), y] -= 1 #minus 1 at the correct class corresponding to the elements
  dW =  X.transpose().dot(p_k)
  loss = np.sum(loss)/num_train
  loss += 0.5*reg*np.sum(W*W)
  
  dW /= num_train
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

