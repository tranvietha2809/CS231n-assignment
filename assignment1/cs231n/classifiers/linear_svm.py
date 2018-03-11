import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      #we are on an incorrect class of image here, which we wants the score of this image to be lower
      #than our correct_class_score
      if margin > 0:
        #basically everytime we found an image with larger score than our correct one
        #we want to move the gradient away from that image
        #and move the gradient towards our correct image
        dW[: , j] += 1*X[i]
        dW[:, y[i]] += -X[i]
        loss += margin
  
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  # Add regularization to the loss.
  dW += reg*W
  loss += 1/2*reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  #calculus analytics for gradient found here: https://cs231n.github.io/optimization-1/
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  
  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  N = num of training example
  D = dimensions of 1 pic (for now, lets assume that D is 1)
  C = num of category (C =10)
  """
  loss = 0.0
  dW = np.zeros_like(W) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W) #score have (N, C) dimesions
  num_classes = W.shape[1]
  num_train = X.shape[0]
  #choosing the correct score
  scores_correct = scores[np.array(range(0,scores.shape[0])), y]
  #Now we need to find ALL the margin
  #https://stackoverflow.com/questions/26333005/numpy-subtract-every-row-of-matrix-by-vector
  margin = np.maximum(0, (scores.transpose()- scores_correct+1).transpose())
  #margin is a matrix where each element is the margin of 1 image compared
  #to the image of the correct class
  #margin have size (N,C)
  
  #Setting the correct class's loss to 0, because they are correct, they should not contribute to toal loss 
  margin[np.arange(num_train), y] = 0
  loss = np.sum(margin)/num_train
  loss += 1/2*reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  #gradient_count will hold binary values, rows are number of training images
  #columns are the number of classes/categories
  #0 means that the margins does not exceed
  #1 means that the margins are exceeding in incorrect classes
  gradient_count = np.zeros(margin.shape) #size (N, C)
  gradient_count[margin > 0] = 1
  #count the number of incorrect predictions that exceed the margins
  incorrect = np.sum(gradient_count, axis =1) #size (N, 1)
  #Now we need to: minus the gradient of correct class with X * number of incorrect predictions for that particular sample
  #              : plus the gradient of incorrect class with X
  gradient_count[np.arange(num_train), y] = np.negative(incorrect) 
  dW = X.transpose().dot(gradient_count)
  #dW[np.arange(num_train), y] = incorrect.dot(X[np.arange(num_train), y])
  dW /= num_train
  dW += reg*W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
