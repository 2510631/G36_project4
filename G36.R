# Practical 4   Group 36

# The aim of this project is to write functions to set up a neural network for 
# classification, and to train it using stochastic gradient descent. More specifically,
# the project implement a neural network for the classification of iris species 
# based on their characteristics.

# Five functions can be seen in this project, including 'netup' function for setting
# up the network architecture, 'forward' and 'backward' functions for performing
# forward and backward propagation, 'train' function for training the network and 
# 'predict' function for making predictions on new data.


netup = function (d){
  # The purpose of this function is to initialize a neural network with the 
  # specified layer sizes, providing initial values for nodes, weight matrices, and offset vectors.
  
  # Input:
  # d: A vector specifying the number of nodes in each layer of the neural network.
  
  # Output:
  # A list containing:
  # 'h': A list of vectors representing the node values for each layer.
  # 'W': A list of matrices representing the weight matrices linking consecutive layers.
  # 'b': A list of vectors representing the offset vectors linking consecutive layers.
  # 'd': The input vector specifying the layer sizes.
  
  
  
  # Initialize lists for nodes 'h', weight matrices 'W', and offset vectors 'b'
  h <- W <- b <- list()
  
  # Iterate through layers to initialize nodes, weight matrices, and offset vectors
  for ( i in 1:(length(d) - 1)){
    
    # Initialize node values for layer i
    h[i] <- list(rep(0, d[i]))
    
    # Initialize weight matrix W[i] with U(0, 0.2) random deviates
    W[i] <- list(matrix(runif(d[i] * d[i+1], 0, 0.2), d[i+1]))
    
    # Initialize offset vector b[i] with U(0, 0.2) random deviates
    b[i] <- list(runif(d[i+1], 0, 0.2))    
  }
  # Return a list containing nodes 'h', weight matrices 'W', offset vectors 'b', and the layer sizes 'd'
  return(list(h = h, W = W, b = b, d = d))
}


forward <- function(nn,inp){
  # The purpose of the forward function is to perform forward propagation in a 
  # neural network, computing the node values for each layer based on the input 
  # values for the first layer.
  
  # Input:
  # nn: A network list containing information about the neural network, as 
  #     returned by the netup function.
  # inp: A vector of input values for the first layer.
  
  # Output:
  # nn: An updated network list containing the computed node values for each layer
  # after forward propagation.
  
  
  # Obtain elements needed from the network list nn.
  W <- nn$W
  b <- nn$b
  d <- nn$d
  h <- nn$h
  
  # Set the input values for the first layer
  h[1] <- list(inp)
  for (i in 1:length(b)){
    
    # Compute the linear transformation
    z=matrix(unlist(W[i]),d[i+1]) %*% unlist(h[i]) + unlist(b[i])
    
    # Apply the ReLU activation function
    z[z < 0] <- 0
    
    # Update the node values for the next layer
    h[i+1] <- list(z)
  }
  
  # Return the updated network list
  return(nn = list(h=h,W=W,b=b,d=d)) 
}

backward=function(nn,k){ 
  # The purpose of the backward function is to compute derivatives of the loss 
  # corresponding to output class k for a neural network, including derivatives 
  # with respect to nodes, weights, and offsets.
  
  # Input:
  # nn: A network list containing information about the neural network, as returned by the forward function.
  # k: The output class for which the derivatives are computed.
  
  # Output:
  # An updated network list containing the computed derivatives dh, dW, and db.
  
  
  # Obtain elements needed from the network list nn.
  h <- nn$h
  W <- nn$W
  b <- nn$b
  d <- nn$d
  
  # Get the output values for the last layer
  output=unlist(h[length(d)])
  
  # Initialize lists for derivatives dh, dW, and db
  dW=dh=db=list()
  
  # Iterate through layers in reverse order to compute derivatives using 
  # back-propagation
  for (i in length(d):1){
    
    if(i==length(d)){
      
      # Compute the derivative of the loss for output class k w.r.t. the nodes 
      # in the output layer.
      Dh <- exp(output) / sum(exp(output))
      Dh[k] <- Dh[k]-1
      dh[i] <- list(Dh)
    } else {
      
      # Compute derivatives w.r.t. nodes, weights, and offsets for each layer 
      # using back-propagation
      d_l1=unlist(dh[i+1])*(unlist(h[i+1])>0)
      
      dh[i]=list(t(matrix(unlist(W[i]),d[i+1]))%*%d_l1)
      
      dW[i]=list(matrix(d_l1,d[i+1]) %*% matrix((unlist(h[i])),ncol=d[i]))
      
      db[i]=list(d_l1)
    }
  }
  
  # Return the updated network list with derivatives
  return(list(h=h,W=W,b=b,d=d,dh=dh,dW=dW,db=db))
}

train=function(nn,inp,k,eta=.01,mb=10,nstep=10000){ 
  # The aim of the train function is to train a neural network using stochastic gradient descent.
  # It performs iterations over the input data with minibatches, computes the gradients, 
  # and updates the parameters of the neural network to minimize the loss.
  
  # Input:
  # nn: A network list containing information about the neural network.
  # inp: Input data in the rows of a matrix, where each row represents an input sample.
  # k: Corresponding labels for the input data.
  # eta: Step size for gradient descent (default is 0.01).
  # mb: Number of data to randomly sample to compute the gradient (default is 10).
  # nstep: Number of optimization steps to take (default is 10000).
  
  # Output:
  # An updated network list after training.
  
  
  
  for (i in 1:nstep){
    
    # Randomly sample a minibatch of data
    index=sample(dim(inp)[1],mb)
    data_train=inp[index,]
    k_train=k[index]
    
    # Extract elements from the network list
    d=nn$d ;b=nn$b ;W=nn$W 
    
    # Initialize lists for averaging gradients
    aver_dW=list()
    aver_db=list()
    
    
    for (j in 1:length(b)){
      
      # Initialize averaged gradients
      aver_dW[j]=list(matrix(0,d[j],d[j+1]))
      aver_db[j]=list(rep(0,d[j+1]))
    }
    
    # Compute gradients and accumulate for each sample in the minibatch
    for (s in 1:mb){
      
      # Do the forward and backward propagation for each sample in the minibatch
      nn_1=forward(nn,data_train[s,])
      nn_1=backward(nn_1,k_train[s])
      
      # Update dW and db
      dW=nn_1$dW ;db=nn_1$db
      
      # Accumulate gradients
      for (j in 1:length(b)){ 
        
        # Compute the average gradients for each layer
        aver_dW[j]=list(matrix(unlist(dW[j]),d[j])/mb+matrix(unlist(aver_dW[j]),d[j]))
        aver_db[j]=list(unlist(db[j])/mb+unlist(aver_db[j]))
      }
    }
    
    # Update parameters using the averaged gradients
    for (j in 1:length(b)){
      
      # Update weights
      W[j]=list(matrix(unlist(W[j]),d[j])-eta*matrix(unlist(aver_dW[j]),d[j]))
      
      # Update offsets
      b[j]= list(unlist(b[j])-eta*unlist(aver_db[j]))
    }
    
    # Update the network list with the newest parameters
    nn$W=W ;nn$b=b
  }  
  
  # Return the updated network list after training
  return(nn)
}


predict=function(nn,inp){
  # The aim of the 'predict' function is to use a trained neural network to predict
  # the class labels of input data samples.
  # This function takes the node values of the output layer obtained through a 
  # forward pass and converts them into probabilities. 
  # The most probable one is then selected as the predicted class.
  
  # Input:
  # nn: A trained neural network model, containing useful parameters.
  # inp: Input data matrix where each row represents a data sample, and columns represent features.
  
  # Output:
  # prediction: A vector containing the predicted class labels for each input data sample.
  
  
  
  # Set up an empty vector to store predictions later
  prediction=c()
  
  
  for (i in 1:dim(inp)[1]){
    
    # Do the forward propagation to get the node values of the output layer
    h=forward(nn,inp[i,])$h
    
    # Extract the output layer node values and convert to a vector
    h=unlist(h[length(h)])
    
    # Obtain the probabilities
    p=exp(h)/sum(exp(h))
    
    # Set the predicted class as the most probable one.
    prediction[i]= which.max(p)
  }
  
  # Return the vector of predictions
  return(prediction)
}


# Set the network to be 4,8,7,3
d=c(4,8,7,3)

# Assign numeric labels to iris species
levels(iris[,5])=1:3
label=as.numeric(iris[,5])

# Obtain training data by excluding every 5th row
train_data=iris[-seq(5,150,5),1:4]
train_k=label[-seq(5,150,5)]

# Obtain test data by using every 5th row
test_data=iris[seq(5,150,5),1:4]
test_k=label[seq(5,150,5)]

# Set seed for reporducibility during training
set.seed(6666)

# Train the neural network using stochastic gradient descent
nn=train(netup(d),train_data,train_k,nstep=10000)
print(nn)

# Predict iris species on the test data by using the trained network
prediction=predict(nn,test_data)
print(prediction)

# Calculate the accuracy
accuracy=mean(prediction==test_k)
print(accuracy)