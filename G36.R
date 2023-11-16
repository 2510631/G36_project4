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
  h=nn$h ;W=nn$W ;b=nn$b ;d=nn$d
  output=unlist(h[length(d)])
  dW=dh=db=list()
  for (i in length(d):1){
    if(i==length(d)){
      Dh=exp(output)/sum(exp(output))
      Dh[k]=Dh[k]-1
      dh[i]=list(Dh)
    } else {
      d_l1=unlist(dh[i+1])*(unlist(h[i+1])>0)
      dh[i]=list(t(matrix(unlist(W[i]),d[i+1]))%*%d_l1)
      dW[i]=list(matrix(d_l1,d[i+1])%*%matrix((unlist(h[i])),ncol=d[i]))
      db[i]=list(d_l1)
    }
  }
  return(list(h=h,W=W,b=b,d=d,dh=dh,dW=dW,db=db))
}

train=function(nn,inp,k,eta=.01,mb=10,nstep=10000){ 
  for (i in 1:nstep){
    index=sample(dim(inp)[1],mb)
    data_train=inp[index,]
    k_train=k[index]
    d=nn$d ;b=nn$b ;W=nn$W 
    aver_dW=list()
    aver_db=list()
    for (j in 1:length(b)){
      aver_dW[j]=list(matrix(0,d[j],d[j+1]))
      aver_db[j]=list(rep(0,d[j+1]))
    }
    for (s in 1:mb){
      nn_1=forward(nn,data_train[s,])
      nn_1=backward(nn_1,k_train[s])
      dW=nn_1$dW ;db=nn_1$db
      for ( j in 1:length(b)){ 
        aver_dW[j]=list(matrix(unlist(dW[j]),d[j])/mb+matrix(unlist(aver_dW[j]),d[j]))
        aver_db[j]=list(unlist(db[j])/mb+unlist(aver_db[j]))
      }
    }
    for (j in 1:length(b)){
      W[j]=list(matrix(unlist(W[j]),d[j])-eta*matrix(unlist(aver_dW[j]),d[j]))
      b[j]= list(unlist(b[j])-eta*unlist(aver_db[j]))
    }
    nn$W=W ;nn$b=b
  }  
  return(nn)
}
predict=function(nn,inp){
  prediction=c()
  for (i in 1:dim(inp)[1]){
    h=forward(nn,inp[i,])$h
    h=unlist(h[length(h)])
    p=exp(h)/sum(exp(h))
    prediction[i]= which.max(p)
  }
  return(prediction)
}

d=c(4,8,7,3)
levels(iris[,5])=1:3
label=as.numeric(iris[,5])
train_data=iris[-seq(5,150,5),1:4]
train_k=label[-seq(5,150,5)]
test_data=iris[seq(5,150,5),1:4]
test_k=label[seq(5,150,5)]
set.seed(6666)
nn=train(netup(d),train_data,train_k,nstep=10000)
print(nn)
prediction=predict(nn,test_data)
print(prediction)
accuracy=mean(prediction==test_k)
print(accuracy)