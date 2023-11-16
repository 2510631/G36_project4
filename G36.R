#G36

netup = function (d){
  h <- W <- b <- list()
  
  for( i in 1:(length(d) - 1)){
    
    h[i]=list(rep(0, d[i]))
    
    W[i]=list(matrix(runif(d[i] * d[i+1], 0, 0.2),d[i+1]))
    
    b[i]=list(runif(d[i+1], 0, 0.2))    
  }
  return(list(h = h, W = W, b = b, d = d))
}


forward= function(nn,inp){
  W=nn$W ;b=nn$b ;d=nn$d ;h=nn$h
  h[1]=list(inp)
  for(i in 1:length(b)){
    z=matrix(unlist(W[i]),d[i+1]) %*% unlist(h[i]) + unlist(b[i])
    z[z<0]=0
    h[i+1]=list(z)
  }
  return(nn=list(h=h,W=W,b=b,d=d)) 
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