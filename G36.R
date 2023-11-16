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