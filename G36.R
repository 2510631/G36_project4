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