# Functions to analyze output of FitDeconfoundedHDAM()

# function to predict at a new X in R^p from the output of FitDeconfoundedHDAM()
estimate.function.1d <- function(x, lreturn){
  y <- lreturn$intercept
  x <- x - lreturn$Xmeans
  for(j in lreturn$active){
    breaksj <- lreturn$breaks[[j]]
    xj <- x[j]
    cj <- lreturn$coefs[[j]]
    # if xj lies outside of span of breaksj, predict the corresponding boundary value
    if(xj < min(breaksj)){
      y <- y + cj[1]
    } else if(xj > max(breaksj)){
      y <- y + cj[length(cj)]
    } else{
      Bj <- bsplineS(xj, breaks = breaksj)
      fhatj <- Bj %*% cj
      y <- y + fhatj
    }
  }
  return(y)
}

# function to predict at a matrix X in R^{n x p} of new values from the output of FitDeconfoundedHDAM()
estimate.function <- function(X, lreturn){
  ef1 <- function(x){
    return(estimate.function.1d(x, lreturn))
  }
  y <- apply(X, 1, ef1)
  return(y)
}

# functions to predict individual component functions f_j from the output of FitDeconfoundedHDAM()
estimate.fj.1d <- function(x, j, lreturn){
  x <- x - lreturn$Xmeans[j]
  breaks <- lreturn$breaks[[j]]
  coefs <- lreturn$coefs[[j]]
  if(x < min(breaks)){
    return(coefs[1])
  } else if(x > max(breaks)){
    return(coefs[length(coefs)])
  } else{
    B <- bsplineS(x, breaks = breaks)
    return(B %*% coefs)
  }
}

estimate.fj <- function(x, j, lreturn){
  ef1 <- function(y){
    return(estimate.fj.1d(y, j, lreturn))
  }
  return(sapply(x, ef1))
}
