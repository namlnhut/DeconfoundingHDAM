# Implementation of a ad-hoc comparative method:
# 1. Learn the number of factors and the factors from X
# 2. Input the learned factors as additional covariates that enter linearly in Y (in addition to the flexible components for X).


library(fda)
library(grplasso)


# Input:  Data matrix X in R^{n x p}
# Output: Estimate qhat of the number of hidden confounders
#         Using the eigenvalue ratio method

estimate.qhat <- function(X){
  qmax <- round(min(NROW(X), NCOL(X))/2)
  d <- svd(X, nu = 0, nv = 0)$d
  drat <- d[1:qmax]/d[2:(qmax+1)]
  qhat <- which(drat == max(drat))
  if(length(qhat) > 1){
    qhat <- qhat[1]
  }
  return(qhat)
}


# Input:  Data matrix X in R^{n x p}
#         Number of factors qhat, if not supplied, it is estimated 
#         using the function estimate.qhat()
# Output: Estimated factors \hat H
estimate.Hhat <- function(X, qhat = NULL){
  if(is.null(qhat)){
    qhat <- estimate.qhat(X)
  }
  U <- svd(X, nu = qhat, nv = 0)$u
  return(sqrt(NROW(U)) * U)
}



# Input: Y, B (whereas the last columns of B correspond to the factors H), index of grouping, vector of lambda values, number of folds k
# Output: Cross-validated MSEs and SEs as well as lambda.min and lambda.1se
cv.hdam.withFactors <- function(Y,  B, index, lambda, k = 5){
  n <- NROW(B)
  ind <-  sample(rep(1:k, length.out = n), replace = FALSE)
  fun.ind <- function(l){
    test <- which(ind == l)
    Ytrain <- Y[-test]
    Ytest <- Y[test]
    Btrain <- B[-test, ]
    Btest <- B[test, ]
    mod <- grplasso(Btrain, Ytrain, index = index, lambda = lambda, model = LinReg(), center = FALSE, standardize = FALSE)
    Ypred <- predict(mod, newdata = Btest)
    fmse <- function(y){
      return(mean((y-Ytest)^2))
    }
    MSEl <- apply(Ypred, 2, fmse)
    return(MSEl)
  }
  mses <- sapply(1:k, fun.ind)
  mse.agg <- apply(mses, 1, mean)
  se.agg <- 1/sqrt(k)*apply(mses, 1, sd)
  ind.min <- which(mse.agg == min(mse.agg))
  lambda.min <- lambda[ind.min]
  lambda.1se <- max(lambda[which(mse.agg <= mse.agg[ind.min]+se.agg[ind.min])])
  lresult <- list()
  lresult$mse <- mse.agg
  lresult$se <- se.agg
  lresult$lambda.min <- lambda.min
  lresult$lambda.1se <- lambda.1se
  return(lresult)
}


# Input: Response Y, factors H entering linearly, covariates X entering additively, given number basis.k of basis functions,
#       cross-validation method, cross validation k, number of lambdas to consider
# Output: Fitted HDAM at best lambda for the given number basis.k of basis functions
HDAM.withFactors <- function(Y, H, X, basis.k, cv.method = "1se", cv.k = 5, n.lambda = 20){
  K <- basis.k
  n <- NROW(X)
  p <- NCOL(X)
  q <- NCOL(H)
  # generate matrix of transformed basis functions
  B <- matrix(nrow = n, ncol = p*K)
  Rlist <- list()
  lbreaks <- list()
  for (j in 1:p){
    # number of breaks is number of basis functions minus order (4 by default) + 2
    breaks <- quantile(X[,j], probs=seq(0, 1, length.out = K-2))
    lbreaks[[j]] <- breaks
    Bj <- bsplineS(X[,j], breaks = breaks)
    Rj.inv <- solve(chol(1/n*t(Bj) %*% Bj))
    B[, ((j-1)*K + 1):(j*K)] <- Bj %*% Rj.inv
    Rlist[[j]] <- Rj.inv
  }
  # add column for intercept and H as the last columns 
  B <- cbind(rep(1, n), B, H)
  # variable grouping, intercept and components corresponding to H not penalized gets NA
  index <-c(NA, rep(1:p, each = K), rep(NA, q))

  # calculate maximal lambda
  lambdamax <- lambdamax(B, Y, index = index, model = LinReg(), center = FALSE, standardize = FALSE)
  # lambdas for cross validation
  lambda <- lambdamax / 1000^(0:(n.lambda-1) / (n.lambda-1))
  # cross validation for lambda
  res.cv <- cv.hdam.withFactors(Y, B, index, lambda, k = cv.k)
  
  if(cv.method == "1se"){
    lambdastar <- res.cv$lambda.1se
  } else if(cv.method == "min"){
    lambdastar <- res.cv$lambda.min
  } else {
    warning("CV method not implemented. Taking '1se'.")
  }
  
  # fit model on full data with lambdastar
  mod <- grplasso(B, Y, index = index, lambda = lambdastar, model = LinReg(), center = FALSE, standardize = FALSE)
  # transform back to original scale
  lcoef <- list()
  active <- numeric()
  for(j in 1:p){
    cj <- mod$coefficients[((j-1)*K + 2):(j*K+1), 1]
    # transform back
    lcoef[[j]] <- Rlist[[j]] %*% cj
    if(sum(cj^2) != 0){
      active <- c(active, j)
    }
  }
  intercept <- unname(mod$coefficients[1,1])
  lreturn <- list()
  # intercept
  lreturn$intercept <-intercept
  # list of breaks of B-spline basis
  lreturn$breaks <- lbreaks
  # list of coefficients
  lreturn$coefs <- lcoef
  # estimated active set
  lreturn$active <- active
  return(lreturn)
}

# Input: Response Y, factors H entering linearly, covariates X entering additively, number n.K of Ks to consider,
#       cross-validation method, cross validation k,
#        number of lambdas to consider for both rounds of cross validation
#        (n.lambda1 is used to find optimal K, n.lambda2 is used to find optimal lambda for this K)
FitHDAM.withFactors <- function(Y, H, X, n.K = 4, cv.method = "1se", cv.k = 5, n.lambda1 = 15, n.lambda2 = 30){
  n <- NROW(X)
  p <- NCOL(X)
  q <- NCOL(H)
  # create vector of candidate values for K
  # intuition: candidate values for K should be between K0 = 4 and 10*n^0.2
  if (n.K > 10){
    n.K <- 10
    warning("n.K set to 10")
  }
  K.up <- round(10*n^0.2)
  vK <- round(seq(4, K.up, length.out = n.K))
  # Generate the design and model parameters for every K in vK
  lmodK <- list()
  for (i in 1:length(vK)){
    K <- vK[i]
    B <- matrix(nrow = n, ncol = p*K)
    Rlist <- list()
    lbreaks <- list()
    for (j in 1:p){
      # number of breaks is number of basis functions minus order (4 by default) + 2
      breaks <- quantile(X[,j], probs=seq(0, 1, length.out = K-2))
      lbreaks[[j]] <- breaks
      Bj <- bsplineS(X[,j], breaks = breaks)
      Rj.inv <- solve(chol(1/n*t(Bj) %*% Bj))
      B[, ((j-1)*K + 1):(j*K)] <- Bj %*% Rj.inv
      Rlist[[j]] <- Rj.inv
    }
    # add column for intercept and H as the last columns 
    B <- cbind(rep(1, n), B, H)
    # variable grouping, intercept and components corresponding to H not penalized gets NA
    index <-c(NA, rep(1:p, each = K), rep(NA, q))
    # calculate maximal lambda
    lambdamax <- lambdamax(B, Y, index = index, model = LinReg(), center = FALSE, standardize = FALSE)
    # lambdas for cross validation
    lambda <- lambdamax / 1000^(0:(n.lambda1-1) / (n.lambda1-1))
    lmodK[[i]] <- list(Rlist = Rlist, lbreaks = lbreaks, index = index, B = B, lambda = lambda)
  }
  # generate folds for CV
  ind <-  sample(rep(1:cv.k, length.out = n), replace = FALSE)
  fun.ind <- function(l){
    test <- which(ind == l)
    Ytrain <- Y[-test]
    Ytest <- Y[test]
    fun.fixK <- function(listK){
      Btrain <- listK$B[-test, ]
      Btest <- listK$B[test, ]
      mod <- grplasso(Btrain, Ytrain, index = listK$index, lambda = listK$lambda, model = LinReg(), center = FALSE, standardize = FALSE)
      Ypred <- predict(mod, newdata = Btest)
      fmse <- function(y){
        return(mean((y-Ytest)^2))
      }
      MSEl.fixK<- apply(Ypred, 2, fmse)
      return(MSEl.fixK)
    }
    MSEl <- lapply(lmodK, fun.fixK)
    return(MSEl)
  }
  MSES <- lapply(1:cv.k, fun.ind)
  # aggregate MSES over folds
  MSES.agg <- matrix(NA, nrow = n.K, ncol = n.lambda1)
  for (i in 1:n.K){
    for (j in 1:n.lambda1){
      sij <- 0
      for (k in 1:cv.k){
        sij <- sij + MSES[[k]][[i]][j]
      }
      MSES.agg[i,j] <- sij / cv.k
    }
  }
  ind.min <- which(MSES.agg == min(MSES.agg), arr.ind = TRUE)
  K.min <- vK[ind.min[1]]
  lambda.min <- lmodK[[ind.min[1]]]$lambda[ind.min[2]]
  # refit model for this K and choose optimal lambda from larger list
  lreturn <- HDAM.withFactors(Y, H, X, basis.k = K.min, cv.method = cv.method, cv.k = cv.k, n.lambda = n.lambda2)
  lreturn$K.min <- K.min
  return(lreturn)
}

# Input: Response Y, covariates X, number n.K of Ks to consider,
#       cross-validation method, cross validation k,
#        number of lambdas to consider for both rounds of cross validation
#        (n.lambda1 is used to find optimal K, n.lambda2 is used to find optimal lambda for this K)
FitHDAM.withEstFactors <- function(Y, X, n.K = 4, cv.method = "1se", cv.k = 5, n.lambda1 = 15, n.lambda2 = 30){
  Xmeans <- colMeans(X)
  X <- scale(X, center = TRUE, scale = FALSE)
  qhat <- estimate.qhat(X)
  Hhat <- estimate.Hhat(X, qhat = qhat)
  lreturn <- FitHDAM.withFactors(Y = Y, H = Hhat, X = X, n.K = n.K, cv.method = cv.method, cv.k = cv.k, n.lambda1 = n.lambda1, n.lambda2 = n.lambda2)
  lreturn$Xmeans <- Xmeans
  return(lreturn)
}

