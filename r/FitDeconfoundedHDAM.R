# Function FitDeconfoundedHDAM() to fit a deconfounded high-dimensional additive model.
# Helper functions: cv.hdam, calcTrim, DeconfoundedHDAM

library(fda)
library(grplasso)

# Input: Transformed QY, QB, index of grouping, vector of lambda values, number of folds k
# Output: Cross-validated MSEs and SEs as well as lambda.min and lambda.1se
cv.hdam <- function(QY, QB, index, lambda, k = 5){
  n <- NROW(QB)
  ind <-  sample(rep(1:k, length.out = n), replace = FALSE)
  fun.ind <- function(l){
    test <- which(ind == l)
    QYtrain <- QY[-test]
    QYtest <- QY[test]
    QBtrain <- QB[-test, ]
    QBtest <- QB[test, ]
    mod <- grplasso(QBtrain, QYtrain, index = index, lambda = lambda, model = LinReg(), center = FALSE, standardize = FALSE)
    QYpred <- predict(mod, newdata = QBtest)
    fmse <- function(y){
      return(mean((y-QYtest)^2))
    }
    MSEl <- apply(QYpred, 2, fmse)
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

# Input: Data matrix X in R^{n x p}
# Output: Trim transformation Q in R^{n x n}
calcTrim <- function(X){
  n <- NROW(X)
  svdX <-  svd(X)
  d <- svdX$d
  U <- svdX$u
  dtilde <- pmin(d, median(d))
  Q <- diag(n) - U %*% diag(1 - dtilde/d) %*% t(U)
  return(Q)
}

# Input: Response Y, predictors X, given number basis.k of basis functions,
#        transformation method, cross-validation method, cross validation k, number of lambdas to consider
# Output: Fitted HDAM at best lambda for the given number basis.k of basis functions
DeconfoundedHDAM <- function(Y, X, basis.k, meth = "trim", cv.method = "1se", cv.k = 5, n.lambda = 20){
  K <- basis.k
  n <- NROW(X)
  p <- NCOL(X)
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
  # add column for intercept
  B <- cbind(rep(1, n), B)
  # variable grouping, intercept not penalized gets NA
  index <-c(NA, rep(1:p, each = K))
  # spectral transformation
  if(meth == "trim"){
    Q <- calcTrim(X)
    QY <- Q%*%Y
    QB <- Q%*%B
  } else if(meth == "none"){
    QY <- Y
    QB <- B
  } else {
    stop("You must choose meth = 'trim' or 'none'.")
  }
  # calculate maximal lambda
  lambdamax <- lambdamax(QB, QY, index = index, model = LinReg(), center = FALSE, standardize = FALSE)
  # lambdas for cross validation
  lambda <- lambdamax / 1000^(0:(n.lambda-1) / (n.lambda-1))
  # cross validation for lambda
  res.cv <- cv.hdam(QY, QB, index, lambda, k = cv.k)
  
  if(cv.method == "1se"){
    lambdastar <- res.cv$lambda.1se
  } else if(cv.method == "min"){
    lambdastar <- res.cv$lambda.min
  } else {
    warning("CV method not implemented. Taking '1se'.")
  }
  
  # fit model on full data with lambdastar
  mod <- grplasso(QB, QY, index = index, lambda = lambdastar, model = LinReg(), center = FALSE, standardize = FALSE)
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

# Input: Response Y, predictors X, number n.K of Ks to consider,
#        transformation method, cross-validation method, cross validation k,
#        number of lambdas to consider for both rounds of cross validation
#        (n.lambda1 is used to find optimal K, n.lambda2 is used to find optimal lambda for this K)
FitDeconfoundedHDAM <- function(Y, X, n.K = 4, meth = "trim", cv.method = "1se", cv.k = 5, n.lambda1 = 15, n.lambda2 = 30){
  n <- NROW(X)
  p <- NCOL(X)
  # Center X_j's
  Xmeans <- colMeans(X)
  X <- scale(X, center = TRUE, scale = FALSE)
  # create vector of candidate values for K
  # intuition: candidate values for K should be between K0 = 4 and 10*n^0.2
  if (n.K > 10){
    n.K <- 10
    warning("n.K set to 10")
  }
  K.up <- round(10*n^0.2)
  vK <- round(seq(4, K.up, length.out = n.K))
  # spectral transformation
  if(meth == "trim"){
    Q <- calcTrim(X)
    QY <- Q%*%Y
  } else if(meth == "none"){
    QY <- Y
    Q <- diag(n)
  } else {
    stop("You must choose meth = 'trim' or 'none'.")
  }
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
    # add column for intercept
    B <- cbind(rep(1, n), B)
    QB <- Q %*% B
    # variable grouping, intercept not penalized gets NA
    index <-c(NA, rep(1:p, each = K))
    # calculate maximal lambda
    lambdamax <- lambdamax(QB, QY, index = index, model = LinReg(), center = FALSE, standardize = FALSE)
    # lambdas for cross validation
    lambda <- lambdamax / 1000^(0:(n.lambda1-1) / (n.lambda1-1))
    lmodK[[i]] <- list(Rlist = Rlist, lbreaks = lbreaks, index = index, B = B, QB = QB, lambda = lambda)
  }
  # generate folds for CV
  ind <-  sample(rep(1:cv.k, length.out = n), replace = FALSE)
  fun.ind <- function(l){
    test <- which(ind == l)
    QYtrain <- QY[-test]
    QYtest <- QY[test]
    fun.fixK <- function(listK){
      QBtrain <- listK$QB[-test, ]
      QBtest <- listK$QB[test, ]
      mod <- grplasso(QBtrain, QYtrain, index = listK$index, lambda = listK$lambda, model = LinReg(), center = FALSE, standardize = FALSE)
      QYpred <- predict(mod, newdata = QBtest)
      fmse <- function(y){
        return(mean((y-QYtest)^2))
      }
      MSEl.fixK<- apply(QYpred, 2, fmse)
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
  lreturn <- DeconfoundedHDAM(Y, X, basis.k = K.min, meth = meth, cv.method = cv.method, cv.k = cv.k, n.lambda = n.lambda2)
  lreturn$K.min <- K.min
  lreturn$Xmeans <- Xmeans
  return(lreturn)
}


