## Code to reproduce Figures 19 and 20
# Nonlinear confounding effects

# Import the functions needed
source("../r/FitDeconfoundedHDAM.R")
source("../r/AnalyzeFittedHDAM.R")
source("../r/EstimationFactors.R")


library("parallel")
library(ggplot2)

f1 <- function(x){-sin(2*x)}
f2 <- function(x){2-2*tanh(x+0.5)}
f3 <- function(x){x}
f4 <- function(x){4/(exp(x)+exp(-x))}
f <- function(X){f1(X[,1])+f2(X[,2])+f3(X[,3])+f4(X[,4])}

# function to induce nonlinearity in H->X
# linear interpolation between t and abs(t)
nlX <- function(al, t){(1-al)*t+al*abs(t)}

# function to induce nonlinearity in H->Y
# linear interpolation between t and abs(t)
nlY <- function(bet, t){(1-bet)*t+bet*abs(t)}

# fix n = 400, p = 500
q <- 5
n <-  400
p <- 500


one.sim <- function(al, bet, seed.val, decreasing.confounding.influence = FALSE){
  set.seed(seed.val)
  # generate data
  Gamma <- matrix(runif(q*p, min = -1, max = 1), nrow=q)
  if(decreasing.confounding.influence){
    gam.weight <- 1/(1:q)
    Gamma <- gam.weight * Gamma
  }
  psi <- runif(q, min = 0, max = 2)
  H <- matrix(rnorm(n*q), nrow = n)
  E <- matrix(rnorm(n*p), nrow = n)
  e <- 0.5*rnorm(n)
  X <- nlX(al, H %*% Gamma) + E
  Y <- f(X) + nlY(bet, H %*% psi) + e
  lres.trim <- FitDeconfoundedHDAM(Y, X, n.K = 5, meth = "trim", cv.method = "1se", cv.k = 5, n.lambda1 = 10, n.lambda2 = 25)
  lres.none <- FitDeconfoundedHDAM(Y, X, n.K = 5, meth = "none", cv.method = "1se", cv.k = 5, n.lambda1 = 10, n.lambda2 = 25)
  lres.estFac <- FitHDAM.withEstFactors(Y, X, n.K = 5, cv.method = "1se", cv.k = 5, n.lambda1 = 10, n.lambda2 = 25)
  ntest <- 1000
  Htest <- matrix(rnorm(ntest * q), nrow = ntest)
  Etest <- matrix(rnorm(ntest * p), nrow = ntest)
  Xtest <- nlX(al, Htest %*% Gamma) + Etest
  fhat.trim <- estimate.function(Xtest, lres.trim)
  fhat.none <- estimate.function(Xtest, lres.none)
  fhat.estFac <- estimate.function(Xtest, lres.estFac)
  fXtest <- f(Xtest)
  mat.return <- rbind(c(mean((fXtest-fhat.trim)^2), mean((fXtest-fhat.none)^2), mean((fXtest-fhat.estFac)^2)),
                      c(length(lres.trim$active), length(lres.none$active), length(lres.estFac$active)))
  rownames(mat.return) <- c("MSE", "Active")
  colnames(mat.return) <- c("trim", "none", "estFac")
  return(mat.return)
}


nrep <- 100
use.cores <- 50



ta <- Sys.time()
al.vec <- seq(0, 1, length.out=7)
bet.vec <- seq(0, 1, length.out=7)


set.seed(1432)
smax <- 2100000000
seed.vec <- sample(1:smax, nrep)

for (i in 1:length(al.vec)){
  al <- al.vec[i]
  for (j in 1:length(bet.vec)){
    bet <- bet.vec[j]
    # with equal confounding influence
    fun = function(seed.val){return(one.sim(al = al, bet = bet, seed.val = seed.val, decreasing.confounding.influence = FALSE))}
    lres <- mclapply(seed.vec, fun, mc.cores = use.cores)
    filename <- paste("SimulationResults/NL_2024_08_13/equalCI_al_", i,"_bet_", j, ".RData", sep = "")
    save(lres, file = filename)
    # with decreasing confounding influence
    fun = function(seed.val){return(one.sim(al = al, bet = bet, seed.val = seed.val, decreasing.confounding.influence = TRUE))}
    lres <- mclapply(seed.vec, fun, mc.cores = use.cores)
    filename <- paste("SimulationResults/NL_2024_08_13/decreasingCI_al_", i,"_bet_", j, ".RData", sep = "")
    save(lres, file = filename)
  }
}
te <- Sys.time()
time.needed <- te-ta
print(paste("Time needed: ", time.needed))



calc.average <- function(quantity, CI, meth, i, j){
  filename <- paste("SimulationResults/NL_2024_08_13/", CI, "CI_al_", i,"_bet_", j, ".RData", sep = "")
  load(filename)
  return(mean(unlist(lapply(lres, function(mat){mat[quantity, meth]}))))
}

quantity.vec <- c("MSE", "Active")
CI.vec <- c("equal", "decreasing")
meth.vec <- c("trim", "none", "estFac")
for(quantity in quantity.vec){
  for(CI in CI.vec){
    for(meth in meth.vec){
      varname <- paste("mat_", quantity, "_", CI, "CI_", meth, sep = "")
      assign(varname, outer(1:7, 1:7, FUN = Vectorize(function(i,j){calc.average(quantity, CI, meth, i, j)})))
    }
  }
}


meth.name.vec <- c("deconfounded", "naive", "estimated factors")
library(viridisLite)
plot_ratios <- function(i, j, CI){
  MSE_meth1 <- get(paste("mat_MSE_", CI, "CI_", meth.vec[i], sep = ""))
  MSE_meth2 <- get(paste("mat_MSE_", CI, "CI_", meth.vec[j], sep = ""))
  mat_ratio <- MSE_meth1/MSE_meth2
  main_text <- paste( meth.name.vec[i], " vs. ", meth.name.vec[j], sep = "")
  image(al.vec, bet.vec, mat_ratio, axes = FALSE, col = magma(100, direction = -1), xlab = expression(alpha), ylab = expression(beta),
       main = main_text)
  axis(1, al.vec, round(al.vec, digits = 2))
  axis(2, bet.vec, round(bet.vec, digits = 2))
  for (i in 1:length(al.vec)){
    for (j in 1:length(bet.vec)){
      if(mat_ratio[i,j] < 0.995){
        text(al.vec[i], bet.vec[j], round(mat_ratio[i,j], 2), cex = 1.3)
      }
      else{
        text(al.vec[i], bet.vec[j], round(mat_ratio[i,j], 2), cex = 1.3, col = "white")
      }
    }
  }
}


par(mfrow = c(1,2), oma = c(0,0, 1, 0))
plot_ratios(1, 2, "equal")
plot_ratios(1, 3, "equal")

mtext("Ratio of average MSE \u2013 equal confounding influence", outer = TRUE, cex = 1.5, line = -1)

par(mfrow = c(1,2), oma = c(0,0, 1, 0))
plot_ratios(1, 2, "decreasing")
plot_ratios(1, 3, "decreasing")

mtext("Ratio of average MSE \u2013 decreasing confounding influence", outer = TRUE, cex = 1.5, line = -1)


