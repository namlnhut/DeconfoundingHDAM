## Code to reproduce Figures 3, 4, 13 and 14
# Vary the sample size n


# Import the functions needed
source("../r/FitDeconfoundedHDAM.R")
source("../r/AnalyzeFittedHDAM.R")
source("../r/EstimationFactors.R")


library("parallel")
detectCores()



f1 <- function(x){-sin(2*x)}
f2 <- function(x){2-2*tanh(x+0.5)}
f3 <- function(x){x}
f4 <- function(x){4/(exp(x)+exp(-x))}
f <- function(X){f1(X[,1])+f2(X[,2])+f3(X[,3])+f4(X[,4])}

# fix p = 300, q = 5
q <- 5
p <- 300


one.sim <- function(n, rho = NULL, seed.val, decreasing.confounding.influence = FALSE){
  # generate data
  set.seed(seed.val)
  Gamma <- matrix(runif(q*p, min = -1, max = 1), nrow=q)
  if(decreasing.confounding.influence){
    gam.weight <- 1/(1:q)
    Gamma <- gam.weight * Gamma
  }
  psi <- runif(q, min = 0, max = 2)
  H <- matrix(rnorm(n*q), nrow = n)
  # E should be Toeplitz if rho is specified
  if(is.null(rho)){
    E <- matrix(rnorm(n*p), nrow = n)
  } else {
    E <- matrix(rnorm(n*p), nrow = n)
    Toe <- toeplitz(rho^(0:(p-1)))
    # t(R.toe)%*%R.toe = Toe
    R.toe <- chol(Toe)
    # E has rows i.i.d with covariance Toe
    E <- E %*% R.toe
  }
  e <- 0.5*rnorm(n)
  X <- H %*% Gamma + E
  Y <- f(X) + H %*% psi + e
  lres.trim <- FitDeconfoundedHDAM(Y, X, n.K = 5, meth = "trim", cv.method = "1se", cv.k = 5, n.lambda1 = 10, n.lambda2 = 25)
  lres.none <- FitDeconfoundedHDAM(Y, X, n.K = 5, meth = "none", cv.method = "1se", cv.k = 5, n.lambda1 = 10, n.lambda2 = 25)
  lres.estFac <- FitHDAM.withEstFactors(Y, X, n.K = 5, cv.method = "1se", cv.k = 5, n.lambda1 = 10, n.lambda2 = 25)
  ntest <- 1000
  Htest <- matrix(rnorm(ntest * q), nrow = ntest)
  Etest <- matrix(rnorm(ntest * p), nrow = ntest)
  Xtest <- Htest %*% Gamma +Etest
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
n.vec <- c(50, 100, 200, 400, 800)


set.seed(1432)
smax <- 2100000000
seed.vec <- sample(1:smax, nrep)

for (i in 1:length(n.vec)){
  n <- n.vec[i]
  
  # equal confounding influence
  fun = function(seed.val){return(one.sim(n, rho = NULL, seed.val = seed.val, decreasing.confounding.influence = FALSE))}
  lres.rhoNULL <- mclapply(seed.vec, fun, mc.cores = use.cores)
  filename <- paste("SimulationResults/VarN_2024_08_13/equalCI_N_", n, "_rho00.RData", sep = "")
  save(lres.rhoNULL, file = filename)
  # rho = 0.4
  fun = function(seed.val){return(one.sim(n, rho = 0.4, seed.val = seed.val, decreasing.confounding.influence = FALSE))}
  lres.rho04 <- mclapply(seed.vec, fun, mc.cores = use.cores)
  filename <- paste("SimulationResults/VarN_2024_08_13/equalCI_N_", n, "_rho04.RData", sep = "")
  save(lres.rho04, file = filename)
  # rho = 0.8
  fun = function(seed.val){return(one.sim(n, rho = 0.8, seed.val = seed.val, decreasing.confounding.influence = FALSE))}
  lres.rho08 <- mclapply(seed.vec, fun, mc.cores = use.cores)
  filename <- paste("SimulationResults/VarN_2024_08_13/equalCI_N_", n, "_rho08.RData", sep = "")
  save(lres.rho08, file = filename)
  
  # decreasing confounding influence
  fun = function(seed.val){return(one.sim(n, rho = NULL, seed.val = seed.val, decreasing.confounding.influence = TRUE))}
  lres.rhoNULL <- mclapply(seed.vec, fun, mc.cores = use.cores)
  filename <- paste("SimulationResults/VarN_2024_08_13/decreasingCI_N_", n, "_rho00.RData", sep = "")
  save(lres.rhoNULL, file = filename)
  # rho = 0.4
  fun = function(seed.val){return(one.sim(n, rho = 0.4, seed.val = seed.val, decreasing.confounding.influence = TRUE))}
  lres.rho04 <- mclapply(seed.vec, fun, mc.cores = use.cores)
  filename <- paste("SimulationResults/VarN_2024_08_13/decreasingCI_N_", n, "_rho04.RData", sep = "")
  save(lres.rho04, file = filename)
  # rho = 0.8
  fun = function(seed.val){return(one.sim(n, rho = 0.8, seed.val = seed.val, decreasing.confounding.influence = TRUE))}
  lres.rho08 <- mclapply(seed.vec, fun, mc.cores = use.cores)
  filename <- paste("SimulationResults/VarN_2024_08_13/decreasingCI_N_", n, "_rho08.RData", sep = "")
  save(lres.rho08, file = filename)
  
  
}
te <- Sys.time()
time.needed <- te-ta
print(paste("Time needed: ", time.needed))



## Generate plots

setting.vec <- c("rhoNULL", "rho04", "rho08")
rho.vec <- c("00", "04", "08")
CI.vec <- c("equal", "decreasing")
df <- data.frame(matrix(NA, ncol = 6, nrow = nrep * length(n.vec) * length(setting.vec) * length(CI.vec) * 3))
colnames(df) <- c("rho", "n", "meth", "CI", "MSE", "s.active")

count <- 1
for(i in 1:length(n.vec)){
  n <- n.vec[i]
  for(j in 1:length(setting.vec)){
    setting <- setting.vec[j]
    rho <- rho.vec[j]
    for(k in 1:length(CI.vec)){
      CI <- CI.vec[k]
      filename <- paste("SimulationResults/VarN_2024_08_13/", CI, "CI_N_", n, "_rho", rho, ".RData", sep = "")
      load(filename)
      lres <- get(paste("lres.", setting, sep = ""))
      for(l in 1:nrep){
        df[count, ] <- data.frame(rho = rho, n = n, meth = "deconfounded", CI = CI, MSE = lres[[l]][1,1], s.active = lres[[l]][2,1])
        df[count+1, ] <- data.frame(rho = rho, n = n, meth = "naive", CI = CI, MSE = lres[[l]][1,2], s.active = lres[[l]][2,2])
        df[count + 2    , ] <- data.frame(rho = rho, n = n, meth = "estimated factors", CI = CI, MSE = lres[[l]][1,3], s.active = lres[[l]][2,3])
        count <- count + 3
      }
    }
  }
}


df$n <- factor(df$n, levels = n.vec)
df$meth <- factor(df$meth, levels = c("deconfounded", "estimated factors", "naive"))

library(gridExtra)
library(ggplot2)


titletext.mse <- c("MSE of f with p=300, q=5, s=4, components of E independent",
                   "MSE of f with p=300, q=5, s=4, E Toeplitz(0.4)",
                   "MSE of f with p=300, q=5, s=4, E Toeplitz(0.8)")
titletext.s <- c("Size of estimated active set of f with p=300, q=5, s=4, components of E independent",
                 "Size of estimated active set of f with p=300, q=5, s=4, E Toeplitz(0.4)",
                 "Size of estimated active set of f with p=300, q=5, s=4, E Toeplitz(0.8)")
subtitletext <- c("equal confounding influence", "decreasing confounding influence")

plot_results <- function(i, k){
  df.ik <- df[which(df$rho == rho.vec[i] & df$CI == CI.vec[k]), ]
  p <- ggplot(df.ik, aes(x = n, y = MSE, fill = meth)) +
    geom_violin(scale = "width", position = position_dodge(width = 0.7), width = 0.6)
  p <- p + stat_summary(fun.y=mean, geom="point",  position=position_dodge(0.7)) +
    xlab("n") + ylab("MSE") + ggtitle(titletext.mse[i], subtitle = subtitletext[k])
  p <- p + theme(axis.text=element_text(size=12),
                 axis.title=element_text(size=12), title=element_text(size=11.5),
                 legend.title=element_blank(), legend.text=element_text(size = 12))
  
  
  q <- ggplot(df.ik, aes(x=n, y=s.active, fill=meth)) +
    geom_violin(scale = "width", position = position_dodge(width = 0.7), width = 0.6)
  q <- q + stat_summary(fun.y=mean, geom="point", position=position_dodge(0.7)) +
    xlab("n") + ylab("Size of active set") + ggtitle(titletext.s[i], subtitle = subtitletext[k])
  q <- q + theme(axis.text=element_text(size=12),
                 axis.title=element_text(size=12), title=element_text(size=11.5),
                 legend.title=element_blank(), legend.text=element_text(size = 12))
  plot.pq <- arrangeGrob(p, q, nrow=2)
  return(plot.pq)
}

for(i in 1:length(setting.vec)){
  for(k in 1:length(CI.vec)){
    plot.ik <- plot_results(i,k)
    plotname <- paste("PlotResults/FinalPlots/VarN_Rho", rho.vec[i], "_", CI.vec[k], "CI.pdf", sep = "")
    ggsave(plotname, plot = plot.ik, width = 20.5, height = 13, units = "cm")
  }
}


