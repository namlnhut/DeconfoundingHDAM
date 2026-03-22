## Code to reproduce Figures 17 and 18
# Vary the proportion of confounded covariates

# Import the functions needed
source("../r/FitDeconfoundedHDAM.R")
source("../r/AnalyzeFittedHDAM.R")
source("../r/EstimationFactors.R")


library("parallel")

f1 <- function(x){-sin(2*x)}
f2 <- function(x){2-2*tanh(x+0.5)}
f3 <- function(x){x}
f4 <- function(x){4/(exp(x)+exp(-x))}
f <- function(X){f1(X[,1])+f2(X[,2])+f3(X[,3])+f4(X[,4])}

# fix n = 400, p = 500, q = 5
q <- 5
n <-  400
p <- 500


one.sim <- function(prop, seed.val, decreasing.confounding.influence = FALSE){
  set.seed(seed.val)
  # generate data
  Gamma <- matrix(runif(q*p, min = -1, max = 1), nrow=q)
  if(decreasing.confounding.influence){
    gam.weight <- 1/(1:q)
    Gamma <- gam.weight * Gamma
  }
  # only prop fraction of entries of Gamma should be nonzero
  Gamma0 <- matrix(rbinom(q*p, 1, prop), nrow=q)
  Gamma <- Gamma*Gamma0
  psi <- runif(q, min = 0, max = 2)
  H <- matrix(rnorm(n*q), nrow = n)
  E <- matrix(rnorm(n*p), nrow = n)
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
prop.vec <- seq(0, 1, 0.05)



set.seed(1432)
smax <- 2100000000
seed.vec <- sample(1:smax, nrep)

for (i in 1:length(prop.vec)){
  prop <- prop.vec[i]
  # with equal confounding influence
  fun = function(seed.val){return(one.sim(prop, seed.val = seed.val, decreasing.confounding.influence = FALSE))}
  lres <- mclapply(seed.vec, fun, mc.cores = use.cores)
  filename <- paste("SimulationResults/VarCP_2024_08_13/equalCI_CP_", i, ".RData", sep = "")
  save(lres, file = filename)
  # with decreasing confounding influence
  fun = function(seed.val){return(one.sim(prop, seed.val = seed.val, decreasing.confounding.influence = TRUE))}
  lres <- mclapply(seed.vec, fun, mc.cores = use.cores)
  filename <- paste("SimulationResults/VarCP_2024_08_13/decreasingCI_CP_", i, ".RData", sep = "")
  save(lres, file = filename)
}
te <- Sys.time()
time.needed <- te-ta
print(paste("Time needed: ", time.needed))


# Generate plots

CI.vec <- c("equal", "decreasing")
df <- data.frame(matrix(NA, ncol = 5, nrow = nrep * length(prop.vec) * length(CI.vec) * 3))
colnames(df) <- c("prop", "meth", "CI", "MSE", "s.active")


count <- 1
for(i in 1:length(prop.vec)){
  prop <- prop.vec[i]
  for(k in 1:length(CI.vec)){
    CI <- CI.vec[k]
    filename <- paste("SimulationResults/VarCP_2024_08_13/", CI, "CI_CP_", i, ".RData", sep = "")
    load(filename)
    for(l in 1:nrep){
      df[count, ] <- data.frame(prop = prop, meth = "deconfounded", CI = CI, MSE = lres[[l]][1,1], s.active = lres[[l]][2,1])
      df[count+1, ] <- data.frame(prop = prop, meth = "naive", CI = CI, MSE = lres[[l]][1,2], s.active = lres[[l]][2,2])
      df[count + 2    , ] <- data.frame(prop = prop, meth = "estimated factors", CI = CI, MSE = lres[[l]][1,3], s.active = lres[[l]][2,3])
      count <- count + 3
    }
  }
}

df$prop <- factor(df$prop, levels = prop.vec)
df$meth <- factor(df$meth, levels = c("deconfounded", "estimated factors", "naive"))

library(gridExtra)
library(ggplot2)


titletext.mse <- "MSE of f with n=400, p=500, q=5, s=4"
titletext.s <- "Size of estimated active set of f with n=400, p=500, q=5, s=4"
subtitletext <- c("equal confounding influence", "decreasing confounding influence")

plot_results <- function(k){
  df.k <- df[which(df$CI == CI.vec[k]), ]
  p <- ggplot(df.k, aes(x = prop, y = MSE, fill = meth)) +
    geom_violin(scale = "width", position = position_dodge(width = 0.7), width = 0.6)
  p <- p + stat_summary(fun.y=mean, geom="point", size = 0.4, position=position_dodge(0.7)) +
    xlab("prop") + ylab("MSE") + ggtitle(titletext.mse, subtitle = subtitletext[k])
  p <- p + theme(axis.text=element_text(size=12),
                 axis.title=element_text(size=12), title=element_text(size=11.5),
                 legend.title=element_blank(), legend.text=element_text(size = 12),
                 axis.text.x = element_text(size = 9.5))
  
  
  q <- ggplot(df.k, aes(x=prop, y=s.active, fill=meth)) +
    geom_violin(scale = "width", position = position_dodge(width = 0.7), width = 0.6)
  q <- q + stat_summary(fun.y=mean, geom="point", size = 0.4, position=position_dodge(0.7)) +
    xlab("prop") + ylab("Size of active set") + ggtitle(titletext.s, subtitle = subtitletext[k])
  q <- q + theme(axis.text=element_text(size=12),
                 axis.title=element_text(size=12), title=element_text(size=11.5),
                 legend.title=element_blank(), legend.text=element_text(size = 12),
                 axis.text.x = element_text(size = 9.5))
  plot.pq <- arrangeGrob(p, q, nrow=2)
  return(plot.pq)
}


for(k in 1:length(CI.vec)){
  plot.k <- plot_results(k)
  plotname <- paste("PlotResults/FinalPlots/VarCP_", CI.vec[k], "CI.pdf", sep = "")
  ggsave(plotname, plot = plot.k, width = 20.5, height = 13, units = "cm")
}


