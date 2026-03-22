## Code to reproduce Figures 9, 10, 11, 12
# Analysis of the motif regression data set

# Import the functions needed
source("../r/FitDeconfoundedHDAM.R")
source("../r/AnalyzeFittedHDAM.R")
source("../r/EstimationFactors.R")

# The original motif data set from "M. A. Beer and S. Tavazoie (2004). 
# Predicting Gene Expression from Sequence, Cell, Volume 117, Issue 2"
# can be obtained at 
# https://www.sciencedirect.com/science/article/pii/S0092867404003046?via%3Dihub#aep-section-id22
# We analyzed an already pre-processed version of the data set from
# "Z. Guo, W. Yuan and C. Zhang (2019). Local Inference in Additive Models with 
# Decorrelated Local Linear Estimator. arXiv preprint arXiv:1907.12732."

load("MotifData/MotifData.RData")

# Data as in DLL-Real-Data.R,(Guo, Z., Yuan W. and Zhang, C. (2019).
# Local Inference in Additive Models with Decorrelated Local Linear Estimator.)
# We take the same response as there, i.e. y = genes[, 131].

t = 131
y = genes[,t]
X = motifs
par(mfrow=c(1,1))
svdX.d <- svd(X)$d
plot(svdX.d)
# Very large first eigenvalue. But Data not centered.
Xmeans <- colMeans(X)
Xc <- scale(X, center = TRUE, scale = FALSE)
svdX.c <- svd(Xc)$d
plot(svdX.c, main = "Singular values of (centered) motif data", ylab="singular value d_l", xlab = "l")
# still confounding might be present

set.seed(1443)
# Note that the functions already scale X internally, so we do not need to work with Xc here
ta <- Sys.time()
fit.null <- FitDeconfoundedHDAM(y, X, n.K = 5, meth = "none", cv.method = "1se", cv.k = 5, n.lambda1 = 15, n.lambda2 = 50)
te <- Sys.time()
print("fit.null done, time needed:")
print(te-ta)

ta <- Sys.time()
fit.trim <- FitDeconfoundedHDAM(y, X, n.K = 5, meth = "trim", cv.method = "1se", cv.k = 5, n.lambda1 = 15, n.lambda2 = 50)
te <- Sys.time()
print("fit.trim done, time needed:")
print(te-ta)

ta <- Sys.time()
fit.estFac <- FitHDAM.withEstFactors(y, X, n.K = 5, cv.method = "1se", cv.k = 5, n.lambda1 = 15, n.lambda2 = 50)
te <- Sys.time()
print("fit.estFac done, time needed:")
print(te-ta)

save(fit.null, fit.trim, fit.estFac, file = "MotifEvaluation_2024_07_29.RData")

load("MotifEvaluation_2024_07_29.RData")

length(fit.null$active)
# fit.null has 211 active variables

length(fit.trim$active)
# fit.trim has 95 active variables

length(fit.estFac$active)
# fit.estFac has 167 active variables

length(intersect(fit.null$active, fit.trim$active))
# 92 variables are in the active set of both fit.null and fit.trim

length(intersect(fit.estFac$active, fit.trim$active))
# 85 variables are in the active set of both fit.estFac and fit.trim

length(intersect(fit.null$active, fit.estFac$active))
# 167 variables are in the active set of both fit.null and fit.estFac

lengthsq <- function(v){sum(v^2)}
coeflength.null <- as.numeric(lapply(fit.null$coefs, lengthsq))
coeflength.trim <- as.numeric(lapply(fit.trim$coefs, lengthsq))
coeflength.estFac <- as.numeric(lapply(fit.estFac$coefs, lengthsq))

# plot 9 strongest functions in terms of coeflength.trim

ord <- order(coeflength.trim)
ind <- tail(ord, n=9)

par(mfrow = c(3,3), mai = c(0.5, 0.5, 0.2, 0.2), mgp = c(2, 0.5, 0))
for (l in 1:9){
  j <- ind[10-l]
  xx <- seq(min(X[,j]), max(X[,j]), length.out=50)
  fhatj.trim <- estimate.fj(xx, j, fit.trim)
  plot(xx, fhatj.trim, type = "l", col = "red", xlim=c(min(xx),max(xx)), xlab = "Xj", ylab = "fj(Xj)", main = paste("j =", j), ylim = c(-0.5, 2.5))
  fhatj.null <- estimate.fj(xx, j, fit.null)
  lines(xx, fhatj.null, type = "l", col = "blue", ylim=c(min(xx),max(xx)), xlab = "Xj", ylab = "fj(Xj)", main = paste("j =", j))
  fhatj.estFac <- estimate.fj(xx, j, fit.estFac)
  lines(xx, fhatj.estFac, type = "l", col = "darkgreen", ylim=c(min(xx),max(xx)), xlab = "Xj", ylab = "fj(Xj)", main = paste("j =", j))
  legend("topleft", legend = c("deconfounded", "estimated factors", "naive"), col = c("red", "darkgreen", "blue"), lwd = 1)

  abline(h=0, col="grey")
  points(X[,j], rep(-0.5, length(X[, j])), pch = 16, cex= 1, col=rgb(0,0,0, 0.1))
}

# Plot the strongest functions from the naive method that are set to zero by using the deconfounded method
ord <- order(coeflength.null)
# active set of "none" minus active set of "trim"
active.diff <- setdiff(fit.null$active, fit.trim$active)

foo <- function(j){return(j %in% active.diff)}

in.diff <- sapply(ord, foo)

int.ind <- tail(ord[in.diff], 9)

par(mfrow = c(3,3), mai = c(0.5, 0.5, 0.2, 0.2), mgp = c(2, 0.5, 0))
for (l in 1:9){
  j <- int.ind[10-l]
  xx <- seq(min(X[,j]), max(X[,j]), length.out=50)
  fhatj.trim <- estimate.fj(xx, j, fit.trim)
  plot(xx, fhatj.trim, type = "l", col = "red", xlim=c(min(xx),max(xx)), xlab = "Xj", ylab = "fj(Xj)", main = paste("j =", j), ylim = c(-0.5, 2.5))
  fhatj.null <- estimate.fj(xx, j, fit.null)
  lines(xx, fhatj.null, type = "l", col = "blue", ylim=c(min(xx),max(xx)), xlab = "Xj", ylab = "fj(Xj)", main = paste("j =", j))
  fhatj.estFac <- estimate.fj(xx, j, fit.estFac)
  lines(xx, fhatj.estFac, type = "l", col = "darkgreen", ylim=c(min(xx),max(xx)), xlab = "Xj", ylab = "fj(Xj)", main = paste("j =", j))
  legend("topleft", legend = c("deconfounded", "estimated factors", "naive"), col = c("red", "darkgreen", "blue"), lwd = 1)

  abline(h=0, col="grey")
  points(X[,j], rep(-0.5, length(X[, j])), pch = 16, cex= 1, col=rgb(0,0,0, 0.1))
}


# coefficient length of deconfounded vs naive
par(mfrow=c(1,1))
plot(sqrt(coeflength.trim), sqrt(coeflength.null), col = "blue", pch = 1, main = "Norm of coefficient vectors", xlab = "deconfounded", ylab = "naive / estimated factors")
points(sqrt(coeflength.trim), sqrt(coeflength.estFac), col = "darkgreen", pch = 4)
legend("topleft", legend = c("deconfounded vs. naive", "deconfounded vs. estimated factors"), col = c("blue", "darkgreen"), pch = c(1,4))
abline(0,1)

# for l = 1,..., 211, record the size of the intersection of the strongest l variables for trim and the strongest l variables for none
ord.null <- order(coeflength.null, decreasing = TRUE)[1:211]
ord.trim <- order(coeflength.trim, decreasing = TRUE)[1:92]
ord.estFac <- order(coeflength.estFac, decreasing = TRUE)[1:167]

size.intersect.null <- numeric(211)
size.union.null <- numeric(211)
for(l in 1:211){
  size.intersect.null[l] <- length(intersect(head(ord.null, l), head(ord.trim, l)))
  size.union.null[l] <- length(union(head(ord.null, l), head(ord.trim, l)))
}
jaccard.sim.null <- size.intersect.null/size.union.null

size.intersect.estFac <- numeric(167)
size.union.estFac <- numeric(167)
for(l in 1:167){
  size.intersect.estFac[l] <- length(intersect(head(ord.estFac, l), head(ord.trim, l)))
  size.union.estFac[l] <- length(union(head(ord.estFac, l), head(ord.trim, l)))
}
jaccard.sim.estFac <- size.intersect.estFac/size.union.estFac


plot(jaccard.sim.null, type = "l", col = "blue")
lines(jaccard.sim.estFac, col = "darkgreen")
legend("topright", legend = c("deconfounded vs. naive", "deconfounded vs. estimated factors"), col = c("blue", "darkgreen"), lty = 1)

# both plots in one
pdf("PlotResults/FinalPlots/MotifCoefLength.pdf", width = 10, height = 5)
par(mfrow = c(1,2))
plot(sqrt(coeflength.trim), sqrt(coeflength.null), col = "blue", pch = 1, main = "Norm of coefficient vectors", xlab = "deconfounded", ylab = "naive / estimated factors")
points(sqrt(coeflength.trim), sqrt(coeflength.estFac), col = "darkgreen", pch = 4)
legend("topleft", legend = c("deconfounded vs. naive", "deconfounded vs. estimated factors"), col = c("blue", "darkgreen"), pch = c(1,4))
abline(0,1)
plot(jaccard.sim.null, type = "l", col = "blue", main = "Jaccard similarity of top l index sets", xlab = "l", ylab = "Jaccard similarity")
lines(jaccard.sim.estFac, col = "darkgreen")
legend("topright", legend = c("deconfounded vs. naive", "deconfounded vs. estimated factors"), col = c("blue", "darkgreen"), lty = 1)
dev.off()
