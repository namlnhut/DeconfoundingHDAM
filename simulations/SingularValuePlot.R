# Generate Figure 2 

# Plots of the singular values of X for the two simulation scenarios

q <- 5
n <- 100
p <- 300

set.seed(1219)

Gamma.equal <- matrix(runif(q*p, min = -1, max = 1), nrow=q)
gam.weight <- 1/(1:q)
Gamma.decreasing <- gam.weight * Gamma.equal

psi <- runif(q, min = 0, max = 2)
H <- matrix(rnorm(n*q), nrow = n)
E <- matrix(rnorm(n*p), nrow = n)
X.equal <- H %*% Gamma.equal + E
X.decreasing <- H %*% Gamma.decreasing + E

sv.equal <-svd(X.equal)$d
sv.decreasing <- svd(X.decreasing)$d

par(mfrow = c(1,2))
plot(sv.equal, xlab = "l", ylab = "singular value d_l", main = "equal confounding influence")
plot(sv.decreasing, xlab = "l", ylab = "singular value d_l", main = "decreasing confounding influence")
