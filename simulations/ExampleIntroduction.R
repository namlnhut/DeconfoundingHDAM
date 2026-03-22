## Code to reproduce Figure 1
# Introductory example

# The R-Script VarP.R needs to be run before.

# consider the setting n = 300, p = 800 from the simulation script VarP.R
load("SimulationResults/VarP_2024_08_13/equalCI_P_800_rho00.RData")

nrep <- 100
p.vec <- c(50, 100, 200, 400, 800)

df <- data.frame(matrix(NA, ncol = 3, nrow = nrep * 3))
colnames(df) <- c("meth", "MSE", "s.active")
count <- 1
for(l in 1:nrep){
  df[count, ] <- data.frame(meth = "deconfounded", MSE = lres.rhoNULL[[l]][1,1], s.active = lres.rhoNULL[[l]][2,1])
  df[count+1, ] <- data.frame(meth = "naive", MSE = lres.rhoNULL[[l]][1,2], s.active = lres.rhoNULL[[l]][2,2])
  df[count + 2    , ] <- data.frame(meth = "estimated factors", MSE = lres.rhoNULL[[l]][1,3], s.active = lres.rhoNULL[[l]][2,3])
  count <- count + 3
}


pdf("PlotResults/FinalPlots/ExampleIntroduction.pdf", width = 7, height = 4)
par(mfrow = c(1,2))
hist(df[df$meth == "deconfounded", "MSE"], xlim = c(0, 12.5), breaks = seq(0, 12.5, 0.25),
     col = rgb(0,0, 1,0.2), xlab = "MSE", main = "Mean squared error of f")
hist(df[df$meth == "naive", "MSE"], add = TRUE, breaks = seq(0, 12.5, 0.25), col = rgb(1,0, 0,0.2))

legend("topright", legend=c("deconfounded","naive"), col=c(rgb(0,0,1,0.2), 
                                                      rgb(1,0,0,0.2)), pt.cex=2, pch=15 )

hist(df[df$meth == "deconfounded", "s.active"], xlim = c(0, 145), ylim = c(0, 22), breaks = seq(0, 145, 5),
     col = rgb(0,0, 1,0.2), xlab = "Size of estimated active set", main = "Size of estimated active set")
hist(df[df$meth == "naive", "s.active"], add = TRUE, breaks = seq(0, 145, 5), col = rgb(1,0, 0,0.2))
legend("topright", legend=c("deconfounded","naive"), col=c(rgb(0,0,1,0.2), 
                                                   rgb(1,0,0,0.2)), pt.cex=2, pch=15 )
abline(v=4)

dev.off()