x = seq(0,0.5,by=0.001)
p <- function(x){
  1-x/(1+x**2)
}
q <- function(x){
  (x-x**3)/(1+x+x**2+x**3)
}
q_p <- function(x){
  -(x-x**3)/(1+x+x**2+x**3)+x/(1+x**2)
}

matplot(x, cbind(p(x), q(x), q_p(x)),
        type = 'l', lty=1, lwd=4,
        col=c("red", "blue", "green"),
        ylab = "Probability", xlab = "a",
        main=substitute(paste("Probabilities in mixed NE plotted against the possible values of ", italic('a'))))
legend("left", legend = c("Action 1: p", "Action 2: q", "Action 3: 1-p-q"), 
       col = c("red", "blue", "green"), 
       lty = 1, lwd = 4)

x = seq(0,0.5,by=0.001)
p <- function(x){
  2*x/(3+6*x-4*x**2)
}
q <- function(x){
  (6*x+8*x**2-8*x**3)/(3+12*x+8*x**2-8*x**3)
}
q_p <- function(x){
  1-(2*x/(3+6*x-4*x**2)+(6*x+8*x**2-8*x**3)/(3+12*x+8*x**2-8*x**3))
}

matplot(x, cbind(p(x), q(x), q_p(x)),
        type = 'l', lty=1, lwd=4,
        col=c("red", "blue", "green"),
        ylab = "Probability", xlab = "a",
        main=substitute(paste("Probabilities in mixed regret minimization plotted against the possible values of ", italic('a'))))

legend("left", legend = c("Action 1: p", "Action 2: q", "Action 3: 1-p-q"), 
       col = c("red", "blue", "green"), 
       lty = 1, lwd = 4)

