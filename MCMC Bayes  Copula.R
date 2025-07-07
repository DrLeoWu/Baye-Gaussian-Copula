
library(MASS)
library(Matrix)
library(mvtnorm)
library(truncnorm)

set.seed(1)

# Simulate data ---------------------------------------------------
n <- 200
d <- 4
k <- 5

alpha <- rep(1, k)  # shared Dirichlet prior

rdirichlet <- function(n, alpha) {
  x <- matrix(rgamma(n * length(alpha), shape = alpha), ncol = length(alpha), byrow = TRUE)
  x / rowSums(x)
}

p_true <- rdirichlet(1, alpha)
cutpoints <- qnorm(cumsum(p_true)[1:(k - 1)])
true_R <- matrix(0.5, d, d); diag(true_R) <- 1
Z <- mvrnorm(n, rep(0, d), true_R)

Y <- matrix(NA, n, d)
for (j in 1:d) {
  bounds <- c(-Inf, cutpoints, Inf)
  for (i in 1:n) {
    Y[i, j] <- which.max(Z[i, j] <= bounds) - 1
  }
}

# Helper functions ------------------------------------------------
compute_log_likelihood <- function(Z, Y, p, R) {
  loglik <- 0
  cuts <- qnorm(cumsum(p)[1:(k - 1)])
  bounds <- c(-Inf, cuts, Inf)
  for (j in 1:d) {
    for (i in 1:n) {
      lb <- bounds[Y[i, j]]
      ub <- bounds[Y[i, j] + 1]
      loglik <- loglik + log(pnorm(ub) - pnorm(lb) + 1e-10)
    }
  }
  loglik + sum(dmvnorm(Z, sigma = R, log = TRUE))
}

lkj_log_prior <- function(R, eta = 2) {
  ldet <- determinant(R, logarithm = TRUE)$modulus
  (eta - 1) * ldet
}

# MCMC Setup ------------------------------------------------------
n_iter <- 2000
burn_in <- 500
thin <- 5
keep <- (n_iter - burn_in) / thin

Z_latent <- matrix(0, n, d)
R_current <- diag(d)
p_current <- rdirichlet(1, alpha)

samples_R <- array(NA, c(keep, d, d))
samples_p <- matrix(NA, keep, k)

for (iter in 1:n_iter) {
  # 1. Gibbs: sample Z given Y, p, R
  cuts <- qnorm(cumsum(p_current)[1:(k - 1)])
  for (i in 1:n) {
    for (j in 1:d) {
      idx <- setdiff(1:d, j)
      R11 <- R_current[j, j]
      R12 <- R_current[j, idx]
      R22 <- R_current[idx, idx]
      R22_inv <- solve(R22)
      mu_cond <- R12 %*% R22_inv %*% Z_latent[i, idx]
      var_cond <- R11 - R12 %*% R22_inv %*% R12
      
      bounds <- c(-Inf, cuts, Inf)
      lb <- bounds[Y[i, j]]
      ub <- bounds[Y[i, j] + 1]
      Z_latent[i, j] <- rtruncnorm(1, a = lb, b = ub, mean = mu_cond, sd = sqrt(var_cond))
    }
  }
  
  # 2. Gibbs: update shared p from Dirichlet posterior
  counts <- rep(0, k)
  for (j in 1:d) {
    counts <- counts + table(factor(Y[, j], levels = 1:k))
  }
  p_current <- as.vector(rdirichlet(1, alpha + counts))
  
  # 3. MH: update R via random walk
  eps <- 0.05
  proposal <- R_current + eps * matrix(rnorm(d^2), d, d)
  proposal <- (proposal + t(proposal)) / 2
  diag(proposal) <- 1
  proposal <- as.matrix(nearPD(proposal)$mat)
  
  loglik_current <- compute_log_likelihood(Z_latent, Y, p_current, R_current)
  loglik_proposal <- compute_log_likelihood(Z_latent, Y, p_current, proposal)
  logprior_current <- lkj_log_prior(R_current)
  logprior_proposal <- lkj_log_prior(proposal)
  
  log_accept_ratio <- (loglik_proposal + logprior_proposal) - (loglik_current + logprior_current)
  if (log(runif(1)) < log_accept_ratio) {
    R_current <- proposal
  }
  
  # Store samples
  if (iter > burn_in && (iter - burn_in) %% thin == 0) {
    i_keep <- (iter - burn_in) / thin
    samples_R[i_keep, , ] <- R_current
    samples_p[i_keep, ] <- p_current
  }
  
  if (iter %% 100 == 0) cat("Iteration:", iter, "\n")
}

# Posterior summaries
posterior_mean_R <- apply(samples_R, c(2, 3), mean)
posterior_mean_p <- apply(samples_p, 2, mean)

cat("Posterior Mean Correlation Matrix:\n")
print(round(posterior_mean_R, 2))

cat("Posterior Mean Shared Marginal Probabilities:\n")
cat(paste(round(posterior_mean_p, 3), collapse = ", "), "\n")

