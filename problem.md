# Problem remain
### 1
- [ ] The chosen of Random Walk Metropolis MVN correlation function
- [ ] The negative part of the GP ? (we need some negative effect on BetaMatrix?)
- [ ] How the Cholesky work during MCMC?
- [ ] When the population greater than 24, the determinant of the Correlation Matrix comes to zero
- [ ] The GP influence is too large for BetaMatrix

• The chosen of Random Walk Metropolis MVN correlation function
I directly use mvn(previousValue,Identity(n)) as randomwalk, but I feel bad about this.
And how to explore the Correlation structure for the posterior?
Well, this question has two parts to the answer. 

First, we must center the GP variables in order to be able to sample effectively. If f is a vector of GP values, and 

p(f) = MVN(0, K)

then instead, run the sampler on v, with 

p(v) = MVN(0, I)
and f=Lv, where LLT=K through the cholesky decomposition.

Second, we'll have to do something better than a random walk, I agree. We'll use a gradient-powered sampler like Hamiltonian Monte Carlo. Once we have the code working, this is a future step. For the moment, the random walk should work, so long as the problem is small enough, as we discussed. 
 
•  The negative part of the GP ? (we need some negative effect on BetaMatrix?)
During the log scale, we only have positive influence on distance decay function beta0*exp(-dij), is this okay?

I think you've understood this now :) 
•  How the Cholesky decomposition work during MCMC?
As I said before, every step of Metropolis-Hastings base on Covariance Matrix as Identity matrix(?)
And the Covariance Matrix sigma*exp(-|d-d`|/2l) is only use in Likelihood as prior
So how to use Cholesky decomposition?
Well, there are several MVNs goin on here, let's think about where the cholesky comes in. The first MVN is the proposal distribution for the sampler: I sugest you make that
N(theta_previous, c I)
 where I is the identitiy matrix and c is a small number.

second, we have 
p(f) = N(0, K). to compute this density, you would have to comute the cholesky decomposition of the covariance. BUT! don't compute this density. Instead, as I described above, compute the density
p(v) = N(0, I). and then inside the computation, use

f = L v

with

L = cholesky(K)

A quick side- exercise: show that f is correctly distributed under this transform.

For the moment, you only need to compute the cholesky once, since the distances are constaant and the covariance function is fixed. recall that we suggested using

k(d, d') = \sigma exp( - ||d-d'||^2 / a)

where a is the median of all the distances. Recall also that the matrix K is composed by computing this function for all distance-pairs. 
•  When the population greater than 24, the determinant of the Correlation Matrix comes to zero
For the Likelihood function, the determinant of correlation Matrix will be very small and close to zero. This shows during the population greater than 20-25. Then the log |K| comes to inf.

You should never compute the determinant of a covariance matrix! It is always going to be numerically poor, because the floating point representation falls apart near 0. 

instead, compute the log-determinant like this

L = cholesky(K)
log_det_K = np.sum(np.log(np.square(np.diag(L))))

Side exercise 2: show that this equation gives the log-determinant of K. 


•  The GP influence is too large for BetaMatrix
The beta matrix is around 10^-1~10^-90, but the random effect from GP is exp(MVN(0,SIGMA)) which is around 1, is too large to influence the betaMatrix
I think you've solved this problem already. But there's one more thing you can do: try rescaling the kernel function, by changing the \sigma parameter in from of the equation. 
