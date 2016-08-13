# Problem remain
### 1
- [x] The chosen of Random Walk Metropolis MVN correlation function
- [x] The negative part of the GP ? (we need some negative effect on BetaMatrix?)
- [x] How the Cholesky work during MCMC?
- [x] When the population greater than 24, the determinant of the Correlation Matrix comes to zero
- [x] The GP influence is too large for BetaMatrix

• Q:The chosen of Random Walk Metropolis MVN correlation function
I directly use mvn(previousValue,Identity(n)) as randomwalk, but I feel bad about this.
And how to explore the Correlation structure for the posterior?

A:Well, this question has two parts to the answer. 

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

###2
1. [x] The positive definite of covariance matrix:
       The cholesky decomposition would have an error "not positive definite".
       This happens link to parameter l in exp(-|x-x`|^2/2l).

       When l=1, this problem happens when population greater than 15~
       
       When l=0.1, this problem happens when population greater than 25~
       
       When l=0.01, this problem happens when population greater than 40~
       
       When l get smaller, the correlation matrix more closer to identity matrix
       
       (I used to think the greater the l is, the population limit is large. But this fact counter it.
       
       
       The issue is probably just about the numerical precision of the matrix. To help stabilize the eigenspectrum of the matrix, you can do

K = CovarianceMatrix()
K = K + np.eye(K.shape[0]) * 1e-9

Which should make the cholesky decomposition work nicely.

       
       
2. [x] When population comes small. the MCMC result (with constant GP) becomes really bad except the recover rate
       This also happens to MCMC without GP
       Most serious is beta0, scale parameter phi is not that bad. Remove rate gamma always behave good
       Also, the GP influence is large for small population
# 在此处输入标题

标签（空格分隔）： GP, epdemicModel,

---
##1.Equivelant
The model exp(log beta0+GP(0,Sigma)) is equivelent to the model exp(GP(log beta0,Sigma)). So the model f(d)=GP(0,Sigma)= previous model with parameter beta0=1, phi=0.
$$K(d)=exp(log \{\beta_{0}e^{-\phi_0d_{ij}}\}+GP(0,\Sigma))$$
And the model
$$K(d)=exp(GP(log \{\beta_{0}e^{-\phi_0d_{ij}}\},\Sigma))$$
is equivelant
So in this way, The model with $GP(0,\Sigma)$ is the model upon with parameter $\beta_0=0,\phi_0=1$


So I can use the framework from the parametric model to non-parametric model.
##2.Problem Reporting

The MCMC results for GP is not good.
There are two problems witht the GP MCMC results.

 1. The accept rate

The serious problem is accept rate is very small. The highest from the experiment is about 0.1 with sigma=1. It did not become good when set a little sigma like 0.4.
And some time it was not move from the Initial value which means accept rate is zero.

 2.The sensitivity about Initial Value 
 
With the problem about the accept rate, the MCMC result get stack at the begining value. 
    
From the picture below, we can see that the MCMC plot may perform good. However, when plot the Initial GP, 95% upper bound and 5% lower bound, median value of GP. These four lines is very close and sometime become one line show in the plot.

btw the code for 95% and 5% bound is 

    np.percent(GPrecord,95,axis=0)
    np.percent(GPrecord,95,axis=0)

But even for the Accept rate with 0.1(highest), it not shows very good.



From the result upon, I think it did not move from initial value to the posterior distribution.


If I set a zero initial value for GP. It runs like following. Which is totally wrong.




