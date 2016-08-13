# Problem for GP results

标签（空格分隔）： GP, epdemicModel, inference

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

 2. The sensitivity about Initial Value 
With the problem about the accept rate, the MCMC result get stack at the begining value. 
From the picture below, we can see that the MCMC plot may perform good. However, when plot the Initial GP, 95% upper bound and 5% lower bound, median value of GP. These four lines is very close and sometime become one line show in the plot.
btw the code for 95% and 5% bound is 

    np.percent(GPrecord,95,axis=0)
    np.percent(GPrecord,95,axis=0)

But even for the Accept rate with 0.1(highest), it not shows very good.This is the best result from simulation
![SimulationwithDiffExp][1]
From the result upon, I think it did not move from initial value to the posterior distribution.
The red line and blue line is upper 95% and lower 5% quantile of the GP kernerl functon with baseline or mean(yellow line)
The black and brown(just in same place) is the median value of GP and Initial value of GP
Green line is the true value from simulation

![The every line plot][2]
This is the general plot line of the GP results which tells a lie to us.
Because most the data is in the graph is in the graph upon but only little line goes well.

If I set a zero initial value for GP. It runs like following. Which is totally wrong.
![ZeroMean][3]
This graph shows that the GP kernel just runs as random and no posterior information in it. The random is based on the initial value(brown line)
![ZeroMeanWrongBaseline][4]


  [1]: https://raw.githubusercontent.com/kmsmgsh/python/master/SimulateWithEXPAcceptRate=0.104.png
  [2]: https://raw.githubusercontent.com/kmsmgsh/python/master/SimulatePowerlaw.png
  [3]: https://raw.githubusercontent.com/kmsmgsh/python/master/zeroMean.png
  [4]: https://raw.githubusercontent.com/kmsmgsh/python/master/ZeroMean2WrongBaseline.png
