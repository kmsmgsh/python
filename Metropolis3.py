import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
class Metropolis:
    def OnestepMetropolis(self,density,theta,sigma,i:"update which parameter"):
        theta_star=np.exp(np.random.normal(np.log(theta[i]), sigma))
        thetastar=theta.copy()
        thetastar[i]=theta_star
        #Accept the new beta value with the probability f(beta_start)/f(beta)
        p=min(density(thetastar)/density(theta),1)####################Transfor p/p,or use log normal:btw: p/p=1 solved
        if np.random.uniform(0,1)<p:
            #Accept the new Value
            return [1,theta_star]
            #count the number of accept
        else:
            return [0,theta[i]]       
class multiMetropolis(Metropolis):
    def __init__(self,IterNa,density,initial_parameter,sigma):
        self.IterNa=IterNa
        self.initial_parameter=np.array(initial_parameter)
        self.dimension=self.initial_parameter.size
        self.density=density
        self.sigma=sigma
        self.Mainprocess()
    def Mainprocess(self):
        parameter=self.initial_parameter
        record=parameter
        Accept=np.zeros((self.IterNa,self.dimension))
        for i in range(0,self.IterNa):
            for j in range(0,self.dimension):
                result=self.OnestepMetropolis(self.density[j],parameter,self.sigma[j],j)
                Accept[i,j]=result[0]
                parameter[j]=result[1]
            record=np.vstack((record,parameter))
        self.record=record
        self.Accept=Accept
    def showplot(self,i):
        plt.plot(range(self.IterNa+1),self.record[:,i])
        plt.show()
        plt.hist(self.record[:,i],bins=50)
        plt.show()
        plt.plot(self.record[:,0],self.record[:,1])
        plt.show()
    def printall(self,i):
        print("Accept rate is")
        print(sum(self.Accept[:,i])/self.IterNa)
        print("Mean is")
        print(np.mean(self.record[:,i]))