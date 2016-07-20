import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
class Metropolis:
    def __init__(self,IterNa,density,initial,sigma):
        self.IterNa=IterNa
        self.density=density
        self.initial=initial
        self.Mainprocess(sigma)
    def printAll(self):
        print("Number of Iterative is")
        print(self.IterNa)
        print("Initial Value of Metropolis is")
        print(self.initial)
        print("initial pdf value of f")
        print(self.density(self.initial))
        print(self.density)
        print("Accept rate")
        print(self.accept/self.IterNa)
        print("Mean")
        print(np.mean(self.record))
    def Mainprocess(self,sigma):
        '''
        This is the random walk metropolis hastings with the random walk using normal distribution
        N(theta,sigma)
        The problem is how to get the Sigma so that the accept rate is around 44%
        '''
        '''
        Maybe need a function to handle the multi-variable occasion
        This just for 1-Dimension
        '''
        theta=self.initial
        record=np.array(theta)
        accept=0
        for i in range(self.IterNa):
            result=self.OnestepMetropolis(self.density,theta,sigma)
            theta=result[1]
            accept=accept+result[0]
            record=np.append(record,result[1])
        self.record=record
        self.accept=accept
    
    def OnestepMetropolis(self,density,theta,sigma):
        theta_star=np.exp(np.random.normal(np.log(theta), sigma))
        #Accept the new beta value with the probability f(beta_start)/f(beta)
        p=min(density(theta_star)/density(theta),1)####################Transfor p/p,or use log normal:btw: p/p=1 solved
        if np.random.uniform(0,1)<p:
            #Accept the new Value
            return [1,theta_star]
            #count the number of accept
        else:
            return [0,theta]       
    def showplot(self):
        plt.plot(range(self.IterNa+1),self.record)
        plt.show()
        plt.hist(self.record,bins=50)
        plt.show()
'''
Test code
sample=Metropolis(1000,norm.pdf,0)
sample.printAll()
print(sample.record)

sample.showplot()
'''