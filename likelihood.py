import numpy as np
import matplotlib.pyplot as plt
import coordinate as cr
import matplotlib.animation as animation
import generator_temp as gc
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
from scipy import optimize as op
class Estimation:
    def __init__(self,record,geo,phi=10):
        '''
        record is a 2-dimensional numpy array which is n*d. n is number of people, d is the infectious time
        '''
        self.record=record
        self.geo=geo
        [self.days,self.num_people]=record.shape
        self.phi=phi
    def BetaMatrix(self,beta0,gamma):
        DistanceMatrix=np.zeros([self.num_people,self.num_people])
        X=self.geo
        DistanceMatrix=-2 * np.dot(X, X.T) + np.sum(np.square( X), 1).reshape(self.num_people, 1) + np.sum(np.square( X), 1).reshape(1, self.num_people)
        DistanceMatrix=np.sqrt(DistanceMatrix)    
        BetaMatrix=beta0*np.exp(-DistanceMatrix/self.phi)
        return BetaMatrix
    
    def transform_prob(self,state,gamma,BetaMatrix):
        """
        Special edition for Likelihood
        set the none infectious as 1
        transform the current state to next step:
        suspect(0)->infectious(1) with probability Lambda(state,beta)
        infectious(1)->removal(2) with probability np.exp(-gamma)
        returns the state after 1 day
        """
        p_S_to_I = self.Lambda(state,BetaMatrix)
        #now it is a num_people array
        p_I_to_R = 1-np.exp(-gamma)
        prob_transitions = np.zeros((self.num_people,))
        if sum(p_S_to_I[p_S_to_I!=0])!=0:
            prob_transitions[state==0] = p_S_to_I[p_S_to_I!=0]
        prob_transitions[state==1] = p_I_to_R
        return prob_transitions
    def Lambda(self,state,BetaMatrix):
        '''
        Lambda function returns a transform probability for every state to next step(S->I, I->R)
        This Lambda function is based on the "infected pressure" model which is differ from the previous one "Infected probability model".
        Infected pressure model is talking about that each infected people have a pressure to the surrounding suspected people, instead of
        the "probability at least one Infected object infect object j". The probability of j get infected in the end of the day is the 
        cumulative pressure for the infected objects surrouding j.
        For more detail, plz check "record 6.23.pdf" 2.1
        '''
        probInfect=np.zeros((self.num_people))
        probInfect[state==0] = BetaMatrix[:, state==1].sum(1)[state==0]
        probInfect=1-np.exp(-probInfect)
        return probInfect
    def Likelihood(self,parameter):
        beta0=parameter[0]
        #beta0=parameter
        gamma=parameter[1]
        #gamma=0.03
        recordN=np.delete(self.record,0,0)#recordN is a matrix that have no first column for record 1,0,0,0,0,0
        change=np.vstack((recordN,self.record[self.days-1,:]))
        change=change-self.record
        change=np.delete(change,-1,0)#change is a matrix to record the change of state, which describe the change and result in record N
        BetaMatrix=self.BetaMatrix(beta0,gamma)
        Likeli=np.ones((self.num_people,1)).T
        for column in self.record:
            #print(column)
            #print("probability")
            k=self.transform_prob(column,gamma,BetaMatrix)
            #print(k)
            Likeli=np.vstack((Likeli,k))
            #print(Likeli)
        Likeli=np.delete(Likeli,-1,0)#delete the final row of the probability matrix(final row is no next generation so that the probaiblity matrix is 1 and 0 )
        Likeli=np.delete(Likeli,0,0)
        Likeli[change==0]=1-Likeli[change==0]
        Likelihood=Likeli.prod(1).prod()
        #print(self.record)
        #print(Likeli)
        #print(Likelihood)
        return (-Likelihood) #Because our optim function is for minimize
    def BetaLikelihoodQ(self,beta0):
        gamma=0.03
        return self.Likelihood(np.array((beta0,gamma)))
    def GammaLikelihoodQ(self,gamma):
        beta0=0.38
        return self.Likelihood(np.array())
    def BetaPosterior(self,beta0,a=1,b=1):
        from scipy.stats import beta
        return beta.pdf(beta0,a,b)*self.BetaLikelihoodQ(beta0)

'''
model1=gc.heteregeneousModel(100,0.4,0.3,10,True,False)
model1.Animate()
estimate=Estimation(model1.record,model1.geo,model1.phi)
r=np.linspace(0.1,0.7,200)
k=0.03*np.ones(200)
n= np.column_stack((r,k))
#for i in n:
#    print(estimate.Likelihood(i))]
k=[estimate.Likelihood(i) for i in n]
plt.plot(r,k)
k=0.4*np.ones(200)
n= np.column_stack((k,r))
k2=[estimate.Likelihood(i) for i in n]
plt.plot(r,k2)
plt.show()
#plt.show()
print("optimize!")
estimate.Likelihood(np.array((0.4,0.03)))

print("results")
#k=minimize(estimate.Likelihood,x0=np.array((0.001,0.03)),method="L-BFGS-B",bounds=np.array(([0.001,0.5],[0.001,0.9])))
#k=minimize(estimate.Likelihood,x0=0.001,method="BFGS",bounds=np.array((0.001,0.5)),tol=1e-190)
x0=[0.3,0.03]
#k=minimize(estimate.Likelihood,x0,method="BFGS")
#k=minimize(estimate.BetaLikelihoodQ,0.3,method="BFGS")
k= minimize(estimate.BetaLikelihoodQ,x0=0.3, method='L-BFGS-B')

k= minimize(estimate.BetaLikelihoodQ,x0=0.3, method='Nelder-Mead')
print(k)
#k=minimize_scalar(estimate.BetaLikelihoodQ)
bnds = np.array(([0.001,0.999],[0.001,0.999]))
k1=minimize(estimate.Likelihood, [0.3,0.02],bounds=bnds, method='Nelder-Mead')#Nelder-Mead can not handle with constrains, But at least works
x0 = [0.3,0.2]
k1=minimize(estimate.Likelihood, x0, method='BFGS')#BFGS do not work
k1=minimize(estimate.Likelihood, x0, method='Powell')#0.4086 and 73.70
k1=minimize(estimate.Likelihood, x0, method='CG')#return initial Value
k1=minimize(estimate.Likelihood, x0, method='COBYLA')#[ 0.40857675,  5.79367207]

print(k1)
#k1=minimize(estimate.Likelihood, [0.3,0.02],bounds=bnds, method='Newton-CG')#Newton CG need jacobian, do not work
k1=minimize(estimate.Likelihood, [0.3,0.02],bounds=bnds, method='SLSQP')#do not work, return initial Value
k1=minimize(estimate.Likelihood, [0.3,0.02],bounds=bnds, method='TNC')# do not work, return initial Value
k1=minimize(estimate.Likelihood, [0.3,0.02],bounds=bnds, method='L-BFGS-B')# do not work, return initial Value
#op.fmin_bfgs(estimate.Likelihood,[0.3,0.02,])
#k1=op.brute(estimate.Likelihood, ((0, 1), (0, 0.999)))
#rranges = (slice(0, 1, 0.01), slice(0, 1, 0.01))
#k1=op.brute(estimate.Likelihood, rranges)
print(k1)
'''