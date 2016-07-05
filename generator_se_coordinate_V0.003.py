import numpy as np
import matplotlib.pyplot as plt
import coordinate as cr
import matplotlib.animation as animation
'''
Changelog for V0.002
1.The method for calculate individual infected changed: From pr(object j infected by object i)=beta
to "infected pressure model" which means pr(object j get infected in end of day t)=exp(-beta*Infected)
rather than at least one people infected by j 
2. The method to calculate Beta also changed.
now beta function afford a matrix Beta which the entry is beta ij for probability that i infec j
3. The data structure changed
   the data structure no longer a huge matrix about what we know but instead for seperate arrays(or matrix)
'''

class heteregeneousModel:
    def __init__(self,num_people,beta0,gamma,phi):
        """
          Initialize the representation of the state vector as a list of integers, 
         [1, 0, 0, ...]
         And simulate the locations data for each object as np array([num_people,2])
        """
        self.state = np.zeros((num_people))
        self.state[0] = 1
        self.num_people=num_people
        self.beta0=beta0 #baseline beta
        self.phi=phi     #scale parameter
        self.gamma=gamma #remove rate

        #change start with here
        #I need add the geo-information into state
        self.geo=cr.geodata(num_people,"cluster",xbound=100.0,ybound=100.0)
        self.BetaMatrix()
        self.mainProcess()
    def transform(self,state):
        """
        transform the current state to next step:
        suspect(0)->infectious(1) with probability Lambda(state,beta)
        infectious(1)->removal(2) with probability np.exp(-gamma)
        returns the state after 1 day
        """
        p_S_to_I = self.Lambda(state)
        #now it is a num_people array
        p_I_to_R = np.exp(-self.gamma)
        prob_transitions = np.zeros((self.num_people,))
        prob_transitions[state==0] = p_S_to_I[p_S_to_I!=0]
        prob_transitions[state==1] = p_I_to_R
        Binomi=np.random.uniform(0, 1, prob_transitions.shape)
        transitions = prob_transitions > Binomi
        #
        self.InfectedOutput(state,prob_transitions,Binomi)
        #Debug function
        return state + transitions
    def InfectiousStop(self,state):
        """
         When all infectious are remove, then we can stop the 
         """
        if (np.sum(state==1)==0):
            return False                           #all infected people have removed
        else:
            return True
    def mainProcess(self):                    #simulation procedure 
        """
         This is the process for the infection infect population in risk
         For each day, use transform function to get the state for next day
         return the number of infected people everyday
    
        """
        stateInit=self.state
        record = [stateInit.copy()]
        state = stateInit                            #state is the current state
        """
        for i in range(0, num_days):                 
            state=transform(state)
        record.append(state.copy())              #record the everday state
        for auto stop, use while instead of for
         """
        nday=0
        #########################################################
        #core code
        while (self.InfectiousStop(state)):
            state=self.transform(state)
            record.append(state.copy())
            nday+=1
        ##########################################################
        cout1 = [np.sum(day==1) for day in record] # list comprehension
        cout2=np.sum(record[-1]==2)# the total people who are infected(finally all of them are removed)
        self.record=np.array(record)
        self.cout1=cout1
        self.cout2=cout2
        self.nday=nday
        return [cout1,cout2]
    def BetaMatrix(self):
        DistanceMatrix=np.zeros([self.num_people,self.num_people])
        '''
        How to write the inner assignment for a matrix(instead of two loop)
         one optional optimal method: only calculate upper diagnoal entry
        '''
        #self.BetaMatrix=[for i,j in geo[,:]]
        X=self.geo
        DistanceMatrix=-2 * np.dot(X, X.T) + np.sum(np.square( X), 1).reshape(self.num_people, 1) + np.sum(np.square( X), 1).reshape(1, self.num_people)
        DistanceMatrix=np.sqrt(DistanceMatrix)
        '''
        for i in range(self.num_people):
            for j in range(self.num_people):
                DistanceMatrix[i][j]=np.linalg.norm(self.geo[i,0:2]-self.geo[j,0:2])#Euclidean distance
        '''
        #BetaMatrix=[self.beta0*np.exp(-i/self.phi) for i in DistanceMatrix] # list comprehension
        BetaMatrix=self.beta0*np.exp(-DistanceMatrix/self.phi)
        #self.BetaMatrix=np.array(BetaMatrix)
        self.BetaMatrix=BetaMatrix
    def Lambda(self,state):
        '''
        Lambda function returns a transform probability for every state to next step(S->I, I->R)
        This Lambda function is based on the "infected pressure" model which is differ from the previous one "Infected probability model".
        Infected pressure model is talking about that each infected people have a pressure to the surrounding suspected people, instead of
        the "probability at least one Infected object infect object j". The probability of j get infected in the end of the day is the 
        cumulative pressure for the infected objects surrouding j.
        For more detail, plz check "record 6.23.pdf" 2.1
        '''
        probInfect=np.zeros((self.num_people))
        '''
        for i in zip(*np.where(state==0)):
            probInfect[i]=sum(self.BetaMatrix[i][state==1])
        '''
        ################################################
        probInfect[state==0] = self.BetaMatrix[:, state==1].sum(1)[state==0]
        probInfect=1-np.exp(-probInfect)
        #################################################
        return probInfect
    def Animate(self):
        numframes = self.nday+1
        numpoints = self.num_people
        color_data =np.zeros((numframes, numpoints)) 
        color_data[self.record==1]=0.9
        color_data[self.record==2]=0.5
        x, y, c = np.random.random((3, numpoints))
        x=self.geo[:,0]
        y=self.geo[:,1]
        c=np.zeros(numpoints)
        c[0]=1
        fig = plt.figure()
        scat = plt.scatter(x, y, c=c, s=70)
        def update_plot(i, data, scat):
            scat.set_array(data[i])
            return scat,
        ani = animation.FuncAnimation(fig, update_plot, frames=range(numframes),
                                  fargs=(color_data, scat),interval=1000)
        plt.show()
    def InfectedOutput(self,state,prob_transitions,Binomi):
        print("Probabitliy")
        print (prob_transitions[prob_transitions>Binomi])
        print("dice")
        print(Binomi[prob_transitions>Binomi])
        print("Location")
        print (self.geo[prob_transitions>Binomi,:])
        
    
model1=heteregeneousModel(100,0.3,0.03,10)
"""
print(model1.record)
print(model1.cout1)
print(model1.cout2)
plt.scatter(model1.geo[:,0],model1.geo[:,1])
plt.show()
record=np.array(model1.record)
print(record[0,:])
plt.scatter(model1.geo[record[0,:]==1,0],model1.geo[record[0,:]==1,1],color="red")
plt.scatter(model1.geo[record[0,:]==0,0],model1.geo[record[0,:]==0,1],color="blue")
plt.show()
#model1.Animate()
print(model1.record)
"""

'''
def main():
    numframes = model1.nday+1
    numpoints = model1.num_people
    color_data =np.zeros((numframes, numpoints)) 
    color_data[model1.record==1]=0.9
    color_data[model1.record==2]=0.5
    x, y, c = np.random.random((3, numpoints))
    x=model1.geo[:,0]
    y=model1.geo[:,1]
    c=np.zeros(numpoints)
    c[0]=1
    fig = plt.figure()
    scat = plt.scatter(x, y, c=c, s=70)
    ani = animation.FuncAnimation(fig, update_plot, frames=range(numframes),
                                  fargs=(color_data, scat),interval=1000)
    plt.show()

def update_plot(i, data, scat):
    scat.set_array(data[i])
    return scat,
'''
model1.Animate()
