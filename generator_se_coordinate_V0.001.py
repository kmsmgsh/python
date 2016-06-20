import numpy as np
import matplotlib.pyplot as plt
import coordinate as cr
import pdb
'''
This version is add the geo information about the suspect people
The simulation for geo information is in coordinate.py



# Most part directly copy from generator V0.03.py
Maybe I should think out a method to peacefully the version with geo information
with the verson without geo-information
On the other hand, promot the "code reusability"
I have to say,it's so stupid to copy and paste code


# The import change is
    The function Lambda, which calculate the probability that object
    j infected
    The data structure, status is newly as a matrix as
    object state coordinate
    1      1     (30.553,34.984)
    2      0     (40.336,40.789)
    ...
    Maybe use a class people to present the data more better?
    (But consider potential matrix parallel calculation, I just not to involve
    the class this time
    
# Maybe can use dynamic graph to show the process SIR model

'''

"""
##This function is change mostly around the code

We should link the probability to coordinate as distance more far,
the probability is less
Another problem is the probability object j get infected by object i
is not same, so we cannot use power to calculate object j not get
infected by other objects
Also, we sill search for the coordinate for state==1 and calculate the
probability 

"""


def Lambda(state, beta,num_people):
    """
    beta is the probability that a S is infected by one particular I
    returns the probabilty of a S becoming a I (transition 0 to 1)
    """
    I_count = np.sum(state[:,0]==1)
    S_count = np.sum(state[:,0]==0)
    """
    Assume that the probability that object j infected by object i is beta,
    so the fair "object j get infected" is "at least one other object infect object j",
    the probability is 1-q, q means the probability that no one infect object j.
    q equals to object 1 not infect j, multiple object 2 not infect j,.....,
    so the p is as following formula
    
    #p = 1.0 - (1.0-beta)**S_count

    #Because the probabilty object j infected by objct i is not identical, the
    probability is not as **S_count but product (1-beta(j,i)) for all infected i
    This time we need a new function beta(j,i) to calculate the probability that
    j infected by i
    So this time     p=1.0-product(for all infected i)(1-beta(i,j))
    """
    probInfect=np.zeros((num_people,1))
    #########################################################################
    for j in zip(*np.where(state[:,0]==0)):
        individualInfect=np.zeros((num_people,1))
        for i in zip(*np.where(state[:,0]==1)):
            individualInfect[i]=Beta(state,i,j,beta)
        probInfect[j]=1-np.prod(1-individualInfect[individualInfect!=0])
    ########################################################################
    return probInfect
    """
    probInfect is a array with num_people *1 dimension, with the suspect object,
    it is equal to the probability get infected, the infected and removal subject
    is equal to 0
    """

def Beta(state,i,j,beta): #i infect j
    """
    I think I need something like datafarme in R so that I can use the syntax like
    state$coordinate or state.coordinate to make it easier....
    """
    #column 2(start 0) is the column store coordinate
    distance=np.linalg.norm(state[i,1:3]-state[j,1:3])
    """
    This is Euclidean distance between object i and j
   
    Assume that baseline probability is beta, we need a function valued in (0,1]
    and decrease with increasing of distance
    ########################################################################
    We consider the exponential function this time
    probability p=beta*exp(-distance)
    From the plot of exp(-x), we found that exp(-x) converge to zero as x greater than 5
    So we try to transform the distance from (0,5)
    Explortatory design is distance=distance/100*5
    This is a point worth discussion
    ########################################################################

    """
    return beta*np.exp(-(distance)/100)

def transform(state,num_people,beta=0.003, gamma=0.5):
    """
    transform the current state to next step:
    suspect(0)->infectious(1) with probability Lambda(state,beta)
    infectious(1)->removal(2) with probability np.exp(-gamma)
    returns the state after 1 day
    """
    p_S_to_I = Lambda(state, beta,num_people)
    #now it is a num_people array
    p_I_to_R = np.exp(-gamma)
    prob_transitions = np.zeros((num_people,))
    prob_transitions[state[:,0]==0] = p_S_to_I[p_S_to_I!=0]
    prob_transitions[state[:,0]==1] = p_I_to_R
    transitions = prob_transitions > np.random.uniform(0, 1, prob_transitions.shape)
    return state[:,0] + transitions
    #state->state[:,0]


def InfectiousStop(state):
    """
    When all infectious are remove, then we can stop the 
    """
    if (np.sum(state[:,0]==1)==0):
        return False                           #all infected people have removed
    else:
        return True
    #state->state[:,0]
    
def mainProcess(num_people,beta, gamma):                    #simulation procedure 
    """
    This is the process for the infection infect population in risk
    For each day, use transform function to get the state for next day
    return the number of infected people everyday
    
    """
    stateInit=initial(num_people)

    
    record = [stateInit[:,0].copy()]
    state = stateInit                            #state is the current state
    """
    for i in range(0, num_days):                 
        state=transform(state)
        record.append(state.copy())              #record the everday state
    for auto stop, use while instead of for
    """
    while (InfectiousStop(state)):
        state[:,0]=transform(state,num_people,beta, gamma)
        record.append(state[:,0].copy())
    print(record)
    cout1 = [np.sum(day==1) for day in record] # list comprehension
    cout2=np.sum(record[-1]==2)                # the total people who are infected(finally all of them are removed)
    return [cout1,cout2]

    
def initial(num_people):
    """
    Initialize the representation of the state vector as a list of integers, 

    [1, 0, 0, ...]
    """
    state = np.zeros((num_people,1))
    state[0] = 1
    #change start with here
    #I need add the geo-information into state
    geo=cr.geodata(num_people,"cluster",xbound=100.0,ybound=100.0)
    state=np.hstack((state,geo))
    return state

a=initial(10)

print(mainProcess(10,0.5,0.3))

