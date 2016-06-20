import pdb
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
def initial(num_people):
    """
    Initialize the representation of the state vector as a list of integers, 

    [1, 0, 0, ...]
    """
    state = np.zeros(num_people)
    state[0] = 1
    return state

def Lambda(state, beta):  #homogeneous model for lambda 
    """
    beta is the probability that a S is infected by one particular I
    returns the probabilty of a S becoming a I (transition 0 to 1)
    """
    S_count = np.sum(state==0)
    """
    Assume that the probability that object j infected by object i is beta,
    so the fair "object j get infected" is "at least one other object infect object j",
    the probability is 1-q, q means the probability that no one infect object j.
    q equals to object 1 not infect j, multiple object 2 not infect j,.....,
    so the p is as following formula
    """
    p = 1.0 - (1.0-beta)**S_count
    return p

def transform(state, beta=0.003, gamma=0.5):
    """
    transform the current state to next step:
    suspect(0)->infectious(1) with probability Lambda(state,beta)
    infectious(1)->removal(2) with probability np.exp(-gamma)
    returns the state after 1 day
    """
    p_S_to_I = Lambda(state, beta)
    p_I_to_R = np.exp(-gamma)
    prob_transitions = np.zeros(state.size)
    prob_transitions[state==0] = p_S_to_I
    prob_transitions[state==1] = p_I_to_R
    transitions = prob_transitions > np.random.uniform(0, 1, state.shape)
    return state + transitions


def InfectiousStop(state):
    """
    When all infectious are remove, then we can stop the 
    """
    if (np.sum(state==1)==0):
        return False                           #all infected people have removed
    else:
        return True
    
    
def mainProcess(num_people,beta, gamma):                    #simulation procedure 
    """
    This is the process for the infection infect population in risk
    For each day, use transform function to get the state for next day
    return the number of infected people everyday
    
    """
    stateInit=initial(num_people)                       
    record = [stateInit.copy()]
    state = stateInit                            #state is the current state
    """
    for i in range(0, num_days):                 
        state=transform(state)
        record.append(state.copy())              #record the everday state
    for auto stop, use while instead of for
    """
    while (InfectiousStop(state)):
        state=transform(state,beta, gamma)
        record.append(state.copy())
    #print("record")
    #print(record)
    cout1 = [np.sum(day==1) for day in record] # list comprehension
    cout2=np.sum(record[-1]==2)                # the total people who are infected(finally all of them are removed)
    #print('the count of infectious everday')
    #print('the final infected people')
    #print(cout2)                            
    #print(cout1)
    return [cout1,cout2]
    #plt.plot(cout1)
    #plt.show()
    
def iteratorMainProcess(num_people=200,  num_simulations=1000,beta=0.003, gamma=0.5):      #simulate "redo" times to get average number
    """
    Simulate the process for several times to say the overall trend and the differences
    plot the infected number against time(per day)
    """
    results = np.array([mainProcess(num_people,beta, gamma) for _ in range(num_simulations)])
    totalInfectious=results[:,1]
    results=results[:,0]
    results = np.array(results)
    #plt.plot(results, 'b')
    """
    Following code is transform the result to same length array
    Because of automatic stop, the length of day for stop infectious for each simulation
    is not same
    Maybe should use another function to make this work
    
    """
    daysEachSimulation=[len(i) for i in results]                    #for auto stop method, the dimensions of the infected people everyday is not equal
    daysEachSimulationArray=np.array(daysEachSimulation)            #In order to plot the data, maybe we should make it into same dimension with 0 append
    rank=max(daysEachSimulationArray)
    ResultsWithSameDimension=np.zeros([len(results),rank])
    for i in range(0,len(results)):
        for j in range(len(results[i])):
            ResultsWithSameDimension[i][j]=results[i][j]
    #print(ResultsWithSameDimension)
    #print(max(daysEachSimulation)-daysEachSimulation)
    




    #plt.plot(ResultsWithSameDimension.T,'b')
    #plt.show()
   # plt.clr()
    #print(totalInfectious)
    #plt.hist(totalInfectious)
    #plt.show()
    #return (np.mean(totalInfectious))
    return sum(totalInfectious)/len(totalInfectious)


n=50
simulate=5
surfaces=np.array([0.01,0.1,iteratorMainProcess(n,simulate,0.01,0.1)])
'''
for beta in np.arange(0.01,0.2,0.0008):
    for gamma in np.arange(0.1,0.8,0.08):
        x=[beta,gamma,iteratorMainProcess(n,simulate,beta,gamma)]
        surfaces=np.vstack((surfaces,x))
        #print(x)
print([surfaces[0,0], surfaces[0,1], surfaces[0,2]])
X=surfaces[:,0]
Y=surfaces[:,1]
Z=surfaces[:,2]
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_zlim(38,50)
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
surf=ax.plot_surface(X,Y,Z)
#fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
'''


'''
failed to plot 3d surface'''


"""
here's the marginal plot which choose gamma as constant
"""
surfaces=np.array([0.01,0.1,iteratorMainProcess(n,simulate,0.01,0.1)])
for beta in np.arange(0.01,0.2,0.0008):
    #for gamma in np.arange(0.1,0.8,0.08):
        gamma=0.1
        x=[beta,gamma,iteratorMainProcess(n,simulate,beta,gamma)]
        surfaces=np.vstack((surfaces,x))
        #print(x)
#print([surfaces[0,0], surfaces[0,1], surfaces[0,2]])
X=surfaces[:,0]
Y=surfaces[:,1]
Z=surfaces[:,2]
plt.plot(X,Z)
plt.show()


'''
if __name__ == "__main__":
    num_people = 20
    num_days = 10
    num_simulations = 100
    results = [mainProcess(num_people, num_days) for _ in range(num_simulations)]

    results = np.array(results)
    plt.plot(results.T, 'b')
'''
