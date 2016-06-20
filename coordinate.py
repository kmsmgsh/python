import numpy as np
import matplotlib.pyplot as plt

class coordinate:
    def __init__(self,x=0.0,y=0.0):
        self.x=x
        self.y=y
a=coordinate()

def geodata(num_people,method="uniform",xbound=100.0,ybound=100.0):
    """
    Generator a n*2 array, which is the x-coordinate and y-coordinate
    in each row,for n people
    The method equal to uniform means each point in 2-D surface is uniform
    distributed
    The method equal to cluster means points x01,x02,...x0m,are normally distributed
    around a point x0, x0 is uniform in this surface
    """
    population=np.zeros((num_people,2))

    if method=="uniform":
        population[:,0]=np.random.uniform(0,xbound,num_people)
        population[:,1]=np.random.uniform(0,ybound,num_people)
        print(population)
        plt.scatter(population[:,0],population[:,1])
        #plt.show()
        return population
    if method=="cluster":
        num_cluster=int(np.random.uniform(0.1,1)*10+1)
        # the number of cluster follow uniform 1,2,...10
        # problem: how to scientifically define the number of cluster?
        people_set=cluster_people(num_people,num_cluster)
        center_point=np.zeros((num_cluster,2))
        center_point[:,0]=np.random.uniform(0,xbound,num_cluster)
        center_point[:,1]=np.random.uniform(0,ybound,num_cluster)
        ######################################################
        iter=0
        cout=0
        for i in people_set:    
            for j in range(i):
                population[iter,:]=np.random.multivariate_normal(center_point[cout,:],25*np.identity(2))
                    #problem: how to detemine the covariance matrix?
                    #btw: 25 is only pick randomly.1 is too small in graph
                iter+=1
            cout+=1
        ######################################################
        plt.scatter(population[0,0],population[0,1],cmap=3)
        plt.scatter(population[1:num_people-1,0],population[1:num_people-1,1])
        #plt.show()
        #maybe I should put the show section into one part(rather than in each if)
        return population        
def cluster_people(num_people,num_cluster):
    '''
        The following code is try to split the number of poeple into num_cluster
        sets, which means n1 people in cluster1, n2 people in cluster2,..., and
        n1+n2+...+n3=num_people
        This function return an array which is num_people*1, each element is the
        number of people in each cluster
    '''
    if num_cluster==1:
        return num_people
    first=int (np.random.uniform(1,num_people-num_cluster))
    return np.hstack((first,cluster_people(num_people-first,num_cluster-1)))
'''     Following is  Test code
population=geodata(100,"uniform",100,100)
beta=0.3

for i in range(100):
    distance1=np.linalg.norm(population[1,:]-population[i,:])
    print("distance is")
    print(distance1)
    print("probability is")
    print(beta*np.exp(-(distance1)/20))
'''
#population=geodata(100,"cluster",100,100)




