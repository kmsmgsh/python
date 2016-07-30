# Todo list for the project
### 
2016/7/30

- [ ] **code work**

   - [x] Finish "powerlaw" decay function for generator
   - [ ] Finish Metropolis Hastings for "powerlaw" decay functin




2016/7/29

- [x] **Notes development tool**

    - [x] jupyter notebook with markdown syntax 1 hour with two poromodo
    - [x] a github blog with markdown to describe my work(this document shows half of it)  
    - [x] The rightnow work procedure
-  [x]  **Theoretical work** 

    - [x] 1 hour reading about the Gaussian Process Machine Learning(more than 1 hour and not finish reading 2.2)(Theoretic is a tough work....than expected)
    - [x] Read the introduction part for MCMC for Variationally Sparse Gaussian process
- [ ] **Code work**

    - [x] Parameterise the Code (For powerlaw part with 3 parameter as beta0, sigma, omega)
    - [ ] Design the work logistic for GP
    There have some problem with the "vector" and "Matrix" transform function.
    We need transfor the Gaussian vector to BetaMatrix base on DistanceMatrix
    Have to find the operator for upper diagnoal distance
**Code work procedure**
# Code tast remainder

标签（空格分隔）： python generator augument   

---

- [ ] **Debug remainder**
 - [ ] parameter
 - [ ] method: cluster uniform
 - [ ] Distance Method: gradient, powerlaw

---

 

 1. parameter+uniform+gradient
```python
model1=gc2.heteregeneousModel(100,[0.4,10,0.3],False,False,"gradient","uniform")
```
passed with problem1

 2. parameter+uniform+powerlaw
```python
model1=gc2.heteregeneousModel(100,[0.9,0.2,1.5,0.3],False,False,"powerlaw","uniform")
```
The parameter map to [beta0=0.9,sigma=0.2,omega=1.5,gamma=0.3]

not work
no spread

```python
model1=gc2.heteregeneousModel(100,[1.9,0.2,1.5,0.3],False,False,"powerlaw","uniform")
```
still not work
no spread
[3,0.2,1.3,0.3]
still not spread

maybe is the problem with code
[5,0.2,1.3,0.3]
still not spread


```python
model1=gc2.heteregeneousModel(100,[5,0.2,1,0.3],False,False,"powerlaw","uniform",True)
```
[5,0.2,1,0.3]
start spread, looks not that bad
---
problem remain:
1.the plt.ion() will cause the plot not responding


**New task**
The port for debug for generator
testProbabilityMode: When this parameter be truth, then open the mode of test mode.The transition probability for every inidvidual everyday will be printed to see whats wrong
