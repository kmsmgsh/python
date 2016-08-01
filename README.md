# Todo list for the project
### 

2016/8/1
- [ ] **log scale of the likelihood**
- [ ] **log transform for the GP**


2016/7/31
- [ ] **GP method in Estimation**

      - [x] Cholesky Decomposition for covariance matrix
            numpy.linalg.cholesky()
            but the covariance matrix is a problem
      - [x] off triangle matrix to vector
- [ ] **main part change**

      - [ ] Use the log and multiple to deal with the likelihoodfunction

2016/7/30

- [x] **code work**

   - [x] Finish "powerlaw" decay function for generator
   - [x] Finish Metropolis Hastings for "powerlaw" decay functin
- [ ] **Design Algorithm**
   - [ ] Design a methodology to update the GP



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
~The port for debug for generator
testProbabilityMode: When this parameter be truth, then open the mode of test mode.The transition probability for every inidvidual everyday will be printed to see whats wrong
~
[FINISHED]
# GP work procedure

标签（空格分隔）： GP MCMC workPlan

---

```flow
st=>start: Distance Matrix nxn
op1=>operation: Correlation Matrix
op2=>operation: Random sample from Gaussian distribution
op3=>operation: reconstruct to be add-ons on beta matrix
op4=>operation: Use new Beta Matrix to construct Likelihood
op6=>operation: update parameter
op7=>operation: update GP matrix
con=>condition: Enough iterations?
e=>end

st->op1->op2->op3->op4->op6->op7->con
con(no)->op3
con(yes)->e
```
After the final GP result, we need reorder the data with distance to get the distance decay function
