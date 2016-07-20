import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import Metropolis as mp
import Metropolis2 as mp2
import generator_temp as gc
import likelihood as lk
import coordinate as cr
from scipy.stats import beta



model1=gc.heteregeneousModel(100,0.4,0.3,10,True,True,"powerlaw")
model1.Animate()