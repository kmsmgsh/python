import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import Metropolis as mp
import generator_temp as gc
import likelihood as lk
import coordinate as cr
from scipy.stats import beta

model1=gc.heteregeneousModel(100,0.4,0.3,10,True,False)
estimate=lk.Estimation(model1.record,model1.geo)
Metro=mp.Metropolis(1000,estimate.BetaPosterior,0.2,0.4)
Metro.printAll()
Metro.showplot()