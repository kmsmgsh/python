import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import Metropolis as mp
import Metropolis2 as mp2
import Metropolis3 as mp3
import generator_temp as gc
import likelihood as lk
import coordinate as cr
from scipy.stats import beta

model1=gc.heteregeneousModel(100,0.4,0.3,10,True,True)
estimate=lk.Estimation(model1.record,model1.geo)
#Metro=mp.Metropolis(1000,estimate.BetaPosterior,0.1,0.4)
Metro=mp3.multiMetropolis(1000,[estimate.GammaPosteriorBeta0,estimate.GammaPosteriorGamma],[0.1,0.1],[0.4,0.4])
Metro.showplot(0)
Metro.printall(0)
Metro.showplot(1)
Metro.printall(1)