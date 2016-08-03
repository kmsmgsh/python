import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
#import Metropolis as mp
#import Metropolis2 as mp2
import Metropolis3 as mp3
#import generator_temp as gc
import generator_temp_transprob as gc2
#import likelihood as lk
import likelihodPhi as lk2
import coordinate as cr
from scipy.stats import beta
from functools import partial
#plt.ion()
#plt.style.use('ggplot')
#model1=gc2.heteregeneousModel(100,[0.4,10,0.3],True,True,"gradient","uniform",True)
model1=gc2.heteregeneousModel(100,[5,0.2,1,0.3],True,True,"powerlaw","uniform",True)
model1.Animate()
#estimate=lk2.Estimation(model1.record,model1.geo,method="powerlaw")
estimate=lk2.Estimation(model1.record,model1.geo,method="gradient")
#Metro=mp3.multiMetropolis(1000,[estimate.GammaPosteriorBeta0,estimate.GammaPosteriorGamma,estimate.GammaPosteriorPhi],[0.1,0.1,5],[0.5,0.5,0.4])
#Metro=mp3.multiMetropolis(1000,[partial(estimate.GammaPriorGeneralPosterior,i=0),partial(estimate.GammaPriorGeneralPosterior,i=1),partial(estimate.GammaPriorGeneralPosterior,i=2)],[0.1,0.1,5],[0.5,0.5,0.4])
#Metro=mp3.multiMetropolis(1000,[estimate.GammaPosteriorBeta0,estimate.GammaPosteriorGamma],[0.1,0.1],[0.4,0.4])
#Metro=mp3.multiMetropolis(1000,[partial(estimate.GammaPriorGeneralPosterior,i=0),partial(estimate.GammaPriorGeneralPosterior,i=1),partial(estimate.GammaPriorGeneralPosterior,i=2),partial(estimate.GammaPriorGeneralPosterior,i=3)],[3,0.1,0.9,1],[0.5,0.5,0.4,0.4])
Metro=mp3.multiMetropolis(1000,[partial(estimate.GammaPriorGeneralPosterior,i=0),partial(estimate.GammaPriorGeneralPosterior,i=1),partial(estimate.GammaPriorGeneralPosterior,i=2)],[0.1,0.9,1],[0.5,0.5,0.4])
Metro.showplot(0)
Metro.printall(0)
Metro.showplot(1)
Metro.printall(1)
Metro.showplot(2)
Metro.printall(2)
Metro.showplot(3)
Metro.printall(3)