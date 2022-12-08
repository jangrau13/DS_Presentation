## Normalization

As already explained in the previous lectures. It is beneficial to normalize the input variables.

Usually this is done with batch normalization.

In NLP however, things are not as they normally are.

A batch normalization has the property, that each single sample is dependent on all training samples. Furthermore, in contrast to image processing or or other optimizations, text sequences may not be always the same length, making the calculations more cumbersome and might lead to instability if batch normalization is naively implemented.


Take formulas 6.45 until 6.46 and implement it. 

\beta and \gamma are introduced to to be adjustable to the whole activation function.

insert image from book batch vs normalization

import random
beta = random.uniform(-1, 1)
gamma = random.uniform(-1,1)
epsilon = 0.00001

v_i = x[0]
mu = 1 / v_i.shape[0] * np.sum(v_i)
omega_squared = 1 / v_i.shape[0] * np.sum((v_i - mu)**2)
Layer_Norm = gamma * (v_i - mu) / (np.sqrt(omega_squared) + epsilon) + beta
print(Layer_Norm)