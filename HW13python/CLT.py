#! /usr/bin/env python

# imports of external packages to use in our code
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats



#exponential distributed population rate parameter
rate = 0.50

#Population mean
mu = 1/rate

# Population standard deviation
sd = np.sqrt(1/(rate**2))


# drawing 50 random samples of size 2 from the exponentially distributed population
sample_size = 2
df2 = pd.DataFrame(index= ['x1', 'x2'] )

for i in range(1, 51):
    exponential_sample = np.random.exponential((1/rate), sample_size)
    col = f'sample {i}'
    df2[col] = exponential_sample

# Calculating sample means and plotting their distribution
df2_sample_means = df2.mean()
sns.displot(df2_sample_means)


# drawing 50 random samples of size 500
sample_size=500

df500 = pd.DataFrame()

for i in range(1, 51):
    exponential_sample = np.random.exponential((1/rate), sample_size)
    col = f'sample {i}'
    df500[col] = exponential_sample


df500_sample_means = pd.DataFrame(df500.mean(),columns=['Sample means'])
sns.displot(df500_sample_means)



# drawing 50 random samples of size 5000
sample_size=5000

df5000 = pd.DataFrame()

for i in range(1, 51):
    exponential_sample = np.random.exponential((1/rate), sample_size)
    col = f'sample {i}'
    df5000[col] = exponential_sample


df5000_sample_means = pd.DataFrame(df5000.mean(),columns=['Sample means'])
sns.displot(df5000_sample_means)



plt.show()



