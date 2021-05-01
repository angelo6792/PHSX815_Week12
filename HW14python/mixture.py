#! /usr/bin/env python

# imports of external packages to use in our code
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans



# Generates data
X, y_true = make_blobs(n_samples=4000, centers=2,
                       cluster_std=0.50, random_state=0)
X = X[:, ::-1]

kmeans = KMeans(2, random_state=0)
labels = kmeans.fit(X).predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis');



plt.show()
