import matplotlib.pyplot as mt
import pandas as pd
from sklearn.datasets import make_classification

x, y = make_classification(n_samples=1000, n_features=2, n_redundant=0,
                           n_clusters_per_class=1, weights=[.90], random_state=1)
print(x)
print(y)
x1 = pd.DataFrame(x, columns=['f1', 'f2'])
x2 = pd.DataFrame(y, columns=['target'])

df = pd.concat([x1, x2], axis=1)
print(df.head())
mt.scatter(df['f1'],df['f2'])

from imblearn.over_sampling import SMOTE

oversample=SMOTE()
x,y=oversample.fit_resample(df[['f1,f2']],df['target'])
print(x)
