import pandas as pd
from sklearn.preprocessing import StandardScaler,normalize
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import numpy as np

data = pd.read_csv('animeFinal.csv',index_col=0)
data = data.iloc[: , :-1]  #Se elimina la columna Score.

scale =  StandardScaler().fit_transform(data)  # Normalización de la data
normalizacion = normalize(scale) 
Ndata= pd.DataFrame(normalizacion) # Creación del nuevo dataframe normalizado
  
PCA = PCA(n_components = 2) 
X = PCA.fit_transform(Ndata)  # Reduccion de dimensiones del nuevo dataframe
X = pd.DataFrame(X) 
X.columns = ['C1', 'C2'] 

db = DBSCAN(eps=0.3, min_samples=13).fit(data)  # DBSCAN
labels = db.labels_

no_clusters = len(np.unique(labels) ) # Aproximación de clusters
no_noise = np.sum(np.array(labels) == -1, axis=0) # Aproximación de outlayers

print('Número estimado de clusters: ' + str(no_clusters))
print('Número estimado de noise points: ' + str(no_noise))

#Basado en https://github.com/christianversloot/machine-learning-articles/blob/main/performing-dbscan-clustering-with-python-and-scikit-learn.md y https://www.kaggle.com/code/vipulgandhi/gaussian-mixture-models-clustering-explained/notebook