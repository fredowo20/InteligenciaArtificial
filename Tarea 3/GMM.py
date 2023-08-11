import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler,normalize
from sklearn.decomposition import PCA

data = pd.read_csv('animeFinal.csv',index_col=0)
data = data.iloc[: , :-1]  # Se elimina la columna Score.

scale =  StandardScaler().fit_transform(data)  # Normalización de la data
normalizacion = normalize(scale) 
Ndata= pd.DataFrame(normalizacion) # Creación del nuevo dataframe normalizado
  
PCA = PCA(n_components = 2) 
X = PCA.fit_transform(Ndata)  # Reducción de dimensiones del nuevo dataframe
X = pd.DataFrame(X) 
X.columns = ['C1', 'C2'] 

GMM = GaussianMixture(n_components = 8).fit(X) 
clasificacion = GMM.fit_predict(X)         # Clasificación a qué clusters pertenece cada anime

print("Clasificación: ") 
print(clasificacion) 

# print(len(clasificacion)) 

plt.scatter(X['C1'], X['C2'],  c = GaussianMixture(n_components = 8).fit_predict(X)) # Gráfica de clusters
plt.show() 

# Codigo basado en https://www.kaggle.com/code/vipulgandhi/gaussian-mixture-models-clustering-explained/notebook