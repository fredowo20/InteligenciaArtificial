import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

datos = pd.read_csv('animeFinal.csv',index_col=0)
df = pd.DataFrame(datos)

X = df[['Action','Drama','Supernatural','Suspense','Adventure','Fantasy','Comedy','Romance','Horror','Sci-Fi','Ecchi','Mystery','Sports','Award Winning','Avant Garde','Slice of Life','Gourmet','Boys Love','Girls Love','Hentai','Erotica']]
Y = df['Score']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.4) # Se define 40% de prueba y 60% de entrenamiento

neighbors = np.arange(1,21)

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error
import matplotlib.pyplot as plt

#Error MAE
error = []

for i in range(1, 40): # Se prueba con valores de K entre 1 y 40
    knn = KNeighborsRegressor(n_neighbors=i)
    knn.fit(X_train, Y_train)
    pred_i = knn.predict(X_test)
    mae = mean_absolute_error(Y_test, pred_i) # Métrica MAE
    error.append(mae)

print("MAE: Error mínimo "+str(min(error))+" en k="+str(np.array(error).argmin()))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red',
         linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
         
plt.title('K Value MAE')
plt.xlabel('K Value')
plt.ylabel('Mean Absolute Error')

plt.show() # Gráfica MAE para cada K

#Error MSE
error = []

for i in range(1, 40): # Se prueba con valores de K entre 1 y 40
    knn = KNeighborsRegressor(n_neighbors=i)
    knn.fit(X_train, Y_train)
    pred_i = knn.predict(X_test)
    mse = mean_squared_error(Y_test, pred_i) # Métrica MSE
    error.append(mse)

print("MSE: Error mínimo "+str(min(error))+" en k="+str(np.array(error).argmin()))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red',
         linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
         
plt.title('K Value MSE')
plt.xlabel('K Value')
plt.ylabel('Mean Squared Error')

plt.show() # Gráfica MSE para cada K

#Error MSLE
error = []

for i in range(1, 40): # Se prueba con valores de K entre 1 y 40
    knn = KNeighborsRegressor(n_neighbors=i)
    knn.fit(X_train, Y_train)
    pred_i = knn.predict(X_test)
    msle = mean_squared_log_error(Y_test, pred_i) # Métrica MSLE
    error.append(msle)

print("MSLE: Error mínimo "+str(min(error))+" en k="+str(np.array(error).argmin()))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red',
         linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
         
plt.title('K Value MSLE')
plt.xlabel('K Value')
plt.ylabel('Mean Squared Log Error')

plt.show() # Gráfica MSLE para cada K

#Predicción KNN

knn = KNeighborsRegressor(n_neighbors=np.array(error).argmin()) # Valor de K según error mínimo conseguido
knn.fit(X_train.values,Y_train)

prediction = knn.predict([[1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]) # Predicción
print("Puntaje predicción: "+str(prediction))
