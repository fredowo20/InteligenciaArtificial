import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

datos = pd.read_csv('animeFinal.csv',index_col=0)
df = pd.DataFrame(datos)

X = df[['Action','Drama','Supernatural','Suspense','Adventure','Fantasy','Comedy','Romance','Horror','Sci-Fi','Ecchi','Mystery','Sports','Award Winning','Avant Garde','Slice of Life','Gourmet','Boys Love','Girls Love','Hentai','Erotica']]
Y = df['Score']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.4) # Se define 40% de prueba y 60% de entrenamiento

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error

error = 0

#Error MAE

ln = LinearRegression()
ln.fit(X_train, Y_train)
pred_i = ln.predict(X_test)
mae = mean_absolute_error(Y_test, pred_i) # Métrica MAE
error = mae

print("MAE: Error de "+str(error))

# #Error MSE

# ln = LinearRegression()
# ln.fit(X_train, Y_train)
# pred_i = ln.predict(X_test)
# mse = mean_squared_error(Y_test, pred_i)
# error = mse

# print("MSE: Error de "+str(error))

# #Error MSLE

# ln = LinearRegression()
# ln.fit(X_train, Y_train)
# pred_i = ln.predict(X_test)
# msle = mean_squared_log_error(Y_test, pred_i)
# error = msle

# print("MSLE: Error de "+str(error))

#Predicción

ln = LinearRegression()
ln.fit(X_train.values,Y_train)

prediction = ln.predict([[1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]) #Predicción
print("Puntaje predicción: "+str(prediction))
