import bnlearn as bn
import pandas as pd

ds = pd.read_csv('dataset18.csv')
print(ds)

model = bn.structure_learning.fit(ds) #Aprendizaje estructura
plot = bn.plot(model, interactive=True)

model = bn.parameter_learning.fit(model, ds, methodtype = 'bayes') #Caracterización parámetros

query1 = bn.inference.fit(model, variables=['no_carne', 'no_confort', 'no_alcohol'], evidence={'inflacion':1, 'precios_altos':1, 'escasez': 0})         #Inferencia 1
query2 = bn.inference.fit(model, variables=['guerra_ucrania'], evidence={'escasez':1, 'inflacion':0})                                                   #Inferencia 2
query3 = bn.inference.fit(model, variables=['inflacion', 'precios_altos'], evidence={'guerra_ucrania':1, 'no_carne':0, 'no_alcohol':1, 'no_confort':0}) #Inferencia 3

print('Inferencia 1:')
print(query1)

print('Inferencia 2:')
print(query2)

print('Inferencia 3:')
print(query3)
