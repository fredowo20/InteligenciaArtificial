import numpy as np
import matplotlib.pyplot as plt

def random_walk(inicio, iteraciones, id):
    i = inicio
    counter = [0,0,0,0,0,0,0,0,0,0]
    path = [inicio] #Desde qué grupo inicia el random walk y luego vector con valores de random walk
    len_array = np.arange(len(group_prob))
    rango=iteraciones #Largo del random walk
    for _ in range(rango):
        i = np.random.choice(9, p = group_prob[i])
        counter[i]+=1
        path.append(i)
    #print(path)

    prob = [] #Probabilidades obtenidas a partir del random walk

    for j in range(9):
        division=counter[j]/rango
        prob.append(division)

    print('Random walk '+str(iteraciones)+' iteraciones: ')
    print(prob)

    plt.title('Probabilidades Random Walk '+str(id))
    plt.plot(groups, prob)

    plt.show()

groups = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9']
group_prob = np.array([
    [0.25, 0.06, 0.08, 0.15, 0.04, 0.02, 0.15, 0.15, 0.10], 
    [0.15, 0.15, 0.10, 0.22, 0.01, 0.02, 0.15, 0.10, 0.10], 
    [0.12, 0.00, 0.05, 0.24, 0.14, 0.04, 0.27, 0.07, 0.07], 
    [0.05, 0.13, 0.05, 0.30, 0.10, 0.10, 0.22, 0.05, 0.00], 
    [0.18, 0.20, 0.07, 0.20, 0.15, 0.05, 0.05, 0.05, 0.05],
    [0.20, 0.10, 0.20, 0.05, 0.05, 0.10, 0.02, 0.15, 0.13],
    [0.01, 0.05, 0.15, 0.14, 0.17, 0.10, 0.12, 0.10, 0.16],
    [0.17, 0.15, 0.07, 0.07, 0.15, 0.10, 0.12, 0.09, 0.08],
    [0.13, 0.11, 0.13, 0.03, 0.20, 0.20, 0.04, 0.15, 0.01]])

random_walk(0,100,1)   #Random Walk 1
random_walk(4,1000,2)  #Random Walk 2
random_walk(8,10000,3) #Random Walk 3

### Distribución Estacionaria ###

evals, evecs = np.linalg.eig(group_prob.T)
evec1 = evecs[:,0]

stationary = (evec1 / evec1.sum()).real

print('Distribución estacionaria: ')
print(stationary)

plt.title('Distribución Estacionaria')
plt.plot(groups, stationary, color="red")

plt.show()
