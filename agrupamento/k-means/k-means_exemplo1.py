import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

x=[20,  27,  21,  37,  46, 53, 55,  47,  52,  32,  39,  41,  39,  48,  48]  
y=[1000,1200,2900,1850,900,950,2000,2100,3000,5900,4100,5100,7000,5000,6500]  
plt.scatter(x,y)

base = []
for i in range(len(x)):
 storage = np.array([x[i],y[i]])
 base.append(storage)
base = np.asarray(base) 
del i, storage

scaler = StandardScaler()
base = scaler.fit_transform(base)

kmeans = KMeans(n_clusters = 3)
kmeans.fit(base)

centroides = kmeans.cluster_centers_
rotulos = kmeans.labels_
base_final = pd.concat([pd.DataFrame(base), pd.DataFrame(rotulos)], axis = 1)

# Ao Commitar as cores irão ficar em pontos 0,0,0 - Rodar somente as linhas abaixo novamente.
cores = ["g.", "r.", "b."]
for i in range(len(x)):
    plt.plot(base[i][0], base[i][1], cores[rotulos[i]], markersize = 15)
plt.scatter(centroides[:,0], centroides[:,1], marker = "x")
