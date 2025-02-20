import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Gerando dados fictícios para agrupamento
np.random.seed(0)
X = np.concatenate([
    np.random.randn(50, 2) + [2, 2],  # Cluster 1
    np.random.randn(50, 2) + [8, 8],  # Cluster 2
    np.random.randn(50, 2) + [2, 8]   # Cluster 3
])

# Aplicando K-Means para encontrar 3 clusters
kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
kmeans.fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Plotando os dados e os centróides dos clusters
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', edgecolors='k', label="Pontos de dados")
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label="Centroides")
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Exemplo de K-Means Clustering')
plt.legend()
plt.show()
