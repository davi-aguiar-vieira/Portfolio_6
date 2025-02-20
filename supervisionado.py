import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Gerando dados fictícios
np.random.seed(0)
x = 10 * np.random.rand(100, 1)
y = 2.5 * x + np.random.randn(100, 1) * 2  # Relação linear com ruído

# Dividindo os dados em treino e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Criando e treinando o modelo de Regressão Linear
modelo = LinearRegression()
modelo.fit(x_train, y_train)

# Fazendo previsões
y_pred = modelo.predict(x_test)

# Plotando os resultados
plt.scatter(x_test, y_test, color='blue', label='Dados reais')
plt.plot(x_test, y_pred, color='red', linewidth=2, label='Regressão Linear')
plt.xlabel('Variável de entrada (x)')
plt.ylabel('Variável de saída (y)')
plt.legend()
plt.title('Exemplo de Regressão Linear')
plt.show()