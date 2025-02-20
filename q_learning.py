import numpy as np
import matplotlib.pyplot as plt

# Configuração do ambiente (Exemplo: 5 estados)
num_states = 5
num_actions = 2  # Exemplo: Esquerda (0) e Direita (1)
Q_table = np.zeros((num_states, num_actions))

# Parâmetros do Q-Learning
alpha = 0.1   # Taxa de aprendizado
gamma = 0.9   # Fator de desconto
epsilon = 0.1 # Probabilidade de explorar

# Simulação do aprendizado
recompensas = []
for episode in range(100):
    estado = np.random.randint(0, num_states)  # Começa em um estado aleatório
    total_reward = 0

    for _ in range(10):  # Número de passos por episódio
        if np.random.rand() < epsilon:
            acao = np.random.choice(num_actions)  # Escolha aleatória (exploração)
        else:
            acao = np.argmax(Q_table[estado, :])  # Melhor ação conhecida (exploração)

        # Definição da recompensa
        if estado == num_states - 1 and acao == 1:  # Último estado e ação direita -> recompensa alta
            recompensa = 10
        else:
            recompensa = -1  # Penalidade por cada movimento errado

        # Atualização da tabela Q
        novo_estado = min(num_states - 1, max(0, estado + (1 if acao == 1 else -1)))
        Q_table[estado, acao] = Q_table[estado, acao] + alpha * (recompensa + gamma * np.max(Q_table[novo_estado, :]) - Q_table[estado, acao])

        estado = novo_estado
        total_reward += recompensa

    recompensas.append(total_reward)

# Visualização do aprendizado
plt.plot(recompensas)
plt.xlabel("Episódios")
plt.ylabel("Recompensa Acumulada")
plt.title("Aprendizado com Q-Learning")
plt.show()