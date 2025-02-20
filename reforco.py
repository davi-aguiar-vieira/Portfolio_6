import numpy as np
import matplotlib.pyplot as plt

# Configuração do ambiente (grid 5x5)
size = 5
num_states = size * size
num_actions = 4  # [0: cima, 1: baixo, 2: esquerda, 3: direita]
goal_state = num_states - 1  # Última célula do grid como objetivo

# Inicialização da Q-table
q_table = np.zeros((num_states, num_actions))
learning_rate = 0.1
discount_factor = 0.9
exploration_rate = 1.0
exploration_decay = 0.995
num_episodes = 500

# Função para converter estado em coordenadas (linha, coluna)
def state_to_coord(state):
    return state // size, state % size

# Função para converter coordenadas em estado
def coord_to_state(row, col):
    return row * size + col

# Função para tomar uma ação no ambiente e obter o próximo estado e recompensa
def step(state, action):
    row, col = state_to_coord(state)
    if action == 0 and row > 0: row -= 1  # Cima
    elif action == 1 and row < size - 1: row += 1  # Baixo
    elif action == 2 and col > 0: col -= 1  # Esquerda
    elif action == 3 and col < size - 1: col += 1  # Direita
    next_state = coord_to_state(row, col)
    reward = 1 if next_state == goal_state else -0.01  # Recompensa apenas na meta
    return next_state, reward

# Treinamento do agente usando Q-Learning
rewards_per_episode = []
for episode in range(num_episodes):
    state = 0  # Estado inicial (canto superior esquerdo)
    total_reward = 0

    while state != goal_state:
        # Escolher ação com exploração/explicação
        if np.random.rand() < exploration_rate:
            action = np.random.choice(num_actions)  # Escolha aleatória (exploração)
        else:
            action = np.argmax(q_table[state])  # Melhor ação conhecida (exploração)

        next_state, reward = step(state, action)
        total_reward += reward

        # Atualizar Q-table
        best_next_action = np.max(q_table[next_state])
        q_table[state, action] += learning_rate * (reward + discount_factor * best_next_action - q_table[state, action])

        state = next_state

    rewards_per_episode.append(total_reward)
    exploration_rate *= exploration_decay  # Decaimento da exploração

# Plotando a evolução da recompensa por episódio
plt.figure(figsize=(8, 5))
plt.plot(rewards_per_episode, label="Recompensa por Episódio")
plt.xlabel("Episódios")
plt.ylabel("Recompensa Acumulada")
plt.title("Aprendizado por Reforço com Q-Learning")
plt.legend()
plt.show()