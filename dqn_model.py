import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class DQN():
    def __init__(self, input_shape, num_actions):
        self.model = tf.keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=input_shape),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_actions, activation='linear')
            ])
        
        self.target_model = tf.keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=input_shape),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_actions, activation='linear')
            ])
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        
    def train_dqn_step(self, replay_buffer, batch_size, gamma):
        # Pobierz próbkę z replay buffer
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        # Przewiduj wartości Q dla aktualnych i następnych stanów
        q_values = self.model(states).numpy()
        next_q_values = self.target_model(next_states).numpy()

        # Oblicz wartości docelowe Q
        target_q_values = q_values.copy()
        for i in range(batch_size):
            target_q_values[i, actions[i]] = rewards[i] + gamma * np.max(next_q_values[i]) * (1 - dones[i])

        # Przeprowadź trening
        with tf.GradientTape() as tape:
            predictions = self.model(states)
            loss = tf.keras.losses.mean_squared_error(target_q_values, predictions)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
