import numpy as np
import tensorflow as tf
from q_network import QNetwork
from iot_environment import IoTEnvironment
from lime_trust_assessment import LimeTrustAssessment
import random
from collections import deque

# Hyperparameters for DQN and Q-Learning
NUM_DEVICES = 120  # Number of IoT devices
LOCAL_EPOCHS = 3  # Number of local training epochs per client
NUM_ITERATIONS = 1000  # Number of iterations for training
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.99
MEMORY_SIZE = 10000
BATCH_SIZE = 64
EXPLORATION_RATE_START = 1.0  # Initial exploration rate (epsilon)
EXPLORATION_RATE_END = 0.01  # Final exploration rate
EXPLORATION_DECAY = 0.995  # Decay rate for exploration
TARGET_UPDATE_FREQUENCY = 10  # Frequency to update target network

# Initialize Global Q-Network Model
state_size = NUM_DEVICES * 3  # State includes energy, trust, utilization for each device
num_actions = NUM_DEVICES

# Global Q-network model
q_network = QNetwork(num_actions=num_actions, state_size=state_size)
target_q_network = QNetwork(num_actions=num_actions, state_size=state_size)
target_q_network.set_weights(q_network.get_weights())
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

# Experience Replay Buffer
replay_memory = deque(maxlen=MEMORY_SIZE)

# Load dataset and initialize the IoT environment
env = IoTEnvironment(num_devices=NUM_DEVICES, max_tasks=20)
trust_assessment = LimeTrustAssessment()

# Training Loop
exploration_rate = EXPLORATION_RATE_START
for iteration in range(NUM_ITERATIONS):
    state = env.reset()  # Reset environment for each iteration
    done = False
    total_reward = 0

    while not done:
        # Epsilon-Greedy Action Selection
        if random.uniform(0, 1) < exploration_rate:
            action = random.randint(0, num_actions - 1)  # Random action (exploration)
        else:
            q_values = q_network(state)
            action = np.argmax(q_values)  # Action with the highest Q-value (exploitation)

        # Take action and observe environment response
        next_state, base_reward, done = env.step(action)

        # LIME-based trust assessment
        trust_score = trust_assessment.lime_trust_assessment(q_network, state)

        # Calculate energy and time factors
        energy_factor = env.device_energy[action] / env.energy_capacity  # Normalize energy usage to total capacity
        time_factor = 1.0 / (env.device_utilization[action] + 1)  # Assuming less utilized devices give faster completion

        # Calculate the reward using the updated formula
        reward = trust_score * energy_factor * time_factor * base_reward
        total_reward += reward

        # Store experience in replay memory
        replay_memory.append((state, action, reward, next_state, done))

        # Train the Q-network with a mini-batch from replay memory
        if len(replay_memory) > BATCH_SIZE:
            minibatch = random.sample(replay_memory, BATCH_SIZE)
            for s, a, r, s_next, d in minibatch:
                with tf.GradientTape() as tape:
                    q_values = q_network(s)
                    q_value = tf.reduce_sum(q_values * tf.one_hot(a, num_actions), axis=1)

                    next_q_values = target_q_network(s_next)
                    max_next_q_value = tf.reduce_max(next_q_values, axis=1)
                    target_q_value = r + (1.0 - d) * DISCOUNT_FACTOR * max_next_q_value

                    loss = tf.reduce_mean(tf.square(target_q_value - q_value))

                grads = tape.gradient(loss, q_network.trainable_variables)
                optimizer.apply_gradients(zip(grads, q_network.trainable_variables))

        # Update state
        state = next_state

    # Decay exploration rate
    exploration_rate = max(EXPLORATION_RATE_END, exploration_rate * EXPLORATION_DECAY)

    # Update target network periodically
    if iteration % TARGET_UPDATE_FREQUENCY == 0:
        target_q_network.set_weights(q_network.get_weights())

    print(f"Iteration {iteration + 1}/{NUM_ITERATIONS} completed with total reward: {total_reward}")

print("Training complete for Dynamic DQN-based Scheduler")

if __name__ == "__main__":
    print("DQN Scheduler Training initialized and executed.")
