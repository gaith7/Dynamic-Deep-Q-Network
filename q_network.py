import tensorflow as tf
import numpy as np

# QNetwork Definition
class QNetwork(tf.keras.Model):
    def __init__(self, num_actions, state_size):
        super(QNetwork, self).__init__()
        # State size should include energy levels, trust, and utilization for each IoT device
        self.dense1 = tf.keras.layers.Dense(512, activation='relu', input_shape=(state_size,))
        self.dense2 = tf.keras.layers.Dense(256, activation='relu')
        self.dense3 = tf.keras.layers.Dense(128, activation='relu')
        self.output_layer = tf.keras.layers.Dense(num_actions, activation='linear')

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.output_layer(x)

if __name__ == "__main__":
    num_devices = 120  # Number of IoT devices as per simulation setup
    state_size = num_devices * 3  # State includes energy, trust, utilization for each device
    q_network = QNetwork(num_actions=num_devices, state_size=state_size)
    print("QNetwork class defined for use in reinforcement learning, ready for integration.")
