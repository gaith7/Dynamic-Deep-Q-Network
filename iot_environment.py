import numpy as np

# IoT Environment
class IoTEnvironment:
    def __init__(self, num_devices, max_tasks, energy_capacity=5000):
        self.num_devices = num_devices
        self.max_tasks = max_tasks
        self.energy_capacity = energy_capacity
        self.device_energy = np.random.randint(energy_capacity // 2, energy_capacity, size=num_devices)
        self.device_trust = np.random.uniform(0.5, 1.0, size=num_devices)
        self.device_utilization = np.zeros(num_devices)  # Tracks how utilized each device is

    def reset(self):
        # Reset the environment to initial states for a new iteration
        self.device_energy = np.random.randint(self.energy_capacity // 2, self.energy_capacity, size=self.num_devices)
        self.device_trust = np.random.uniform(0.5, 1.0, size=self.num_devices)
        self.device_utilization = np.zeros(self.num_devices)
        return self._get_state()

    def _get_state(self):
        # State representation includes energy, trust, and utilization of all devices
        state = np.concatenate((self.device_energy, self.device_trust, self.device_utilization))
        return state.reshape(1, -1)

    def step(self, action):
        # Simulate task assignment to the selected device
        energy_consumption = np.random.randint(50, 100)  # Random energy consumption for the task
        reward = 0
        done = False

        # Enhanced reward model: Reward is influenced by trust, energy remaining, and utilization
        if self.device_energy[action] >= energy_consumption:
            # If the device has enough energy, assign the task
            self.device_energy[action] -= energy_consumption
            self.device_utilization[action] += 1
            reward = self.device_trust[action] * (self.device_energy[action] / self.energy_capacity)  # Trust + energy factor
        else:
            # Penalize if the selected device cannot handle the task
            reward = -1

        # Update state
        next_state = self._get_state()

        # Enhanced done criteria to reflect a more realistic termination
        if np.all(self.device_utilization >= self.max_tasks) or np.all(self.device_energy < 50):
            done = True

        return next_state, reward, done

if __name__ == "__main__":
    env = IoTEnvironment(num_devices=120, max_tasks=20)  # Parameters updated as per simulation setup
    print("IoTEnvironment class defined and refined for use in reinforcement learning.")
