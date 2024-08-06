import connection as cn
import numpy as np


# Initialize connection
connection = cn.connect(2037)

# Load Q table or initialize if file not found
try:
    Q = np.loadtxt('resultado.txt')
except IOError:
    Q = np.zeros([24 * 4, 3])

# Parameters
alpha = 0.7  # Learning rate
gamma = 0.95  # Discount factor
epsilon = 0.1  # Exploration rate

# Possible actions
actions = ["left", "right", "jump"]

# Get initial state and reward
state, reward = cn.get_state_reward(connection, "")

# Extract initial platform and direction
platform = int(state[2:7], 2)
direction = int(state[-2:], 2)

divider = "=" * 30
print(divider)
print(f"\nState: {state}\nPlatform: {platform}\nDirection: {direction}\nReward: {reward}\n")

# Main loop
while True:
    print(divider + "\n")

    # Convert state to integer
    state_index = int(state, 2)

    # Choose action based on epsilon-greedy policy
    if np.random.uniform(0, 1) < epsilon:
        action = np.random.choice([0, 1, 2])
    else:
        action = np.argmax(Q[state_index])
    
    # Execute action and get new state and reward
    new_state, reward = cn.get_state_reward(connection, actions[action])
    new_state_int = int(new_state, 2)
    
    # Extract new platform and direction
    platform = int(new_state[2:7], 2)
    direction = int(new_state[-2:], 2)
    
    print(f"Action: {actions[action]}\nNew state: {new_state}\nPlatform: {platform}\nDirection: {direction}\nReward: {reward}\n")
    
    # Update Q table
    best_future_q = np.max(Q[new_state_int])
    Q[state_index, action] = (1 - alpha) * Q[state_index, action] + alpha * (reward + gamma * best_future_q)
    
    # Update state
    state = new_state

    # Save updated Q table
    np.savetxt('resultado.txt', Q)