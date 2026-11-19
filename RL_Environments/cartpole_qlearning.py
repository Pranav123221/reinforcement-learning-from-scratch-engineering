
import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt

# ==========================================
# CARTPOLE - Q-LEARNING WITH DISCRETIZATION
# ==========================================

ENV_NAME = "CartPole-v1"

EPISODES = 20000
MAX_STEPS = 500

ALPHA = 0.1
GAMMA = 0.99

EPSILON = 1.0
MIN_EPSILON = 0.05
EPSILON_DECAY = 0.9997

EVALUATION_EPISODES = 100
SEED = 42

random.seed(SEED)
np.random.seed(SEED)


# ==========================================
# CREATE ENVIRONMENT
# ==========================================

env = gym.make(ENV_NAME)

n_actions = env.action_space.n

print("Environment:", ENV_NAME)
print("Action Space:", env.action_space)
print("Observation Space:", env.observation_space)


# ==========================================
# STATE DISCRETIZATION
# ==========================================

# Observation:
# [cart_position, cart_velocity,
#  pole_angle, pole_angular_velocity]

NUM_BINS = [6, 12, 12, 12]

STATE_BOUNDS = [
    [-2.4, 2.4],
    [-3.0, 3.0],
    [-0.2095, 0.2095],
    [-3.5, 3.5]
]

BINS = [
    np.linspace(low, high, count - 1)
    for (low, high), count
    in zip(STATE_BOUNDS, NUM_BINS)
]


def discretize_state(observation):

    discrete_state = []

    for value, bins in zip(observation, BINS):

        index = np.digitize(value, bins)

        discrete_state.append(int(index))

    return tuple(discrete_state)


# ==========================================
# INITIALIZE Q-TABLE
# ==========================================

Q = np.zeros(
    (*NUM_BINS, n_actions),
    dtype=np.float32
)

print("Q-Table Shape:", Q.shape)
print("Total Discrete States:", np.prod(NUM_BINS))


# ==========================================
# EPSILON-GREEDY POLICY
# ==========================================

def choose_action(state, epsilon):

    if random.random() < epsilon:

        return env.action_space.sample()

    return int(np.argmax(Q[state]))


# ==========================================
# TRAINING
# ==========================================

epsilon = EPSILON

episode_rewards = []

for episode in range(EPISODES):

    observation, info = env.reset(
        seed=SEED + episode
    )

    state = discretize_state(observation)

    total_reward = 0

    for step in range(MAX_STEPS):

        # Select action
        action = choose_action(
            state,
            epsilon
        )

        # Interact with environment
        next_observation, reward, terminated, truncated, info = (
            env.step(action)
        )

        next_state = discretize_state(
            next_observation
        )

        total_reward += reward

        # True terminal transition:
        # Do not bootstrap after failure.
        if terminated:

            td_target = reward

        else:

            # Bootstrap for non-terminal state.
            # Also handles time-limit truncation.
            max_next_q = np.max(
                Q[next_state]
            )

            td_target = (
                reward + GAMMA * max_next_q
            )

        # TD Error
        td_error = (
            td_target - Q[state][action]
        )

        # Q-Learning update
        Q[state][action] += (
            ALPHA * td_error
        )

        state = next_state

        if terminated or truncated:
            break

    episode_rewards.append(total_reward)

    # Decay exploration
    epsilon = max(
        MIN_EPSILON,
        epsilon * EPSILON_DECAY
    )

    # Display training progress
    if (episode + 1) % 1000 == 0:

        recent_average = np.mean(
            episode_rewards[-100:]
        )

        print(
            f"Episode: {episode + 1:5d} | "
            f"Epsilon: {epsilon:.3f} | "
            f"Average Reward: {recent_average:.2f}"
        )


# ==========================================
# EVALUATION - GREEDY POLICY
# ==========================================

def evaluate_agent():

    evaluation_rewards = []

    for episode in range(EVALUATION_EPISODES):

        observation, info = env.reset(
            seed=100000 + episode
        )

        state = discretize_state(observation)

        total_reward = 0

        for step in range(MAX_STEPS):

            # Pure greedy action selection
            action = int(np.argmax(Q[state]))

            (
                next_observation,
                reward,
                terminated,
                truncated,
                info
            ) = env.step(action)

            state = discretize_state(
                next_observation
            )

            total_reward += reward

            if terminated or truncated:
                break

        evaluation_rewards.append(total_reward)

    return evaluation_rewards


evaluation_rewards = evaluate_agent()

success_rate = (
    np.mean(np.array(evaluation_rewards) >= 475) * 100
)

average_evaluation_reward = np.mean(
    evaluation_rewards
)


# ==========================================
# TRAINING METRICS
# ==========================================

window_size = 100

moving_average = np.convolve(
    episode_rewards,
    np.ones(window_size) / window_size,
    mode="valid"
)

plt.figure(figsize=(10, 5))

plt.plot(
    moving_average,
    label="100-Episode Moving Average"
)

plt.axhline(
    y=475,
    linestyle="--",
    label="475 Reward Benchmark"
)

plt.title("CartPole Q-Learning Training Performance")
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.legend()
plt.grid(True)

plt.savefig(
    "cartpole_training.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# SAVE Q-TABLE
# ==========================================

np.save(
    "cartpole_q_table.npy",
    Q
)


# ==========================================
# FINAL RESULTS
# ==========================================

print("\nTRAINING SUMMARY")
print("----------------")

print("Algorithm: Q-Learning")
print("Episodes:", EPISODES)
print("Learning Rate:", ALPHA)
print("Discount Factor:", GAMMA)
print("Final Epsilon:", round(epsilon, 4))

print(
    "Last 100 Training Episodes Average:",
    round(np.mean(episode_rewards[-100:]), 2)
)

print("\nEVALUATION RESULTS")
print("------------------")

print("Evaluation Episodes:", EVALUATION_EPISODES)

print(
    "Average Evaluation Reward:",
    round(average_evaluation_reward, 2)
)

print(
    "Success Rate (Reward >= 475):",
    f"{success_rate:.2f}%"
)

if average_evaluation_reward >= 475:
    print("Benchmark Reached!")
else:
    print("More training or tuning may be required.")

env.close()

print("\nCARTPOLE TRAINING COMPLETED!")
