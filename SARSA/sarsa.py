
import random

# ==========================================
# SARSA - ON-POLICY TD CONTROL
# ==========================================

GRID_SIZE = 4

START_STATE = (0, 0)
GOAL_STATE = (3, 3)
PIT_STATE = (2, 2)

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

ALPHA = 0.1
GAMMA = 0.9

EPSILON = 1.0
MIN_EPSILON = 0.05
EPSILON_DECAY = 0.9995

EPISODES = 10000
MAX_STEPS = 100

STEP_REWARD = -1
GOAL_REWARD = 10
PIT_REWARD = -10

random.seed(42)


# ==========================================
# ENVIRONMENT
# ==========================================

def is_terminal(state):
    return state == GOAL_STATE or state == PIT_STATE


def take_action(state, action):

    row, col = state

    if action == "UP":
        row -= 1

    elif action == "DOWN":
        row += 1

    elif action == "LEFT":
        col -= 1

    elif action == "RIGHT":
        col += 1

    row = max(0, min(GRID_SIZE - 1, row))
    col = max(0, min(GRID_SIZE - 1, col))

    next_state = (row, col)

    if next_state == GOAL_STATE:
        reward = GOAL_REWARD

    elif next_state == PIT_STATE:
        reward = PIT_REWARD

    else:
        reward = STEP_REWARD

    return next_state, reward


# ==========================================
# INITIALIZE Q-TABLE
# ==========================================

Q = {}

for row in range(GRID_SIZE):

    for col in range(GRID_SIZE):

        state = (row, col)

        for action in ACTIONS:

            Q[(state, action)] = 0.0


# ==========================================
# EPSILON-GREEDY POLICY
# ==========================================

def choose_action(state, epsilon):

    # Exploration
    if random.random() < epsilon:
        return random.choice(ACTIONS)

    # Exploitation with random tie-breaking
    q_values = [
        Q[(state, action)]
        for action in ACTIONS
    ]

    max_value = max(q_values)

    best_actions = [
        action
        for action in ACTIONS
        if Q[(state, action)] == max_value
    ]

    return random.choice(best_actions)


# ==========================================
# SARSA TRAINING
# ==========================================

epsilon = EPSILON

episode_rewards = []

for episode in range(EPISODES):

    state = START_STATE

    # Select initial action
    action = choose_action(state, epsilon)

    total_reward = 0

    for step in range(MAX_STEPS):

        # Take current action
        next_state, reward = take_action(
            state, action
        )

        total_reward += reward

        # Terminal state handling
        if is_terminal(next_state):

            td_target = reward

            td_error = (
                td_target - Q[(state, action)]
            )

            Q[(state, action)] += ALPHA * td_error

            break

        # Select next action using same policy
        next_action = choose_action(
            next_state, epsilon
        )

        # SARSA TD Target
        td_target = (
            reward
            + GAMMA * Q[(next_state, next_action)]
        )

        # SARSA TD Error
        td_error = (
            td_target - Q[(state, action)]
        )

        # Update Q-value
        Q[(state, action)] += ALPHA * td_error

        # Move to next state-action pair
        state = next_state
        action = next_action

    episode_rewards.append(total_reward)

    # Decay exploration gradually
    epsilon = max(
        MIN_EPSILON,
        epsilon * EPSILON_DECAY
    )


# ==========================================
# EXTRACT GREEDY POLICY
# ==========================================

def get_greedy_action(state):

    q_values = [
        Q[(state, action)]
        for action in ACTIONS
    ]

    max_value = max(q_values)

    best_actions = [
        action
        for action in ACTIONS
        if Q[(state, action)] == max_value
    ]

    return random.choice(best_actions)


# ==========================================
# DISPLAY LEARNED POLICY
# ==========================================

ARROWS = {
    "UP": "^",
    "DOWN": "v",
    "LEFT": "<",
    "RIGHT": ">"
}

print("\nLEARNED SARSA POLICY")
print("--------------------")

for row in range(GRID_SIZE):

    for col in range(GRID_SIZE):

        state = (row, col)

        if state == START_STATE:
            symbol = " S "

        elif state == GOAL_STATE:
            symbol = " G "

        elif state == PIT_STATE:
            symbol = " P "

        else:
            action = get_greedy_action(state)
            symbol = f" {ARROWS[action]} "

        print(symbol, end=" | ")

    print()


# ==========================================
# DISPLAY Q-VALUES
# ==========================================

print("\nLEARNED Q-VALUES")
print("----------------")

for row in range(GRID_SIZE):

    for col in range(GRID_SIZE):

        state = (row, col)

        if is_terminal(state):
            continue

        print(f"\nState {state}")

        for action in ACTIONS:

            print(
                f"{action:5s}: "
                f"{Q[(state, action)]:.4f}"
            )


# ==========================================
# TRAINING SUMMARY
# ==========================================

print("\nTRAINING SUMMARY")
print("----------------")

print("Algorithm: SARSA")
print("Episodes:", EPISODES)
print("Learning Rate:", ALPHA)
print("Discount Factor:", GAMMA)
print("Final Epsilon:", round(epsilon, 4))

print(
    "Average Reward (last 100 episodes):",
    round(sum(episode_rewards[-100:]) / 100, 2)
)

print("\nSARSA TRAINING COMPLETED!")
