
import random

# ==========================================
# TEMPORAL DIFFERENCE LEARNING - TD(0)
# ==========================================

GRID_SIZE = 4

START_STATE = (0, 0)
GOAL_STATE = (3, 3)
PIT_STATE = (2, 2)

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

ALPHA = 0.1       # Learning rate
GAMMA = 0.9       # Discount factor
EPISODES = 5000
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
# FIXED RANDOM POLICY
# ==========================================

def policy(state):
    return random.choice(ACTIONS)


# ==========================================
# INITIALIZE STATE VALUES
# ==========================================

V = {}

for row in range(GRID_SIZE):
    for col in range(GRID_SIZE):

        state = (row, col)

        if is_terminal(state):
            V[state] = 0.0
        else:
            V[state] = 0.0


# ==========================================
# TD(0) LEARNING
# ==========================================

for episode in range(EPISODES):

    state = START_STATE

    for step in range(MAX_STEPS):

        # Select action using fixed policy
        action = policy(state)

        # Interact with environment
        next_state, reward = take_action(
            state, action
        )

        # Terminal states have zero future value
        if is_terminal(next_state):
            next_state_value = 0.0
        else:
            next_state_value = V[next_state]

        # Calculate TD Target
        td_target = (
            reward + GAMMA * next_state_value
        )

        # Calculate TD Error
        td_error = td_target - V[state]

        # Update current state value
        V[state] += ALPHA * td_error

        # Move to next state
        state = next_state

        # Stop when terminal state is reached
        if is_terminal(state):
            break


# ==========================================
# DISPLAY LEARNED STATE VALUES
# ==========================================

print("\nTD(0) LEARNED STATE-VALUE FUNCTION")
print("----------------------------------")

for row in range(GRID_SIZE):

    for col in range(GRID_SIZE):

        state = (row, col)

        if state == GOAL_STATE:
            print(" GOAL ", end=" | ")

        elif state == PIT_STATE:
            print(" PIT  ", end=" | ")

        else:
            print(f"{V[state]:6.2f}", end=" | ")

    print()


# ==========================================
# DISPLAY IMPORTANT LEARNING DETAILS
# ==========================================

print("\nLEARNING CONFIGURATION")
print("----------------------")

print("Episodes:", EPISODES)
print("Learning Rate:", ALPHA)
print("Discount Factor:", GAMMA)

print("\nSTART STATE VALUE:", round(V[START_STATE], 4))
print("TD(0) Learning Completed!")
