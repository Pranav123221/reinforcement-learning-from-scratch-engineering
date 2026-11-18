
import numpy as np

# ==========================================
# ENVIRONMENT CONFIGURATION
# ==========================================

GRID_SIZE = 4

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

MOVES = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1)
}

GOAL = (3, 3)
PIT = (1, 3)

TERMINAL_STATES = {GOAL, PIT}

GAMMA = 0.9
THETA = 1e-8
MAX_ITERATIONS = 100_000


# ==========================================
# ENVIRONMENT TRANSITION MODEL
# ==========================================

def get_transition(state, action):

    if state in TERMINAL_STATES:
        return state, 0, 1.0

    row, col = state

    dr, dc = MOVES[action]

    next_row = row + dr
    next_col = col + dc

    # Boundary handling
    if not (
        0 <= next_row < GRID_SIZE
        and 0 <= next_col < GRID_SIZE
    ):
        next_row, next_col = row, col

    next_state = (next_row, next_col)

    # Reward model
    if next_state == GOAL:
        reward = 10
    elif next_state == PIT:
        reward = -10
    else:
        reward = -1

    return next_state, reward, 1.0


# ==========================================
# STATE INITIALIZATION
# ==========================================

def get_states():

    return [
        (row, col)
        for row in range(GRID_SIZE)
        for col in range(GRID_SIZE)
    ]


def initialize_values():

    return {
        state: 0.0
        for state in get_states()
    }


# ==========================================
# BELLMAN OPTIMALITY UPDATE
# ==========================================

def calculate_action_value(state, action, V):

    next_state, reward, probability = (
        get_transition(state, action)
    )

    return probability * (
        reward + GAMMA * V[next_state]
    )


def value_iteration():

    V = initialize_values()

    for iteration in range(1, MAX_ITERATIONS + 1):

        new_V = V.copy()

        delta = 0.0

        for state in get_states():

            # Terminal states have no future value.
            if state in TERMINAL_STATES:
                new_V[state] = 0.0
                continue

            action_values = []

            for action in ACTIONS:

                value = calculate_action_value(
                    state, action, V
                )

                action_values.append(value)

            # Bellman optimality update
            new_V[state] = max(action_values)

            delta = max(
                delta,
                abs(new_V[state] - V[state])
            )

        V = new_V

        if delta < THETA:
            break

    else:
        raise RuntimeError(
            "Value iteration did not converge."
        )

    return V, iteration


# ==========================================
# EXTRACT OPTIMAL POLICY
# ==========================================

def extract_policy(V):

    policy = {}

    for state in get_states():

        if state in TERMINAL_STATES:
            continue

        action_values = {}

        for action in ACTIONS:

            action_values[action] = (
                calculate_action_value(state, action, V)
            )

        best_action = max(
            action_values,
            key=action_values.get
        )

        policy[state] = best_action

    return policy


# ==========================================
# DISPLAY POLICY
# ==========================================

def display_policy(policy):

    symbols = {
        "UP": " ↑ ",
        "DOWN": " ↓ ",
        "LEFT": " ← ",
        "RIGHT": " → "
    }

    print("\nOPTIMAL POLICY")
    print("-" * 40)

    for row in range(GRID_SIZE):

        line = []

        for col in range(GRID_SIZE):

            state = (row, col)

            if state == GOAL:
                line.append(" G ")
            elif state == PIT:
                line.append(" P ")
            else:
                line.append(symbols[policy[state]])

        print(" | ".join(line))


# ==========================================
# DISPLAY VALUE FUNCTION
# ==========================================

def display_values(V):

    print("\nOPTIMAL STATE VALUE FUNCTION")
    print("-" * 50)

    for row in range(GRID_SIZE):

        line = []

        for col in range(GRID_SIZE):

            state = (row, col)

            if state == GOAL:
                line.append(" GOAL ")
            elif state == PIT:
                line.append("  PIT  ")
            else:
                line.append(f"{V[state]:8.2f}")

        print(" | ".join(line))


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    print("=" * 55)
    print("REINFORCEMENT LEARNING: VALUE ITERATION")
    print("=" * 55)

    V, iterations = value_iteration()

    policy = extract_policy(V)

    print(f"\nDiscount Factor: {GAMMA}")
    print(f"Convergence Threshold: {THETA}")
    print(f"Total Iterations: {iterations}")

    display_values(V)

    display_policy(policy)

    print("\nFINAL RESULT")
    print("-" * 40)
    print("Value Iteration Converged.")
    print("Optimal state values calculated.")
    print("Greedy policy extracted successfully.")
