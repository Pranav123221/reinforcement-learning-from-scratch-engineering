
import numpy as np

# ------------------------------------------
# Environment Configuration
# ------------------------------------------

GRID_SIZE = 4
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

ACTION_MOVES = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1)
}

GOAL = (3, 3)
PIT = (1, 3)

GAMMA = 0.9
THETA = 1e-8

TERMINAL_STATES = {GOAL, PIT}


# ------------------------------------------
# Environment Transition Model
# ------------------------------------------

def get_transition(state, action):
    """
    Deterministic transition model.

    Returns:
        next_state, reward, probability
    """

    if state in TERMINAL_STATES:
        return state, 0, 1.0

    row, col = state
    dr, dc = ACTION_MOVES[action]

    next_row = row + dr
    next_col = col + dc

    # Prevent movement outside the grid.
    if not (
        0 <= next_row < GRID_SIZE
        and 0 <= next_col < GRID_SIZE
    ):
        next_row, next_col = row, col

    next_state = (next_row, next_col)

    # Reward is based on the destination state.
    if next_state == GOAL:
        reward = 10
    elif next_state == PIT:
        reward = -10
    else:
        reward = -1

    return next_state, reward, 1.0


# ------------------------------------------
# Stochastic Policy
# ------------------------------------------

def get_policy(state):
    """
    Uniform random policy:
    Each action has probability 0.25.
    """

    if state in TERMINAL_STATES:
        return {}

    probability = 1.0 / len(ACTIONS)

    return {
        action: probability
        for action in ACTIONS
    }


# ------------------------------------------
# Bellman Expectation:
# State Value Function V(s)
# ------------------------------------------

def evaluate_policy():
    """
    Iterative policy evaluation using
    the Bellman expectation equation.
    """

    states = [
        (row, col)
        for row in range(GRID_SIZE)
        for col in range(GRID_SIZE)
    ]

    V = {state: 0.0 for state in states}

    for iteration in range(1, 100_001):

        # Synchronous update: every new value
        # is calculated from the previous V.
        new_V = V.copy()

        for state in states:

            if state in TERMINAL_STATES:
                new_V[state] = 0.0
                continue

            value = 0.0
            policy = get_policy(state)

            for action, action_probability in policy.items():

                next_state, reward, transition_probability = (
                    get_transition(state, action)
                )

                expected_return = (
                    reward + GAMMA * V[next_state]
                )

                value += (
                    action_probability
                    * transition_probability
                    * expected_return
                )

            new_V[state] = value

        # Check convergence.
        delta = max(
            abs(new_V[state] - V[state])
            for state in states
        )

        V = new_V

        if delta < THETA:
            break

    return V, iteration


# ------------------------------------------
# Bellman Expectation:
# Action Value Function Q(s, a)
# ------------------------------------------

def calculate_q_values(V):
    """
    Calculate Q(s,a) using the converged
    state values V(s).
    """

    Q = {}

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):

            state = (row, col)

            if state in TERMINAL_STATES:
                continue

            Q[state] = {}

            for action in ACTIONS:

                next_state, reward, probability = (
                    get_transition(state, action)
                )

                Q[state][action] = probability * (
                    reward + GAMMA * V[next_state]
                )

    return Q


# ------------------------------------------
# Verify V(s) = Sum pi(a|s) Q(s,a)
# ------------------------------------------

def verify_value_relationship(V, Q):

    max_error = 0.0

    for state, action_values in Q.items():

        calculated_value = sum(
            get_policy(state)[action] * action_values[action]
            for action in ACTIONS
        )

        error = abs(V[state] - calculated_value)
        max_error = max(max_error, error)

    return max_error


# ------------------------------------------
# Display Results
# ------------------------------------------

def display_value_function(V):

    print("\nSTATE VALUE FUNCTION V(s)")
    print("-" * 45)

    for row in range(GRID_SIZE):

        values = []

        for col in range(GRID_SIZE):

            state = (row, col)

            if state == GOAL:
                values.append(" GOAL ")
            elif state == PIT:
                values.append("  PIT  ")
            else:
                values.append(f"{V[state]:7.2f}")

        print(" | ".join(values))


def display_q_values(Q):

    print("\nACTION VALUE FUNCTION Q(s,a)")
    print("-" * 55)

    for state, action_values in Q.items():

        print(f"\nState: {state}")

        for action, value in action_values.items():
            print(f"  {action:5s}: {value:8.4f}")


# ------------------------------------------
# Main Execution
# ------------------------------------------

if __name__ == "__main__":

    V, iterations = evaluate_policy()

    Q = calculate_q_values(V)

    max_error = verify_value_relationship(V, Q)

    print("=" * 55)
    print("BELLMAN EXPECTATION EQUATION")
    print("=" * 55)

    print(f"Discount Factor (gamma): {GAMMA}")
    print(f"Convergence Threshold: {THETA}")
    print(f"Policy: Uniform Random Policy")
    print(f"Converged in: {iterations} iterations")

    display_value_function(V)
    display_q_values(Q)

    print("\nVALUE FUNCTION VERIFICATION")
    print("-" * 45)

    print("V(s) = Sum pi(a|s) * Q(s,a)")
    print(f"Maximum numerical error: {max_error:.12f}")

    if max_error < 1e-7:
        print("Verification: PASSED")
    else:
        print("Verification: FAILED")
