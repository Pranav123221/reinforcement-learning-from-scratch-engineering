
import numpy as np

# ==========================================
# 1. ENVIRONMENT CONFIGURATION
# ==========================================

GRID_SIZE = 4

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

ACTION_MOVES = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1)
}

ACTION_SYMBOLS = {
    "UP": "↑",
    "DOWN": "↓",
    "LEFT": "←",
    "RIGHT": "→"
}

START = (0, 0)
GOAL = (3, 3)
PIT = (1, 3)

TERMINAL_STATES = {GOAL, PIT}

GAMMA = 0.9
THETA = 1e-8
MAX_EVALUATION_STEPS = 100_000


# ==========================================
# 2. ENVIRONMENT TRANSITION MODEL
# ==========================================

def get_transition(state, action):

    if state in TERMINAL_STATES:
        return state, 0.0, 1.0

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

    if next_state == GOAL:
        reward = 10.0
    elif next_state == PIT:
        reward = -10.0
    else:
        reward = -1.0

    return next_state, reward, 1.0


# ==========================================
# 3. INITIALIZE POLICY
# ==========================================

def initialize_policy():

    policy = {}

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):

            state = (row, col)

            if state in TERMINAL_STATES:
                continue

            # Initially choose RIGHT in every state.
            policy[state] = "RIGHT"

    return policy


# ==========================================
# 4. POLICY EVALUATION
# ==========================================

def policy_evaluation(policy, V):

    """
    Iteratively estimate Vπ(s) for a fixed policy.

    Vπ(s) = r + gamma * Vπ(s')
    for this deterministic environment.
    """

    for iteration in range(1, MAX_EVALUATION_STEPS + 1):

        new_V = V.copy()

        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):

                state = (row, col)

                if state in TERMINAL_STATES:
                    new_V[state] = 0.0
                    continue

                action = policy[state]

                next_state, reward, probability = (
                    get_transition(state, action)
                )

                new_V[state] = probability * (
                    reward + GAMMA * V[next_state]
                )

        delta = max(
            abs(new_V[state] - V[state])
            for state in V
        )

        V = new_V

        if delta < THETA:
            return V, iteration

    raise RuntimeError(
        "Policy evaluation did not converge."
    )


# ==========================================
# 5. ACTION VALUE CALCULATION
# ==========================================

def calculate_action_value(state, action, V):

    """
    Calculate Qπ(s,a) using the current
    estimated state-value function.
    """

    next_state, reward, probability = (
        get_transition(state, action)
    )

    return probability * (
        reward + GAMMA * V[next_state]
    )


# ==========================================
# 6. POLICY IMPROVEMENT
# ==========================================

def policy_improvement(policy, V):

    """
    Greedily improve the current policy.

    If multiple actions have equal values,
    the first action in ACTIONS is selected.
    """

    policy_stable = True

    new_policy = policy.copy()

    for state in policy:

        old_action = policy[state]

        action_values = {
            action: calculate_action_value(
                state, action, V
            )
            for action in ACTIONS
        }

        best_action = max(
            ACTIONS,
            key=lambda action: action_values[action]
        )

        new_policy[state] = best_action

        if best_action != old_action:
            policy_stable = False

    return new_policy, policy_stable


# ==========================================
# 7. COMPLETE POLICY ITERATION
# ==========================================

def policy_iteration():

    policy = initialize_policy()

    V = {
        (row, col): 0.0
        for row in range(GRID_SIZE)
        for col in range(GRID_SIZE)
    }

    total_evaluation_steps = 0

    for policy_iteration_number in range(1, 101):

        # Step 1: Evaluate current policy.
        V, evaluation_steps = policy_evaluation(
            policy, V
        )

        total_evaluation_steps += evaluation_steps

        # Step 2: Improve current policy.
        new_policy, policy_stable = policy_improvement(
            policy, V
        )

        print(
            f"Policy Iteration {policy_iteration_number}: "
            f"Evaluation Steps = {evaluation_steps}, "
            f"Policy Stable = {policy_stable}"
        )

        policy = new_policy

        if policy_stable:
            return (
                policy,
                V,
                policy_iteration_number,
                total_evaluation_steps
            )

    raise RuntimeError(
        "Policy iteration did not converge."
    )


# ==========================================
# 8. DISPLAY VALUE FUNCTION
# ==========================================

def display_values(V):

    print("\nOPTIMAL STATE VALUES")
    print("-" * 42)

    for row in range(GRID_SIZE):

        values = []

        for col in range(GRID_SIZE):

            state = (row, col)

            if state == GOAL:
                values.append(" GOAL ")
            elif state == PIT:
                values.append("  PIT ")
            else:
                values.append(f"{V[state]:7.2f}")

        print(" | ".join(values))


# ==========================================
# 9. DISPLAY POLICY
# ==========================================

def display_policy(policy):

    print("\nIMPROVED POLICY")
    print("-" * 42)

    for row in range(GRID_SIZE):

        symbols = []

        for col in range(GRID_SIZE):

            state = (row, col)

            if state == GOAL:
                symbols.append(" G ")
            elif state == PIT:
                symbols.append(" P ")
            else:
                action = policy[state]
                symbols.append(
                    f" {ACTION_SYMBOLS[action]} "
                )

        print(" | ".join(symbols))


# ==========================================
# 10. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    print("=" * 50)
    print("DYNAMIC PROGRAMMING: POLICY ITERATION")
    print("=" * 50)

    print(f"Grid Size: {GRID_SIZE}x{GRID_SIZE}")
    print(f"Discount Factor: {GAMMA}")
    print(f"Goal: {GOAL}")
    print(f"Pit: {PIT}")

    (
        optimal_policy,
        optimal_values,
        iterations,
        evaluation_steps
    ) = policy_iteration()

    display_values(optimal_values)
    display_policy(optimal_policy)

    print("\nTRAINING SUMMARY")
    print("-" * 42)

    print(f"Policy Iterations: {iterations}")
    print(f"Total Evaluation Sweeps: {evaluation_steps}")
    print("Policy Converged: True")

    print("\nDynamic Programming completed successfully.")
