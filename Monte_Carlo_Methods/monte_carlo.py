
import numpy as np
from collections import defaultdict

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
NUM_EPISODES = 20_000
MAX_STEPS = 100

SEED = 42


# ==========================================
# GRIDWORLD ENVIRONMENT
# ==========================================

class GridWorld:

    def __init__(self):

        self.start_state = (0, 0)
        self.state = self.start_state

    def reset(self):

        self.state = self.start_state
        return self.state

    def step(self, action):

        if self.state in TERMINAL_STATES:
            raise RuntimeError("Episode has terminated.")

        row, col = self.state

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

        if next_state == GOAL:
            reward = 10
        elif next_state == PIT:
            reward = -10
        else:
            reward = -1

        terminated = next_state in TERMINAL_STATES

        self.state = next_state

        return next_state, reward, terminated


# ==========================================
# FIXED STOCHASTIC POLICY
# ==========================================

def policy(state, rng):

    """
    Uniform random policy.

    Every action has probability 0.25.
    """

    return rng.choice(ACTIONS)


# ==========================================
# GENERATE ONE EPISODE
# ==========================================

def generate_episode(env, rng):

    episode = []

    state = env.reset()

    for _ in range(MAX_STEPS):

        action = policy(state, rng)

        next_state, reward, terminated = env.step(action)

        # Store the state and reward received
        # after taking the action.
        episode.append((state, reward))

        state = next_state

        if terminated:
            break

    return episode


# ==========================================
# CALCULATE DISCOUNTED RETURNS
# ==========================================

def calculate_returns(episode):

    returns = []

    G = 0.0

    for state, reward in reversed(episode):

        G = reward + GAMMA * G

        returns.append((state, G))

    returns.reverse()

    return returns


# ==========================================
# MONTE CARLO PREDICTION
# ==========================================

def monte_carlo_prediction(method="first_visit"):

    if method not in {"first_visit", "every_visit"}:
        raise ValueError(
            "Method must be first_visit or every_visit."
        )

    rng = np.random.default_rng(SEED)

    env = GridWorld()

    # Store total returns and visit counts.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)

    V = {}

    for episode_number in range(NUM_EPISODES):

        episode = generate_episode(env, rng)

        episode_returns = calculate_returns(episode)

        visited_states = set()

        for state, G in episode_returns:

            if method == "first_visit":

                if state in visited_states:
                    continue

                visited_states.add(state)

            returns_sum[state] += G
            returns_count[state] += 1

            V[state] = (
                returns_sum[state] / returns_count[state]
            )

    return V, returns_count


# ==========================================
# DISPLAY VALUE FUNCTION
# ==========================================

def display_values(V):

    print("\nMONTE CARLO STATE VALUE FUNCTION")
    print("-" * 50)

    for row in range(GRID_SIZE):

        line = []

        for col in range(GRID_SIZE):

            state = (row, col)

            if state == GOAL:
                line.append(" GOAL ")
            elif state == PIT:
                line.append("  PIT  ")
            elif state in V:
                line.append(f"{V[state]:8.2f}")
            else:
                line.append("   N/A  ")

        print(" | ".join(line))


# ==========================================
# DISPLAY VISIT COUNTS
# ==========================================

def display_counts(counts):

    print("\nSTATE VISIT COUNTS")
    print("-" * 50)

    for row in range(GRID_SIZE):

        line = []

        for col in range(GRID_SIZE):

            state = (row, col)

            if state in TERMINAL_STATES:
                line.append("  --- ")
            else:
                line.append(f"{counts[state]:6d}")

        print(" | ".join(line))


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    print("=" * 55)
    print("MONTE CARLO PREDICTION")
    print("=" * 55)

    print("\nChoose Monte Carlo method:")
    print("1. First-Visit Monte Carlo")
    print("2. Every-Visit Monte Carlo")

    choice = input("\nEnter choice (1/2): ").strip()

    if choice == "1":
        method = "first_visit"
    elif choice == "2":
        method = "every_visit"
    else:
        raise ValueError("Invalid choice. Enter 1 or 2.")

    V, counts = monte_carlo_prediction(method)

    print(f"\nMethod: {method}")
    print(f"Episodes: {NUM_EPISODES}")
    print(f"Discount Factor: {GAMMA}")

    display_values(V)
    display_counts(counts)

    print("\nRESULT")
    print("-" * 40)
    print("Monte Carlo prediction completed.")
    print("State values estimated from sampled returns.")
