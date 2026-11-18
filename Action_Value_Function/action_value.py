
import numpy as np


# ==========================================
# ENVIRONMENT
# ==========================================

class GridWorld:

    def __init__(self):

        self.rows = 4
        self.cols = 4

        self.start_state = (0, 0)
        self.goal_state = (3, 3)
        self.pit_state = (1, 3)

        self.actions = ["UP", "DOWN", "LEFT", "RIGHT"]

        self.max_steps = 50

        self.reset()

    def reset(self, start_state=(0, 0)):

        self.agent_position = start_state
        self.steps = 0

        return self.agent_position

    def step(self, action):

        row, col = self.agent_position

        if action == "UP":
            row -= 1

        elif action == "DOWN":
            row += 1

        elif action == "LEFT":
            col -= 1

        elif action == "RIGHT":
            col += 1

        else:
            raise ValueError("Invalid action")

        row = int(np.clip(row, 0, self.rows - 1))
        col = int(np.clip(col, 0, self.cols - 1))

        next_state = (row, col)

        self.agent_position = next_state
        self.steps += 1

        if next_state == self.goal_state:
            reward = 10
            terminated = True

        elif next_state == self.pit_state:
            reward = -10
            terminated = True

        else:
            reward = -1
            terminated = False

        truncated = (
            self.steps >= self.max_steps
            and not terminated
        )

        return next_state, reward, terminated, truncated


# ==========================================
# POLICY
# ==========================================

class EpsilonSoftPolicy:

    def __init__(self, actions, epsilon=0.2):

        self.actions = actions
        self.epsilon = epsilon

    def preferred_action(self, state):

        row, col = state

        if col == 3 and row < 3:
            return "LEFT"

        if row < 3:
            return "DOWN"

        if col < 3:
            return "RIGHT"

        return "UP"

    def select_action(self, state):

        if np.random.random() < self.epsilon:

            return np.random.choice(self.actions)

        return self.preferred_action(state)


# ==========================================
# DISCOUNTED RETURN
# ==========================================

def calculate_return(rewards, gamma):

    G = 0

    for reward in reversed(rewards):

        G = reward + gamma * G

    return G


# ==========================================
# MONTE CARLO Q-VALUE ESTIMATION
# ==========================================

def estimate_action_values(
    env,
    policy,
    episodes=500,
    gamma=0.9
):

    q_values = {}

    all_states = [
        (r, c)
        for r in range(env.rows)
        for c in range(env.cols)
        if (r, c) not in [
            env.goal_state,
            env.pit_state
        ]
    ]

    for state in all_states:

        for action in env.actions:

            sampled_returns = []

            for episode in range(episodes):

                # Start from the selected state
                current_state = env.reset(state)

                rewards = []

                # Force the first action to be evaluated
                next_state, reward, terminated, truncated = (
                    env.step(action)
                )

                rewards.append(reward)
                current_state = next_state

                # Continue using the evaluation policy
                while not (terminated or truncated):

                    next_action = policy.select_action(
                        current_state
                    )

                    (
                        next_state,
                        reward,
                        terminated,
                        truncated
                    ) = env.step(next_action)

                    rewards.append(reward)
                    current_state = next_state

                # Calculate return for this state-action pair
                G = calculate_return(rewards, gamma)

                sampled_returns.append(G)

            # Average sampled returns
            q_values[(state, action)] = np.mean(
                sampled_returns
            )

    return q_values


# ==========================================
# Q-TABLE DISPLAY
# ==========================================

def display_q_table(env, q_values):

    print("\nACTION-VALUE TABLE Q(s,a)")
    print("=" * 75)

    for row in range(env.rows):

        for col in range(env.cols):

            state = (row, col)

            if state == env.goal_state:
                print(state, "GOAL")
                continue

            if state == env.pit_state:
                print(state, "PIT")
                continue

            print("\nState:", state)

            for action in env.actions:

                value = q_values[(state, action)]

                print(
                    f"  {action:<8}: {value:8.3f}"
                )


# ==========================================
# GREEDY ACTION SELECTION
# ==========================================

def select_best_actions(env, q_values):

    print("\nBEST ACTION FOR EACH STATE")
    print("=" * 55)

    for row in range(env.rows):

        for col in range(env.cols):

            state = (row, col)

            if state in [
                env.goal_state,
                env.pit_state
            ]:
                continue

            best_action = max(
                env.actions,
                key=lambda action: q_values[
                    (state, action)
                ]
            )

            best_value = q_values[
                (state, best_action)
            ]

            print(
                f"State {state} | "
                f"Best Action: {best_action} | "
                f"Q-value: {best_value:.3f}"
            )


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    np.random.seed(42)

    GAMMA = 0.9
    EPISODES = 500

    env = GridWorld()

    policy = EpsilonSoftPolicy(
        actions=env.actions,
        epsilon=0.2
    )

    print("ACTION-VALUE FUNCTION EXPERIMENT")

    print("Gamma:", GAMMA)
    print("Episodes per state-action pair:", EPISODES)

    q_values = estimate_action_values(
        env,
        policy,
        episodes=EPISODES,
        gamma=GAMMA
    )

    display_q_table(
        env,
        q_values
    )

    select_best_actions(
        env,
        q_values
    )
