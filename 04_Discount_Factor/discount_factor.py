
import numpy as np


# ==========================================
# REWARD SEQUENCES
# ==========================================

rewards = [10, 5, 20, 10]

gamma_values = [0.0, 0.5, 0.9, 0.99, 1.0]


# ==========================================
# DISCOUNTED RETURN CALCULATION
# ==========================================

def calculate_return(rewards, gamma):

    G = 0

    for reward in reversed(rewards):

        G = reward + gamma * G

    return G


# ==========================================
# RETURNS AT EVERY TIMESTEP
# ==========================================

def calculate_all_returns(rewards, gamma):

    returns = []

    G = 0

    for reward in reversed(rewards):

        G = reward + gamma * G

        returns.insert(0, G)

    return returns


# ==========================================
# GAMMA COMPARISON EXPERIMENT
# ==========================================

def compare_gamma_values(rewards, gamma_values):

    print("\nDISCOUNT FACTOR EXPERIMENT")
    print("=" * 65)

    print(f"{'Gamma':<12}{'Discounted Return':<25}{'Interpretation'}")

    print("-" * 65)

    for gamma in gamma_values:

        discounted_return = calculate_return(
            rewards,
            gamma
        )

        if gamma == 0:
            interpretation = "Immediate reward only"

        elif gamma < 0.9:
            interpretation = "Lower future reward weight"

        elif gamma < 1.0:
            interpretation = "High future reward weight"

        else:
            interpretation = "No discounting"

        print(
            f"{gamma:<12}"
            f"{discounted_return:<25.4f}"
            f"{interpretation}"
        )


# ==========================================
# TIMESTEP RETURN EXPERIMENT
# ==========================================

def timestep_experiment(rewards, gamma):

    returns = calculate_all_returns(
        rewards,
        gamma
    )

    print("\nTIMESTEP RETURN ANALYSIS")
    print("=" * 45)

    for timestep, (reward, G) in enumerate(
        zip(rewards, returns)
    ):

        print(
            f"Time: {timestep} | "
            f"Reward: {reward} | "
            f"Return: {G:.4f}"
        )


# ==========================================
# IMMEDIATE VS DELAYED REWARD EXPERIMENT
# ==========================================

def reward_timing_experiment(gamma):

    immediate_rewards = [10, 0, 0, 0]

    delayed_rewards = [0, 0, 0, 10]

    immediate_return = calculate_return(
        immediate_rewards,
        gamma
    )

    delayed_return = calculate_return(
        delayed_rewards,
        gamma
    )

    print("\nREWARD TIMING EXPERIMENT")
    print("=" * 45)

    print("Gamma:", gamma)

    print("Immediate Reward Return:",
          immediate_return)

    print("Delayed Reward Return:",
          delayed_return)


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    np.set_printoptions(precision=4)

    print("Reinforcement Learning")
    print("Discount Factor and Discounted Returns")

    print("\nReward Sequence:", rewards)

    compare_gamma_values(
        rewards,
        gamma_values
    )

    timestep_experiment(
        rewards,
        gamma=0.9
    )

    reward_timing_experiment(
        gamma=0.9
    )
