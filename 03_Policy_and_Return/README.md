
# 03 - Policy, Return and Episode Rewards

## Overview

This project explores Policy and Return concepts
in Reinforcement Learning through a custom
GridWorld environment.

Three different policy strategies are implemented
to understand how agents select actions and
accumulate rewards.

## Learning Objectives

- Understand Policy in Reinforcement Learning.
- Implement Random Policy.
- Implement Deterministic Policy.
- Implement Stochastic Policy.
- Calculate Episode Rewards.
- Calculate Discounted Returns.
- Compare different action-selection strategies.

## Policy Types

### 1. Random Policy

Selects an action randomly from the available
action space.

### 2. Deterministic Policy

Maps each state to a fixed action.

Mathematically:

a = pi(s)

### 3. Stochastic Policy

Selects actions according to a probability
distribution.

Mathematically:

pi(a|s) = P(A_t = a | S_t = s)

## Return Calculation

The undiscounted return is the sum of rewards
collected throughout an episode.

G_t = Sum of future rewards

The discounted return is calculated as:

G_t = R_(t+1) + gamma * G_(t+1)

Where:

- G_t: Discounted return
- R: Immediate reward
- gamma: Discount factor

## Environment Configuration

| Parameter | Value |
|---|---|
| Grid Size | 4 x 4 |
| Start State | (0, 0) |
| Goal State | (3, 3) |
| Pit State | (1, 3) |
| Goal Reward | +10 |
| Pit Penalty | -10 |
| Step Reward | -1 |
| Discount Factor | 0.9 |

## Deterministic Policy Experiment

The agent follows a predefined path:

DOWN -> DOWN -> DOWN ->
RIGHT -> RIGHT -> RIGHT

Expected results:

- Total Steps: 6
- Total Episode Reward: +5
- Initial Discounted Return: 1.8098

## Random and Stochastic Policy Experiments

Both policies select actions differently.

Their episode rewards and discounted returns
depend on the selected actions.

The random policy selects actions uniformly.

The stochastic policy uses state-dependent
action probabilities.

## Project Structure

03_Policy_and_Return/
├── policy_return.py
├── README.md
└── requirements.txt

## Installation

```bash
pip install -r requirements.txt
```

## Execution

```bash
python policy_return.py
```

## Technologies

- Python
- NumPy

## Key Concepts Learned

1. Policy
2. Random Policy
3. Deterministic Policy
4. Stochastic Policy
5. Episode Reward
6. Cumulative Return
7. Discounted Return
8. Discount Factor

## Implementation Notes

All three policy strategies are implemented
from scratch.

The policies are not optimized through learning.
This experiment focuses on action selection
and reward calculation.

## Next Step

Study Discounted Return and the mathematical
role of the Discount Factor (Gamma) through
practical numerical experiments.

---

Part of the Reinforcement Learning From Scratch series.
