
# SARSA - On-Policy Temporal Difference Control

## Overview

Implementation of SARSA (State-Action-Reward-State-Action)
using a custom 4x4 GridWorld environment.

The agent learns action-value estimates using
an epsilon-greedy behavior policy.

## Concepts

- On-Policy Reinforcement Learning
- Temporal Difference Control
- State-Action Value Function
- Epsilon-Greedy Policy
- TD Target
- TD Error
- Exploration vs Exploitation

## Mathematical Formulation

TD Target:

TD_Target = R + gamma * Q(S', A')

TD Error:

delta = R + gamma * Q(S', A') - Q(S, A)

SARSA Update:

Q(S,A) = Q(S,A) + alpha * delta

Terminal transition:

TD_Target = R

## Environment

- Grid Size: 4x4
- Start State: (0, 0)
- Goal State: (3, 3)
- Pit State: (2, 2)
- Step Reward: -1
- Goal Reward: +10
- Pit Reward: -10

## Hyperparameters

- Learning Rate: 0.1
- Discount Factor: 0.9
- Initial Epsilon: 1.0
- Minimum Epsilon: 0.05
- Epsilon Decay: 0.9995
- Training Episodes: 10000

## Algorithm Workflow

1. Initialize the Q-table.
2. Start from the initial state.
3. Select an action using epsilon-greedy.
4. Execute the action.
5. Observe reward and next state.
6. Select the next action using the same policy.
7. Update Q(S,A) using the SARSA equation.
8. Continue until a terminal state is reached.
9. Decay epsilon and repeat.

## Run

python sarsa.py

## Expected Output

- Learned policy grid
- Action-value estimates
- Training configuration
- Average reward over the last 100 episodes

Exact numerical values can vary with training and random
action selection.

## Learning Outcomes

Understanding on-policy TD control, SARSA updates,
Q-value learning, and epsilon-greedy exploration.
