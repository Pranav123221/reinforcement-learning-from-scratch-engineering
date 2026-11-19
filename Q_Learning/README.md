
# Q-Learning - Off-Policy Temporal Difference Control

## Overview

Implementation of Q-Learning using a custom 4x4 GridWorld.

The agent learns an action-value function using
an epsilon-greedy behavior policy and greedy TD targets.

## Concepts

- Model-Free Reinforcement Learning
- Off-Policy Learning
- Temporal Difference Control
- Q-Value Function
- Bellman Optimality Equation
- Epsilon-Greedy Exploration
- TD Target and TD Error

## Mathematical Formulation

TD Target:

TD_Target = R + gamma * max Q(S', a)

TD Error:

delta = R + gamma * max Q(S', a) - Q(S, A)

Q-Learning Update:

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
- Maximum Steps: 100

## Algorithm Workflow

1. Initialize the Q-table.
2. Start from the initial state.
3. Select an action using epsilon-greedy.
4. Execute the action.
5. Observe reward and next state.
6. Find the maximum Q-value of the next state.
7. Calculate the TD target and TD error.
8. Update the current Q-value.
9. Repeat until termination or step limit.
10. Decay epsilon and train again.

## Run

python q_learning.py

## Expected Output

- Learned greedy policy grid
- State-action Q-values
- Average training reward
- Greedy policy evaluation
- Agent's final state and path

Exact numerical values can vary with training.

## Learning Outcomes

Understanding off-policy TD control, greedy target
policies, Bellman optimality, Q-value updates,
and exploration-exploitation.
