
# Temporal Difference Learning (TD 0)

## Overview

Implementation of Temporal Difference Learning using
the TD(0) prediction algorithm in a custom 4x4 GridWorld.

## Concepts

- Model-free Reinforcement Learning
- Bootstrapping
- TD Target
- TD Error
- State-value estimation
- Learning Rate
- Discount Factor

## Mathematical Formulation

TD Target:

TD_Target = R + gamma * V(S')

TD Error:

delta = R + gamma * V(S') - V(S)

TD(0) Update:

V(S) = V(S) + alpha * delta

For terminal states:

V(Terminal) = 0

## Environment

- Grid size: 4x4
- Start: (0, 0)
- Goal: (3, 3)
- Pit: (2, 2)
- Step Reward: -1
- Goal Reward: +10
- Pit Reward: -10

## Hyperparameters

- Learning Rate: 0.1
- Discount Factor: 0.9
- Episodes: 5000
- Maximum Steps: 100

## Run

```bash
python td_learning.py
```

## Learning Outcome

Understanding online value updates, bootstrapping,
TD targets, TD errors, and state-value prediction.
