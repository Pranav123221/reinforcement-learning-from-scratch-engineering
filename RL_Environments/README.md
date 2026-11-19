
# CartPole-v1: Q-Learning with State Discretization

## Overview

This project implements tabular Q-Learning on the
Gymnasium CartPole-v1 environment.

CartPole has a continuous observation space.
State discretization converts continuous observations
into discrete states so that a Q-table can be used.

## Environment

- Environment: CartPole-v1
- Library: Gymnasium
- State Dimensions: 4
- Actions: 2
- Maximum Episode Length: 500 steps

## State Representation

S = [x, x_dot, theta, theta_dot]

Where:

- x: Cart Position
- x_dot: Cart Velocity
- theta: Pole Angle
- theta_dot: Pole Angular Velocity

## Action Space

- 0: Push Left
- 1: Push Right

## Reward

Reward = +1 for every timestep the pole remains balanced.

## Mathematical Formulation

TD Target:

TD_Target = R + gamma * max Q(S', a)

TD Error:

delta = TD_Target - Q(S,A)

Q-Learning Update:

Q(S,A) = Q(S,A) + alpha * delta

Terminal transition:

TD_Target = R

## State Discretization

The continuous observation dimensions are converted
into discrete bins.

Number of bins:

[6, 12, 12, 12]

Total discrete states:

6 * 12 * 12 * 12 = 10368

Q-table dimensions:

[6, 12, 12, 12, 2]

## Hyperparameters

- Episodes: 20000
- Learning Rate: 0.1
- Discount Factor: 0.99
- Initial Epsilon: 1.0
- Minimum Epsilon: 0.05
- Epsilon Decay: 0.9997
- Evaluation Episodes: 100

## Installation

pip install -r requirements.txt

## Run

python cartpole_qlearning.py

## Generated Files

- cartpole_training.png
- cartpole_q_table.npy

## Evaluation

The agent is evaluated using a greedy policy
without exploration.

A reward of 475 or more in an episode is used
as a success indicator.

The average reward across 100 evaluation episodes
is also reported.

## Learning Outcomes

- Gymnasium environment API
- Continuous observation spaces
- State discretization
- Tabular Q-Learning
- Epsilon-Greedy exploration
- Training performance visualization
- Greedy policy evaluation

## Limitations

Tabular Q-Learning with discretization is a
baseline approach for continuous-state environments.

Performance depends on bin selection, hyperparameters,
training duration, and exploration.

More advanced function approximation methods can
be explored in future projects.
