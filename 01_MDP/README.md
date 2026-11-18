
# 01 - Markov Decision Process (MDP)

## Overview

This project implements a Markov Decision Process (MDP)
from scratch using Python and NumPy.

A Markov Decision Process is a mathematical framework
used to model sequential decision-making problems
under uncertainty.

It forms the foundation of Reinforcement Learning.

## Objectives

- Understand the MDP framework.
- Define states and action spaces.
- Implement state transitions.
- Design a reward function.
- Handle terminal states.
- Build a custom GridWorld environment.

## MDP Components

An MDP is represented by:

MDP = (S, A, P, R, Gamma)

Where:

- S: State Space
- A: Action Space
- P: State Transition Probability
- R: Reward Function
- Gamma: Discount Factor

## Environment Design

We create a 4x4 GridWorld environment.

| Component | Configuration |
|---|---|
| Grid Size | 4 x 4 |
| Initial State | (0, 0) |
| Goal State | (3, 3) |
| Pit State | (1, 3) |
| Goal Reward | +10 |
| Pit Penalty | -10 |
| Step Reward | -1 |
| Discount Factor | 0.9 |

## Available Actions

The agent can perform four actions:

- UP
- DOWN
- LEFT
- RIGHT

Boundary conditions prevent the agent
from moving outside the environment.

## Reward Mechanism

- Reaching the goal gives +10.
- Entering the pit gives -10.
- Every normal movement gives -1.
- Goal and pit states terminate the episode.

## Project Structure

01_MDP/
├── mdp_environment.py
└── README.md

## Installation

Install the required dependency:

```bash
pip install numpy
```

## Execution

```bash
python mdp_environment.py
```

## Key Concepts Learned

1. Markov Decision Process
2. State Space
3. Action Space
4. State Transitions
5. Reward Function
6. Terminal States
7. Discount Factor

## Implementation Notes

The environment is implemented from scratch
without using an existing RL environment library.

The agent follows predefined actions.
No learning algorithm has been implemented yet.

## Next Step

Implement Agent, Environment, State, Action
and Reward concepts in a separate practical module.

---

Part of the Reinforcement Learning From Scratch series.
