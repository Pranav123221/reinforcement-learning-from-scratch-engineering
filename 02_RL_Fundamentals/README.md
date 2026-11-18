
# 02 - Reinforcement Learning Fundamentals

## Overview

This project implements the fundamental components
of Reinforcement Learning using Python and NumPy.

A custom GridWorld environment is created where
an agent interacts with its environment by selecting
actions and receiving rewards.

## Learning Objectives

- Understand Agent and Environment architecture.
- Implement State and Action spaces.
- Understand Reward feedback.
- Implement Agent-Environment interaction.
- Understand Episodes and termination.
- Implement a Random Policy.

## Core Components

### 1. Agent

The agent interacts with the environment and
selects actions based on its policy.

### 2. Environment

The environment defines the rules, state transitions
and reward mechanism.

### 3. State

Represents the current position of the agent.

### 4. Action

Available actions:

- UP
- DOWN
- LEFT
- RIGHT

### 5. Reward

| Condition | Reward |
|---|---:|
| Reaching Goal | +10 |
| Entering Pit | -10 |
| Normal Movement | -1 |

## Environment Configuration

| Parameter | Value |
|---|---|
| Grid Size | 4 x 4 |
| Starting State | (0, 0) |
| Goal State | (3, 3) |
| Pit State | (1, 3) |
| Maximum Steps | 30 |

## Agent Architecture

The RandomAgent class implements random
action selection.

The agent selects one action from the available
action space using NumPy.

No learning or optimization is performed yet.

## Interaction Workflow

1. Reset the environment.
2. Observe the current state.
3. Agent selects an action.
4. Environment executes the action.
5. Environment returns next state and reward.
6. Agent updates its current state.
7. Repeat until termination or truncation.

## Termination Conditions

- Terminated: Agent reaches the goal or pit.
- Truncated: Maximum episode steps are reached.

## Project Structure

02_RL_Fundamentals/
├── rl_fundamentals.py
├── README.md
└── requirements.txt

## Installation

```bash
pip install -r requirements.txt
```

## Execution

```bash
python rl_fundamentals.py
```

## Technologies

- Python
- NumPy

## Key Concepts Learned

- Agent-Environment Architecture
- State and Action Spaces
- Reward Mechanism
- Random Policy
- Episode Execution
- Environment Interaction
- Termination vs Truncation

## Implementation Notes

The environment and agent are implemented
from scratch.

The agent currently follows a random policy.
It does not learn from previous experiences.

## Next Step

Implement Policy, Return and Episode Reward
calculations to understand how an RL agent
evaluates its decisions.

---

Part of the Reinforcement Learning From Scratch series.
