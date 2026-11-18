
# 06 - Action-Value Function Q(s,a)

## Overview

This project implements Action-Value Function
estimation using Monte Carlo simulation.

A custom GridWorld environment is used to estimate
the expected future return of taking different
actions in different states.

## Learning Objectives

- Understand Action-Value Functions.
- Understand State-Value vs Action-Value.
- Implement state-action pair evaluation.
- Generate a Q-table.
- Estimate Q-values using Monte Carlo sampling.
- Select greedy actions using Q-values.

## Mathematical Foundation

The Action-Value Function is defined as:

Q_pi(s,a) = E_pi[G_t | S_t=s, A_t=a]

Where:

- Q_pi(s,a): Action-value under policy pi
- s: Current state
- a: Selected action
- G_t: Discounted future return
- E_pi: Expected value under policy pi

## State-Value vs Action-Value

### State-Value Function

V_pi(s) evaluates the expected return
from a particular state under policy pi.

### Action-Value Function

Q_pi(s,a) evaluates the expected return
after taking action a in state s and
following policy pi afterward.

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
| Episodes per Pair | 500 |
| Epsilon | 0.2 |

## Action Space

- UP
- DOWN
- LEFT
- RIGHT

## Monte Carlo Q-Value Estimation

For each non-terminal state-action pair:

1. Reset the environment to the selected state.
2. Execute the selected action.
3. Follow the epsilon-soft policy afterward.
4. Collect rewards until termination or truncation.
5. Calculate discounted return.
6. Repeat across multiple episodes.
7. Average the sampled returns.

The estimated Q-value is:

Q(s,a) approximately equals:

(1/N) * Sum(G_i)

Where N represents the number of sampled episodes.

## Greedy Action Selection

The action with the highest estimated
Q-value is selected:

a* = argmax_a Q(s,a)

This provides a way to identify the
highest-valued action for each state
under the estimated action values.

## Project Structure

06_Action_Value_Function/
├── action_value.py
├── README.md
└── requirements.txt

## Installation

```bash
pip install -r requirements.txt
```

## Execution

```bash
python action_value.py
```

## Technologies

- Python
- NumPy

## Key Concepts Learned

1. Action-Value Function Q(s,a)
2. State-Value Function V(s)
3. State-Action Pairs
4. Monte Carlo Policy Evaluation
5. Q-Table
6. Greedy Action Selection
7. Epsilon-Soft Policy

## Implementation Notes

This implementation estimates action values
using Monte Carlo sampling.

The first action is forced to evaluate a
specific state-action pair. Subsequent actions
follow an epsilon-soft evaluation policy.

The resulting Q-values estimate the value
of the current policy, not the optimal policy.

No Q-learning update rule is implemented here.

## Next Step

Implement the Bellman Equation and understand
how state values and action values are related
through recursive reward calculations.

---

Part of the Reinforcement Learning From Scratch series.
