
# 05 - State-Value Function (V(s))

## Overview

This project implements State-Value Function
estimation using Monte Carlo simulation.

A custom GridWorld environment is used to estimate
the expected future return from different states
under an epsilon-soft policy.

## Learning Objectives

- Understand State-Value Functions.
- Understand expected future returns.
- Implement epsilon-soft action selection.
- Simulate multiple episodes.
- Calculate discounted returns.
- Estimate state values using Monte Carlo sampling.
- Generate a state-value table.

## Mathematical Foundation

The State-Value Function is defined as:

V_pi(s) = E_pi[G_t | S_t = s]

Where:

- V_pi(s): State value under policy pi
- G_t: Discounted future return
- S_t: Current state
- E_pi: Expected value under policy pi

## Monte Carlo Estimation

The state value is estimated using sampled returns:

V(s) approximately equals:

(1 / N) * Sum(G_i)

Where:

- N: Number of sampled episodes
- G_i: Discounted return from the starting state

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
| Episodes per State | 500 |
| Epsilon | 0.2 |

## Policy Implementation

An epsilon-soft policy is implemented.

- With probability 1 - epsilon, the agent
  selects its preferred action.
- With probability epsilon, it selects a
  random action.

The policy attempts to navigate towards the goal
while avoiding the pit.

## State-Value Estimation Workflow

1. Enumerate all non-terminal states.
2. Start an episode from each state.
3. Execute the epsilon-soft policy.
4. Collect rewards until termination or truncation.
5. Calculate discounted return.
6. Repeat for multiple episodes.
7. Average sampled returns.
8. Display the estimated state values.

## Project Structure

05_State_Value_Function/
├── state_value.py
├── README.md
└── requirements.txt

## Installation

```bash
pip install -r requirements.txt
```

## Execution

```bash
python state_value.py
```

## Technologies

- Python
- NumPy

## Key Concepts Learned

1. State-Value Function V(s)
2. Expected Return
3. Monte Carlo Estimation
4. Epsilon-Soft Policy
5. Exploration
6. Discounted Rewards
7. State Evaluation

## Implementation Notes

The implementation estimates state values
through Monte Carlo sampling.

Each non-terminal state is evaluated independently
using multiple episodes.

The values are policy-dependent estimates.
They are not optimal state values.

No policy optimization or value-function
learning algorithm is implemented.

## Next Step

Implement the Action-Value Function Q(s, a)
to estimate the expected return of taking
a particular action in a given state.

---

Part of the Reinforcement Learning From Scratch series.
