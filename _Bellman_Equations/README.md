
# : Bellman Equations

Implementation of Bellman Expectation Equations for State-Value
and Action-Value functions using a custom GridWorld environment.

## Concepts Covered

- Bellman Expectation Equation
- State Value Function V(s)
- Action Value Function Q(s, a)
- Stochastic Policies
- Transition and Reward Models
- Iterative Policy Evaluation
- Recursive Value Estimation
- Relationship between V(s) and Q(s, a)

## Environment

A deterministic 4x4 GridWorld:

- Goal state: (3, 3)
- Pit state: (1, 3)
- Goal reward: +10
- Pit reward: -10
- Normal transition reward: -1
- Discount factor: 0.9

The agent follows a uniform random policy, selecting each
available action with probability 0.25.

## Mathematical Foundation

### 1. Bellman Expectation Equation: V(s)

Vπ(s) = Σa π(a|s) Σs',r p(s',r|s,a)
        [r + γVπ(s')]

### 2. Bellman Expectation Equation: Q(s,a)

Qπ(s,a) = Σs',r p(s',r|s,a)
          [r + γΣa' π(a'|s')Qπ(s',a')]

### 3. Relationship Between V and Q

Vπ(s) = Σa π(a|s)Qπ(s,a)

## Implementation

The program:

1. Creates a deterministic GridWorld transition model.
2. Defines a uniform stochastic policy.
3. Initializes all state values to zero.
4. Iteratively applies the Bellman expectation equation.
5. Stops when the maximum value change is below theta.
6. Calculates Q(s,a) using the converged V(s).
7. Verifies the relationship between V(s) and Q(s,a).

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python bellman_equations.py
```

## Experiments

Try changing:

- GAMMA = 0.5, 0.9, 0.99
- THETA = 1e-6, 1e-8
- The policy from uniform random to deterministic.
- The rewards for the goal and pit.

Observe how the state values and action values change.

## Important Note

This module performs policy evaluation, not optimal policy
learning.

The agent follows a fixed policy. It does not improve its
policy or learn an optimal strategy.

Policy improvement and optimal value estimation are covered
in upcoming modules.

## Technologies

- Python
- NumPy

## Learning Outcome

Understand how Bellman equations recursively connect immediate
rewards with expected future values, and how Vπ(s) and Qπ(s,a)
are mathematically related.
