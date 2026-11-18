
#  Dynamic Programming — Policy Iteration

Implementation of Dynamic Programming using Policy Iteration
in a custom deterministic GridWorld environment.

## Concepts Covered

- Dynamic Programming
- Model-Based Reinforcement Learning
- Policy Evaluation
- Policy Improvement
- Policy Iteration
- Bellman Expectation Equation
- Greedy Action Selection
- Policy Convergence

## Environment

A 4x4 GridWorld environment.

| Parameter | Value |
|---|---|
| Grid size | 4 x 4 |
| Start state | (0, 0) |
| Goal state | (3, 3) |
| Pit state | (1, 3) |
| Goal reward | +10 |
| Pit reward | -10 |
| Normal reward | -1 |
| Discount factor | 0.9 |

The environment uses deterministic transitions.

If an action attempts to move outside the grid,
the agent remains in the same state and receives -1.

Goal and pit are terminal states.

## Algorithms

### 1. Policy Evaluation

Estimate the value of every state under the current policy.

Vπ(s) = Σs',r p(s',r|s,π(s))
        [r + γVπ(s')]

### 2. Policy Improvement

Calculate action values and choose the greedy action.

Qπ(s,a) = Σs',r p(s',r|s,a)
          [r + γVπ(s')]

π_new(s) = argmax_a Qπ(s,a)

### 3. Policy Iteration

1. Initialize a policy.
2. Evaluate the policy.
3. Improve the policy.
4. Check whether the policy changed.
5. Repeat until the policy becomes stable.

## Implementation Details

- Initial policy selects RIGHT in every non-terminal state.
- Policy evaluation uses iterative Bellman updates.
- Policy improvement uses greedy action selection.
- A fixed action ordering resolves equal-value ties.
- Evaluation stops when the maximum state-value change
  is smaller than the convergence threshold.

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python dynamic_programming.py
```

## Experiments

Change the following parameters:

- GAMMA = 0.5
- GAMMA = 0.9
- GAMMA = 0.99
- Goal reward
- Pit reward
- Initial policy

Observe how the state values and improved policy change.

## Important Note

This implementation assumes that the environment's transition
and reward model is already known.

Dynamic Programming evaluates and improves a policy using
that known model. It does not learn the transition model
from real-world interaction.

Policy Iteration converges to an optimal policy for this
finite discounted MDP under the implemented model.

## Technologies

- Python
- NumPy

## Learning Outcome

Understand how policy evaluation and policy improvement
work together to find an optimal policy using Dynamic
Programming.
