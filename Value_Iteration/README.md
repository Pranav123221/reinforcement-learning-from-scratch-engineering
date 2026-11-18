
#  Value Iteration

Implementation of the Value Iteration algorithm using
the Bellman Optimality Equation and a custom GridWorld.

## Concepts Covered

- Value Iteration
- Bellman Optimality Equation
- Optimal State Value Function
- Greedy Policy Extraction
- Discount Factor
- Convergence Threshold
- Model-Based Reinforcement Learning

## Environment

Custom deterministic 4x4 GridWorld.

| Parameter | Value |
|---|---|
| Grid Size | 4x4 |
| Goal | (3,3) |
| Pit | (1,3) |
| Goal Reward | +10 |
| Pit Reward | -10 |
| Normal Reward | -1 |
| Discount Factor | 0.9 |

## Mathematical Foundation

### Bellman Optimality Equation

V*(s) = max_a Σs',r p(s',r|s,a)
        [r + γV*(s')]

### Value Iteration Update

V(k+1)(s) = max_a Σs',r p(s',r|s,a)
            [r + γVk(s')]

### Optimal Policy

π*(s) = argmax_a Σs',r p(s',r|s,a)
        [r + γV*(s')]

## Implementation Workflow

1. Initialize state values to zero.
2. Calculate action values for every possible action.
3. Update each state using the maximum action value.
4. Calculate the maximum change in state values.
5. Repeat until the convergence threshold is reached.
6. Extract the greedy policy from the converged values.

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python value_iteration.py
```

## Experiments

Try modifying:

- GAMMA = 0.5, 0.9, 0.99
- THETA = 1e-6, 1e-8
- Goal reward
- Pit penalty
- Grid dimensions

Compare the resulting optimal values and policies.

## Difference Between Policy Iteration and Value Iteration

Policy Iteration:
- Evaluates the current policy.
- Improves the policy.
- Repeats until policy stability.

Value Iteration:
- Directly applies the Bellman optimality update.
- Uses the maximum action value.
- Extracts the policy after convergence.

## Important Note

This is model-based Dynamic Programming.

The environment's transition and reward model is known.
The algorithm does not learn from sampled experience.

Terminal rewards are received on entering terminal states.
Terminal state values are initialized and maintained at zero.

## Technologies

- Python
- NumPy

## Learning Outcome

Understand how Value Iteration calculates optimal state
values and extracts a greedy optimal policy using the
Bellman Optimality Equation.
