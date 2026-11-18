
# Monte Carlo Control

Implementation of Monte Carlo Control using First-Visit
Monte Carlo estimation and an epsilon-soft behavior policy.

## Concepts Covered

- Monte Carlo Control
- Action Value Function Q(s,a)
- Exploring Starts
- First-Visit Monte Carlo
- Epsilon-Soft Policy
- Exploration and Exploitation
- Greedy Policy Extraction
- Policy Improvement

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
| Epsilon | 0.1 |
| Episodes | 30,000 |

## Mathematical Foundation

### Action Value

Qπ(s,a) = Eπ[G(t) | S(t)=s, A(t)=a]

### Monte Carlo Estimation

Q(s,a) = Sum of observed returns / Visit count

### Epsilon-Soft Policy

The agent selects a random action with probability epsilon.
Otherwise, it selects a greedy action.

### Greedy Policy

π(s) = argmax_a Q(s,a)

## Algorithm Workflow

1. Initialize Q(s,a) values.
2. Select a random non-terminal starting state.
3. Generate an episode using epsilon-soft action selection.
4. Calculate discounted returns backwards.
5. Use first-visit state-action returns.
6. Update Q(s,a) using sample averages.
7. Extract a greedy policy from learned Q-values.

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python monte_carlo_control.py
```

## Experiments

Try modifying:

- EPSILON = 0.05, 0.1, 0.2
- GAMMA = 0.5, 0.9, 0.99
- Number of episodes
- Reward values
- Starting-state distribution

Compare the learned Q-values and policies.

## Important Notes

- Exploring Starts samples initial states randomly.
- Epsilon-soft action selection supports exploration
  throughout an episode.
- Q-values are estimated from complete episode returns.
- The final policy is greedy with respect to learned Q.
- Finite training does not guarantee that every
  state-action pair has received enough visits.
- A timeout truncates an episode; its remaining
  continuation value is not included in the return.

This implementation is an educational Exploring Starts
variant with epsilon-soft behavior, not a proof of
convergence to the optimal policy.

## Technologies

- Python
- NumPy

## Learning Outcome

Understand how Monte Carlo methods estimate action values
and use exploration plus greedy policy extraction to
improve an agent's behavior.
