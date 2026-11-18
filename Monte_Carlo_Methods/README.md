
#: Monte Carlo Methods

Implementation of Monte Carlo Prediction using a custom
GridWorld environment.

## Concepts Covered

- Model-Free Reinforcement Learning
- Monte Carlo Prediction
- First-Visit Monte Carlo
- Every-Visit Monte Carlo
- Episode Generation
- Discounted Returns
- State Value Estimation
- Sample Averaging

## Environment

Custom deterministic 4x4 GridWorld.

| Parameter | Value |
|---|---|
| Grid Size | 4x4 |
| Start State | (0,0) |
| Goal | (3,3) |
| Pit | (1,3) |
| Goal Reward | +10 |
| Pit Reward | -10 |
| Normal Reward | -1 |
| Discount Factor | 0.9 |
| Episodes | 20,000 |

The agent follows a uniform random policy.

## Mathematical Foundation

### Return

G(t) = R(t+1) + γR(t+2) + γ²R(t+3) + ...

### Monte Carlo Prediction

Vπ(s) = Eπ[G(t) | S(t) = s]

### Sample-Based Estimation

V(s) = Sum of observed returns / Number of visits

## Algorithms

### First-Visit Monte Carlo

Uses only the return associated with the first
occurrence of a state in each episode.

### Every-Visit Monte Carlo

Uses returns from every occurrence of a state
within each episode.

## Implementation Workflow

1. Initialize GridWorld.
2. Generate an episode using a fixed policy.
3. Record visited states and received rewards.
4. Calculate discounted returns backwards.
5. Update state return sums and visit counts.
6. Estimate V(s) using sample averages.

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python monte_carlo.py
```

Select First-Visit or Every-Visit Monte Carlo.

## Experiments

Try changing:

- Number of episodes
- Discount factor
- Random policy
- Maximum episode steps
- Random seed

Compare state value estimates between both methods.

## Important Notes

- Monte Carlo prediction does not require a known
  transition probability model.
- Episodes must reach termination to obtain complete
  Monte Carlo returns.
- The maximum step limit prevents endless episodes.
- First-Visit and Every-Visit use different sampling rules.
- This module evaluates a fixed policy; it does not
  improve the policy.

## Technologies

- Python
- NumPy

## Learning Outcome

Understand how complete episodes and sampled discounted
returns can be used to estimate a policy's state values
without a known environment model.
