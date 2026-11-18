
# 04 - Discount Factor and Discounted Rewards

## Overview

This project explores the Discount Factor (Gamma)
in Reinforcement Learning.

The implementation demonstrates how different
discount factor values affect the total return
of an agent over a sequence of rewards.

## Learning Objectives

- Understand Discount Factor (Gamma).
- Calculate Discounted Returns.
- Compare different Gamma values.
- Understand immediate and delayed rewards.
- Calculate returns at every timestep.
- Analyze future reward importance.

## Mathematical Foundation

The discounted return is:

G_t = R_(t+1) + gamma * G_(t+1)

Where:

- G_t: Discounted Return
- R_(t+1): Next reward
- Gamma: Discount Factor

## Gamma Experiments

The following values are evaluated:

- 0.00
- 0.50
- 0.90
- 0.99
- 1.00

## Reward Sequence

[10, 5, 20, 10]

## Experimental Results

| Gamma | Discounted Return |
|---|---:|
| 0.00 | 10.0000 |
| 0.50 | 18.7500 |
| 0.90 | 37.9900 |
| 0.99 | 44.2550 |
| 1.00 | 45.0000 |

## Reward Timing Experiment

Two reward sequences are compared:

Immediate:
[10, 0, 0, 0]

Delayed:
[0, 0, 0, 10]

At Gamma = 0.9:

- Immediate Return = 10
- Delayed Return = 7.29

This demonstrates the effect of discounting
on future rewards.

## Project Structure

04_Discount_Factor/
├── discount_factor.py
├── README.md
└── requirements.txt

## Installation

```bash
pip install -r requirements.txt
```

## Execution

```bash
python discount_factor.py
```

## Technologies

- Python
- NumPy

## Key Concepts Learned

1. Discount Factor
2. Discounted Return
3. Future Reward Weighting
4. Immediate vs Delayed Rewards
5. Timestep Returns
6. Gamma Sensitivity Analysis

## Implementation Notes

All discounted return calculations are
implemented from scratch.

The experiments use predefined reward sequences.
No RL agent is trained in this module.

## Next Step

Implement State-Value Functions V(s) and
understand how an RL agent estimates the
expected future return from a state.

---

Part of the Reinforcement Learning From Scratch series.
****
