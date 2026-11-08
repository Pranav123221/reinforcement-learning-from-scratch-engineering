<div align="center">

# Reinforcement Learning Implementations

### Mathematical Foundations | Algorithm Development | Autonomous Agent Training

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Gymnasium](https://img.shields.io/badge/Gymnasium-RL-0081A7?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

**From Mathematical Theory to Autonomous Intelligence**

</div>

---

## Reinforcement Learning Architecture

<img width="1024" height="576" alt="Slide5-1024x576" src="https://github.com/user-attachments/assets/fa3913a7-845c-4cc4-b544-e3334dee679d" />




---

### Mathematical Foundations • Algorithm Development • Autonomous Agent Training • Simulation Engineering

A comprehensive, hands-on exploration of **Reinforcement Learning (RL)**, focusing on the mathematical foundations, algorithmic implementations, experimental evaluation, and development of autonomous decision-making agents.

This repository documents the systematic study and practical implementation of Reinforcement Learning algorithms, progressing from classical tabular methods to Deep Reinforcement Learning.

The primary objective is to understand how intelligent agents learn optimal or near-optimal decision-making policies through interaction with dynamic environments, using reward-driven learning and sequential optimization.

Rather than relying entirely on pre-built implementations, the core algorithms are developed from scratch using **Python and NumPy**, with Gymnasium providing standardized environment interfaces. PyTorch will be introduced during the Deep RL phase for neural network-based function approximation.

---

## Table of Contents

1. [Project Objectives](#-project-objectives)
2. [Reinforcement Learning Fundamentals](#-reinforcement-learning-fundamentals)
3. [Mathematical Foundations](#-mathematical-foundations)
4. [Learning Roadmap and Algorithm Implementations](#-learning-roadmap-and-algorithm-implementations)
5. [Environment Engineering with Gymnasium](#-environment-engineering-with-gymnasium)
6. [Experimental Methodology](#-experimental-methodology)
7. [Performance Evaluation](#-performance-evaluation)
8. [Technology Stack](#-technology-stack)
9. [Repository Architecture](#-repository-architecture)
10. [Installation and Setup](#-installation-and-setup)
11. [Running Experiments](#-running-experiments)
12. [Visualization and Analysis](#-visualization-and-analysis)
13. [Flagship Project: NEXUS RL](#-flagship-project-nexus-rl)
14. [Learning Outcomes](#-learning-outcomes)
15. [Development Roadmap](#-development-roadmap)
16. [References](#-references)

---

## Project Objectives

This repository is designed around the following engineering and learning objectives:

- Develop a rigorous understanding of the mathematical principles behind Reinforcement Learning.
- Model sequential decision-making problems using Markov Decision Processes.
- Understand state-transition dynamics, reward structures, policies, and value estimation.
- Derive and implement Bellman equations and recursive value-function updates.
- Implement classical RL algorithms from scratch using NumPy.
- Understand the differences between model-based and model-free learning.
- Analyze exploration-exploitation trade-offs and convergence behaviour.
- Train agents in discrete and continuous control environments.
- Progress towards neural network-based function approximation.
- Compare algorithms using reproducible experiments and quantitative evaluation.
- Design custom simulation environments for autonomous decision-making.

**Core implementation philosophy:** Understand the mathematics, derive the update equations, implement the algorithm, validate its behaviour, and evaluate its performance.

---

## Reinforcement Learning Fundamentals

Reinforcement Learning is a machine learning paradigm in which an agent learns a decision-making strategy by interacting with an environment.

Unlike supervised learning, where the model receives labelled training examples, an RL agent learns through sequential interaction and reward feedback.

### Agent–Environment Interaction

The fundamental interaction cycle consists of:

1. State or observation acquisition.
2. Action selection through a policy.
3. Environment transition.
4. Reward generation.
5. Policy or value-function improvement.

The interaction can be represented as:

`S_t → A_t → (R_{t+1}, S_{t+1})`

Where:

- `S_t`: Current state.
- `A_t`: Action selected by the agent.
- `R_{t+1}`: Immediate reward.
- `S_{t+1}`: Next state.

The objective is to learn a policy that maximizes expected cumulative discounted reward.

### Core Components

| Component | Description |
|---|---|
| Agent | Decision-making entity |
| Environment | System with which the agent interacts |
| State | Representation of the current situation |
| Observation | Information received by the agent |
| Action | Decision taken by the agent |
| Reward | Scalar feedback from the environment |
| Policy | Strategy for selecting actions |
| Return | Cumulative future reward |
| Episode | Complete sequence of interactions |
| Value Function | Expected return from a state |
| Q-Function | Expected return from a state-action pair |

---

## Mathematical Foundations

The mathematical component of this repository focuses on understanding the theory behind RL algorithms rather than treating them as black-box implementations.

### 1. Markov Decision Processes (MDPs)

A Markov Decision Process provides a mathematical framework for modeling sequential decision-making under uncertainty.

An MDP is commonly represented as:

`M = (S, A, P, R, γ)`

Where:

- `S`: State space.
- `A`: Action space.
- `P(s' | s, a)`: State transition probability.
- `R(s, a, s')`: Reward function.
- `γ`: Discount factor.

Topics:

- Markov property.
- State-transition dynamics.
- Transition probability distributions.
- Reward functions.
- Episodic and continuing tasks.
- Fully observable and partially observable environments.
- Deterministic and stochastic environments.

### 2. Return and Discount Factor

The discounted return is:

`G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ...`

Equivalently:

`G_t = Σ(k=0 to ∞) γ^k R_{t+k+1}`

Where:

- `G_t`: Return at time t.
- `γ`: Discount factor, generally between 0 and 1.
- `R`: Reward received at each timestep.

The discount factor controls how future rewards contribute to the current decision-making objective.

### 3. Policy Functions

A policy defines the behaviour of an agent.

A stochastic policy can be expressed as:

`π(a | s) = P(A_t = a | S_t = s)`

A deterministic policy can be expressed as:

`a = π(s)`

Topics:

- Deterministic policies.
- Stochastic policies.
- Policy evaluation.
- Policy improvement.
- Optimal policies.

### 4. State-Value Function

The state-value function measures the expected return when following policy π from state s.

`Vπ(s) = Eπ[G_t | S_t = s]`

### 5. Action-Value Function

The action-value function estimates the expected return when an agent takes action a in state s and subsequently follows policy π.

`Qπ(s, a) = Eπ[G_t | S_t = s, A_t = a]`

### 6. Advantage Function

The advantage function measures how much better an action performs compared with the expected value of the current state.

`Aπ(s, a) = Qπ(s, a) - Vπ(s)`

These concepts provide the theoretical foundation for value-based and policy-based learning algorithms.

### 7. Bellman Equations

Bellman equations express recursive relationships between current and future expected rewards.

**Bellman Expectation Equation for State Values:**

`Vπ(s) = Σ_a π(a|s) Σ_s' P(s'|s,a) [R(s,a,s') + γVπ(s')]`

**Bellman Expectation Equation for Action Values:**

`Qπ(s,a) = Σ_s' P(s'|s,a) [R(s,a,s') + γΣ_a' π(a'|s')Qπ(s',a')]`

**Bellman Optimality Equation:**

`V*(s) = max_a Σ_s' P(s'|s,a) [R(s,a,s') + γV*(s')]`

**Optimal Action-Value Function:**

`Q*(s,a) = Σ_s' P(s'|s,a) [R(s,a,s') + γ max_a' Q*(s',a')]`

These equations form the foundation of Dynamic Programming and many model-free RL algorithms.

---

## Learning Roadmap and Algorithm Implementations

The repository follows a progressive implementation strategy, beginning with fundamental concepts and advancing towards deep and policy-based learning.

### Phase 1: Reinforcement Learning and Environment Fundamentals

Topics:

- Introduction to Reinforcement Learning.
- Agent–Environment interaction.
- State, Action, Reward and Observation.
- Episodes and timesteps.
- Episodic and continuing tasks.
- Reward signals.
- Discounted return.
- Exploration vs exploitation.
- Deterministic and stochastic policies.
- Gymnasium environment interface.

Practical implementations:

- Environment initialization.
- Observation-space analysis.
- Action-space exploration.
- Random agent implementation.
- Episode execution.
- Reward collection.
- Episode statistics.
- Environment visualization.

### Phase 2: Markov Decision Processes and Bellman Theory

Topics:

- Markov property.
- MDP formulation.
- Transition probability matrices.
- Reward matrices.
- State-value functions.
- Action-value functions.
- Bellman expectation equations.
- Bellman optimality equations.
- Optimal policies.

Practical implementations:

- Constructing small MDPs.
- Defining transition and reward matrices.
- Computing expected returns.
- Manually evaluating policies.
- Solving small decision-making problems.

### Phase 3: Dynamic Programming

Dynamic Programming algorithms solve MDPs when the environment model is known.

**Policy Evaluation**

Estimate the state-value function for a given policy.

`V(s) ← Σ_a π(a|s) Σ_s' P(s'|s,a)[R + γV(s')]`

**Policy Improvement**

Improve the policy by selecting actions that maximize expected future value.

`π'(s) = argmax_a Σ_s' P(s'|s,a)[R + γV(s')]`

Algorithms:

- Iterative Policy Evaluation.
- Policy Improvement.
- Policy Iteration.
- Value Iteration.
- Generalized Policy Iteration.

Implementation objectives:

- Implement algorithms using NumPy.
- Understand convergence criteria.
- Analyze Bellman backups.
- Compare policy iteration and value iteration.
- Visualize value functions and optimal policies.

### Phase 4: Monte Carlo Methods

Monte Carlo methods learn from sampled episodes without requiring a known transition model.

Topics:

- Episode-based learning.
- First-visit Monte Carlo.
- Every-visit Monte Carlo.
- Monte Carlo prediction.
- Monte Carlo control.
- Exploring starts.
- On-policy learning.
- Off-policy learning.
- Importance sampling.

Implementation objectives:

- Generate episodes.
- Calculate returns.
- Estimate state-value functions.
- Estimate action-value functions.
- Implement Monte Carlo control.
- Evaluate learning across multiple episodes.

### Phase 5: Temporal Difference Learning

Temporal Difference learning combines ideas from Dynamic Programming and Monte Carlo methods.

The fundamental TD(0) update is:

`V(S_t) ← V(S_t) + α[R_{t+1} + γV(S_{t+1}) - V(S_t)]`

Topics:

- Bootstrapping.
- TD prediction.
- TD error.
- Online learning.
- TD(0).
- n-step TD.
- Eligibility traces.
- TD(λ).

Implementation objectives:

- Implement TD prediction.
- Compare TD and Monte Carlo learning.
- Analyze bootstrapping behaviour.
- Study bias and variance considerations.

### Phase 6: SARSA and Q-Learning

This phase focuses on temporal-difference control algorithms.

**SARSA**

SARSA is an on-policy TD control algorithm.

Update equation:

`Q(S_t,A_t) ← Q(S_t,A_t) + α[R_{t+1} + γQ(S_{t+1},A_{t+1}) - Q(S_t,A_t)]`

**Q-Learning**

Q-Learning is an off-policy TD control algorithm.

Update equation:

`Q(S_t,A_t) ← Q(S_t,A_t) + α[R_{t+1} + γ max_a Q(S_{t+1},a) - Q(S_t,A_t)]`

Topics:

- Tabular Q-Learning.
- SARSA.
- Epsilon-greedy action selection.
- Exploration-exploitation trade-off.
- On-policy vs off-policy learning.
- Learning-rate selection.
- Discount-factor analysis.
- Q-Table initialization.
- Terminal-state handling.

Implementation objectives:

- Implement Q-Learning from scratch.
- Implement SARSA from scratch.
- Train agents on discrete environments.
- Compare learned policies.
- Analyze reward convergence.
- Evaluate performance across random seeds.

### Phase 7: Deep Reinforcement Learning

Deep Reinforcement Learning combines Reinforcement Learning with neural network-based function approximation.

When state spaces become too large for tabular methods, neural networks can approximate value functions or policies.

This phase introduces PyTorch.

**Deep Q-Network (DQN)**

DQN uses a neural network to approximate the action-value function.

`Q(s, a; θ)`

Topics:

- Function approximation.
- Neural network architecture.
- Q-value prediction.
- Experience replay.
- Target networks.
- Temporal-difference targets.
- Loss functions.
- Gradient-based optimization.
- Exploration strategies.
- Training stability.

DQN target:

`y = r + γ(1 - terminated) max_a' Q_target(s', a'; θ-)`

For a truncated episode, bootstrapping should generally remain possible when the final observation is valid. Termination and truncation must therefore be handled separately.

Implementation objectives:

- Build neural networks using PyTorch.
- Implement experience replay.
- Implement target-network updates.
- Train DQN agents.
- Analyze training stability.
- Compare DQN against tabular methods where appropriate.

### Phase 8: Policy Gradient Methods

Policy Gradient methods directly optimize parameterized policies.

Topics:

- Policy parameterization.
- Stochastic policies.
- Policy optimization.
- Log-probability gradients.
- REINFORCE.
- Reward-to-go.
- Baselines.
- Advantage estimation.
- Actor-Critic methods.
- Policy-gradient variance reduction.

The fundamental policy-gradient objective is:

`J(θ) = Eπθ[G_t]`

The REINFORCE gradient estimator is:

`∇θ J(θ) = Eπθ[∇θ log πθ(A_t|S_t) G_t]`

Advanced extensions may include:

- Actor-Critic.
- Advantage Actor-Critic (A2C).
- Proximal Policy Optimization (PPO).

These extensions will be introduced when the required foundations are understood.

---

## Environment Engineering with Gymnasium

Gymnasium provides a standardized API for Reinforcement Learning environments.

It is used to create, configure, interact with, and evaluate simulation environments.

**Installation:**

`pip install "gymnasium[classic-control]"`

### Core Environment API

```python
import gymnasium as gym

env = gym.make("CartPole-v1")

observation, info = env.reset()

action = env.action_space.sample()

observation, reward, terminated, truncated, info = env.step(action)

env.close()
```

### Core Methods

| Method | Purpose |
|---|---|
| `gym.make()` | Create an environment |
| `env.reset()` | Initialize or restart an episode |
| `env.step(action)` | Execute an action |
| `env.render()` | Render the environment |
| `env.close()` | Release environment resources |

### Observation Spaces

Gymnasium supports multiple observation and action space representations:

- `Discrete`
- `Box`
- `MultiDiscrete`
- `MultiBinary`
- `Dict`
- `Tuple`

### Environment Categories

Initial experimentation will use:

- CartPole.
- FrozenLake.
- Taxi.
- MountainCar.
- Other suitable Gymnasium environments.

### Custom Environment Development

Custom environments will be developed by subclassing `gymnasium.Env`.

The environment design will define:

- State or observation representation.
- Action-space specification.
- Transition dynamics.
- Reward calculation.
- Episode termination conditions.
- Optional rendering functionality.

Custom environments will be introduced as the project progresses.

---

## Experimental Methodology

Every major algorithm implementation will follow a structured experimental workflow.

1. Mathematical formulation.
2. Algorithm derivation.
3. Implementation from scratch where applicable.
4. Environment configuration.
5. Hyperparameter selection.
6. Training execution.
7. Reward and performance monitoring.
8. Multiple-run evaluation.
9. Comparative analysis.
10. Documentation of findings.

### Reproducibility

Experiments will aim to use:

- Explicit random seeds.
- Consistent environment configurations.
- Documented hyperparameters.
- Multiple training runs.
- Recorded evaluation metrics.
- Reproducible experiment scripts.

---

## Performance Evaluation

An RL agent should not be evaluated using a single reward value or a single successful episode.

The evaluation process will include:

| Metric | Purpose |
|---|---|
| Episode Return | Total reward collected per episode |
| Average Return | Mean performance over episodes |
| Episode Length | Number of timesteps per episode |
| Success Rate | Percentage of successful episodes |
| Reward Variance | Variability in performance |
| Convergence Behaviour | Learning progression |
| Sample Efficiency | Learning relative to interactions |
| Training Stability | Consistency across training runs |

### Learning Curve Analysis

Training curves will be used to study:

- Reward progression.
- Convergence behaviour.
- Exploration and exploitation.
- Training instability.
- Performance differences between algorithms.

Matplotlib and Plotly will be used for visualization and analysis.

---

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Primary programming language |
| NumPy | Numerical computing and tabular RL implementations |
| Matplotlib | Training curves and experimental visualization |
| Plotly | Interactive visualizations |
| Gymnasium | Environment creation and interaction |
| PyTorch | Neural networks and Deep RL |
| Stable-Baselines3 | Optional benchmarking and reference implementations |
| Streamlit | Interactive agent and simulation dashboard |
| FastAPI | Optional backend API layer when required |
| Docker | Containerization and deployment |
| Git and GitHub | Version control and project documentation |

**Implementation principle:** Core classical RL algorithms will be developed independently. Stable-Baselines3 will be considered an optional benchmarking tool, not a replacement for understanding or implementing the algorithms.

---

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Pranav123221/reinforcement-learning-implementations.git

cd reinforcement-learning-implementations
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate on Windows:

```bash
.venv\Scripts\activate
```

Activate on Linux or macOS:

```bash
source .venv/bin/activate
```

### 3. Install Initial Dependencies

```bash
pip install numpy matplotlib gymnasium
```

For classic control environments:

```bash
pip install "gymnasium[classic-control]"
```

Additional dependencies, including PyTorch and Streamlit, will be introduced during the relevant development phases.

---

## Running Experiments

The following example demonstrates the initial environment-interaction workflow.

### Random Agent on CartPole

```python
import gymnasium as gym

env = gym.make("CartPole-v1")

for episode in range(5):

    observation, info = env.reset(seed=episode)

    total_reward = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):

        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = (
            env.step(action)
        )

        total_reward += reward

    print(
        f"Episode: {episode + 1}, "
        f"Total Reward: {total_reward}"
    )

env.close()
```

This is a baseline random policy, not a trained RL agent.

Future experiments will replace random action selection with learned policies and algorithm-specific action-selection mechanisms.

---

## Visualization and Analysis

Visualization will be used to understand agent behaviour and learning dynamics.

Planned visualizations include:

- Episode reward curves.
- Moving-average reward curves.
- State-value heatmaps.
- Q-Table heatmaps.
- Policy visualizations.
- Exploration-rate decay.
- Algorithm comparison charts.
- Training and evaluation performance.

The objective is to make learning behaviour measurable, interpretable, and reproducible.

---

## Flagship Project: NEXUS RL

### An Adaptive Multi-Agent Intelligence System for Autonomous Decision-Making

NEXUS RL is the proposed advanced project direction for applying the concepts developed throughout this repository.

The goal is to design a dynamic simulation environment in which autonomous agents learn to make sequential decisions under uncertainty.

Potential engineering components include:

- Custom simulation environment.
- State and action-space engineering.
- Reward-function design.
- Autonomous decision-making agents.
- Classical RL baselines.
- Deep RL agents.
- Multi-agent interaction as an advanced extension.
- Training and evaluation pipelines.
- Performance comparison.
- Interactive Streamlit dashboard.

### Proposed Technical Workflow

```text
Simulation Environment
        |
        v
State / Observation Generation
        |
        v
Agent Policy
        |
        v
Action Selection
        |
        v
Environment Transition
        |
        v
Reward Calculation
        |
        v
Learning and Policy Update
        |
        v
Agent Evaluation
        |
        v
Interactive Dashboard
```

The final environment, agent architecture, and algorithm selection will be determined after completing the foundational learning and evaluating suitable simulation problems.

The focus will be on meaningful algorithm comparisons and measurable improvements rather than implementing algorithms merely to increase the project count.

---

## Learning Outcomes

By completing the roadmap and practical implementations, the intended outcomes are:

- Understand the mathematical foundations of Reinforcement Learning.
- Formulate decision-making problems as MDPs.
- Derive and implement Bellman equations.
- Implement Dynamic Programming algorithms.
- Understand Monte Carlo and Temporal Difference learning.
- Develop SARSA and Q-Learning agents.
- Build Deep Q-Networks using PyTorch.
- Understand policy-gradient optimization.
- Design custom Gymnasium environments.
- Train and evaluate autonomous agents.
- Analyze learning curves and algorithm performance.
- Build interactive simulation-based applications.

---

## Development Roadmap

Progress will be tracked as implementations are completed and validated.

| Phase | Focus | Status |
|---|---|---|
| 01 | Gymnasium fundamentals | In progress |
| 02 | RL mathematical foundations | Planned |
| 03 | MDPs and Bellman equations | Planned |
| 04 | Dynamic Programming | Planned |
| 05 | Monte Carlo methods | Planned |
| 06 | Temporal Difference learning | Planned |
| 07 | SARSA and Q-Learning | Planned |
| 08 | Deep Q-Networks | Planned |
| 09 | Policy Gradient methods | Planned |
| 10 | NEXUS RL flagship project | Planned |

---

## References and Learning Resources

### Official Documentation

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [NumPy Documentation](https://numpy.org/doc/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/)
- [Matplotlib Documentation](https://matplotlib.org/stable/)

### Foundational Reinforcement Learning Literature

- Richard S. Sutton and Andrew G. Barto — *Reinforcement Learning: An Introduction*, Second Edition.
- [Reinforcement Learning: An Introduction — Official Book Website](http://incompleteideas.net/book/the-book-2nd.html)
- [David Silver's Reinforcement Learning Course](https://www.davidsilver.uk/teaching/)

### Additional Practical Learning

- [CampusX Reinforcement Learning Playlist](https://www.youtube.com/playlist?list=PLKnIA16_RmvbMK0_fdp0DZHZKm4Q1slAB)

---

## Repository Philosophy

**Understand the mathematics. Implement the algorithm. Train the agent. Evaluate the results. Improve the system.**

This repository is a continuously evolving engineering and learning project dedicated to understanding how autonomous agents learn, adapt, and make decisions through Reinforcement Learning.

The emphasis is on mathematical clarity, independent implementation, practical experimentation, reproducibility, and real-world engineering principles.

**Author:** Pranav Sharma

**GitHub:** [Pranav123221](https://github.com/Pranav123221)

**Primary Language:** Python

**Domain:** Reinforcement Learning | Deep Reinforcement Learning | Autonomous Agents | Decision Intelligence
