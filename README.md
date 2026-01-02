#  Using Deep Q-Networks to Learn How to Play Flappy Bird

https://github.com/user-attachments/assets/720dc067-4a9c-4e97-b3c2-92b5428ce13d

This project uses **Deep Reinforcement Learning** to train an agent to play **Flappy Bird** using **Double DQN** and **Dueling DQN** architectures.

---

##  Overview

- **Environment:** Gymnasium  
- **Algorithms:**  
  - Double Deep Q-Network (Double DQN)  
  - Dueling Deep Q-Network (Dueling DQN)

---

##  State Representation

At each timestep, the agent observes the following state features:

### Pipe Information

- Last pipe horizontal position  
- Last top pipe vertical position  
- Last bottom pipe vertical position  

- Next pipe horizontal position  
- Next top pipe vertical position  
- Next bottom pipe vertical position  

- Next-next pipe horizontal position  
- Next-next top pipe vertical position  
- Next-next bottom pipe vertical position  

### Player Information

- Player vertical position  
- Player vertical velocity  
- Player rotation  

These values are combined into a **state vector** and used as input to the neural network.

---

##  Action Space

The agent can take one of two discrete actions:

| Action | Description |
|------|------------|
| `0` | Do nothing |
| `1` | Flap |

---

## 🏆 Reward Function

The reward function encourages survival and successful navigation:

| Event | Reward |
|------|--------|
| Stay alive (per frame) | `+0.1` |
| Successfully pass a pipe | `+1.0` |
| Death | `-1.0` |
| Touch top of the screen | `-0.5` |

---

##  Training Results
<img width="640" height="480" alt="flappybird1" src="https://github.com/user-attachments/assets/299af587-9a54-45f6-8c75-cfbea4a69bc4" />

These plots demonstrate learning stability and performance improvements over time.

---

## Learning Objective

The agent learns a policy that:
- Maximizes survival time  
- Successfully passes pipes  
- Avoids unsafe vertical positions  

**Double DQN** reduces Q-value overestimation, while **Dueling DQN** improves value estimation by separating state value and advantage functions.

---


