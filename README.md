# Stochastic Deep Q-Learning Model for Joint Optimization of 5G Cellular Network Delay and Energy Efficiency

**Abstract:** 

**Objective:** Minimize the number of active radio units and distributed units, assign RUs to DUs.

**Topology:** We have a mesh topology between the RUs and DUs, i.e., any RU can have any DU perform its higher PHY functions. There are many RUs connected to one DU. One RU cannot be connected to more than one DU.
## Model
A [stochastic Q-learning model](https://arxiv.org/abs/2405.10310) is used to solve this problem, because it can have a large discrete action space.

Q-learning requires three structures:
### 1. State Representation

| Variables                    |                                                              |
| ---------------------------- | ------------------------------------------------------------ |
| Number of radio units        | $\mathcal{N}$                                                |
| Number of user equipments    | $\mathcal{K}$                                                |
| Number of distributed units  | $L$                                                          |
| Number of traffic types      | $\mathbf{K}$                                                 |
| **Sets**                     |                                                              |
| Set of UEs                   | $\mathcal{U}=u_1, u_2, \dots, u_{\mathcal{K}}$               |
| Set of DUs                   | $\mathcal{D} = d_1,d_2,\dots,d_{L}$                          |
| Set of RUs                   | $\mathcal{R}=r_1,r_2,\dots,r_{\mathcal{N}}$                  |
| Set of delay budgets (in ms) | $\mathfrak{D} = \delta_1,\delta_2,\dots,\delta_{\mathbf{K}}$ |

The $\mathcal{K}$ RUs are connected to $L$ DUs through a mesh topology, so these RUs can choose particular DUs to perform their higher-level physical layer functions.

At time $t$, define the channel quality matrix $\mathcal{H}^{(t)} \in \mathbb{R}^{\mathcal{N} \times \mathcal{K}}$:

```math
\mathcal{H}^{(t)}=\begin{bmatrix}
h^{(t)}_{1,1} &  h^{(t)}_{1,2} &  \cdots &  h^{(t)}_{1,K}\\
h^{(t)}_{2,1} &  h^{(t)}_{2,2} &  \cdots &  h^{(t)}_{2,K}\\
\vdots  &  \vdots &  \ddots &  \vdots\\
h^{(t)}_{\mathcal{N},1} &  h^{(t)}_{\mathcal{N},2} & \cdots & h^{(t)}_{\mathcal{N},K}
\end{bmatrix}
```
where $h^{(t)}_{i,j}$ is the RSRP from radio unit $r_i$ to UE $u_j$ at time $t$.

At time $t$, define the geolocation matrix $G^{(t)} \in \mathbb{R}^{\mathcal{K} \times 2}$:

```math
G^{(t)}=\begin{bmatrix}
x_1^{(t)} & y_1^{(t)} \\
x_2^{(t)} & y_2^{(t)} \\
\vdots & \vdots  \\
x_{\mathcal{K}}^{(t)} & y_{\mathcal{K}} ^{(t)}
\end{bmatrix}
```
where $x_i^{(t)}, y_i^{(t)}$ represents the geolocation of UE $u_i$ at time $t$.

We define the delay matrix $\mathcal{P} \in \mathbb{R}^{\mathcal{N} \times L}$
```math
\mathcal{P}^{(t)}=
\begin{bmatrix}
p^{(t)}_{1,1} &  p^{(t)}_{1,2}&  \cdots&  p^{(t)}_{1,L}\\
p^{(t)}_{2,1} &  p^{(t)}_{2,2}&  \cdots&  p^{(t)}_{2,L}\\
 \vdots&  \vdots&  \ddots&  \vdots\\
 p^{(t)}_{\mathcal{N},1}&  p^{(t)}_{\mathcal{N},2}&  \cdots& p^{(t)}_{\mathcal{N}, L}
\end{bmatrix}
```
where 

```math
p^{(t)}_{i,j}=\text{propagation delay}_\text{DL} + \text{scheduling delay}_\text{DL}
```
from RU $\mathcal{R}_i$ to DU $\mathcal{D}_j$.

At time $t$, define a processing load vector $\mathcal{Z}^{(t)} \in \mathbb{R}^L$, where $\mathcal{Z}_i$ is the overall CPU utilization of du $d_i$.


We represent our state with the vector $s^{(t)}$:
```math
s^{(t)}=
[
P^{(t)},
\mathcal{H}^{(t)},
G^{(t)},
\mathcal{Z}^{(t)}
]
```
### 2. Action space
The action space here is discrete-- we only want the agent to be able to reassign RUs to DUs and either wake up RUs/DUs or put them to sleep.
```math
 \mathfrak{A}= 
 \begin{bmatrix} 
 a_{1,1} & a_{1,2}& \cdots& a_{1,L}\\ a_{2,1} & a_{2,2}& \cdots& a_{2,L}\\ \vdots& \vdots & \ddots& \vdots\\ a_{\mathcal{N},1} & a_{\mathcal{N},2} & \cdots & a_{\mathcal{N},L} 
 \end{bmatrix}
```
 where 
 ```math
 a_{i,j}=\begin{cases} 1 & \text{RU i is assisted by DU j }\ 0 & \text{otherwise} \end{cases}
 ```

Define a vector in $\mathcal{B} \in \mathbb{R}^{\mathcal{N}}$, which represents the sleep status of a given RU:

```math
\mathcal{B}=[b_1,b_2,\cdots,b_{\mathcal{N}}]
```

where
```math
b_i=\begin{cases}
1 & \text{RU } r_i \text{ is active}\\
0 & \text{RU } r_i \text{ is asleep}
\end{cases}
```
Define a vector $\mathcal{F} \in \mathbb{R}^{L}$, which represents the sleep status of a given DU:
```math
\mathcal{F}=[f_1,f_2,\cdots,f_L]
```

where
```math
f_i=\begin{cases}
1 & \text{DU } d_i \text{ is active}\\
0 & \text{DU } d_i \text{ is asleep}
\end{cases}
```
now we can represent our action space $\mathcal{A}$ as a vector:

```math
\mathcal{A}=[\mathfrak{A},\mathcal{B},\mathcal{F}, \emptyset]
```

where $\emptyset$ represents no action taken.

### 3. Reward function
Elaborate: Will we make a decision once every time interval or at every time step?

Discourage switching DUs and RUs too frequently.

Define functions
```math
C_\text{UE}(r_i) \triangleq \text{number of UEs connected to RU } r_i \in \mathcal{R}

```

```math
C_\text{RU}(d_i) \triangleq \text{number of RUs connected to DU } d_i \in \mathcal{D}
```

```math
R(u_i) \triangleq \text{RU serving UE $u_i$}
```
At time $t$, define $\mathfrak{R}$ to be our reward function:
```math
\mathfrak{R}^{(t)}=\alpha(\sum_{u_i\in\mathcal{U}}\mathfrak{R}^{(t)}_\text{RSRP}(u_i)+\sum_{d_i\in\mathcal{D}}\mathfrak{R}^{(t)}_\text{DU capacity}(d_i)+\sum_{r_i\in\mathcal{R}}\mathfrak{R}^{(t)}_\text{RU capacity}(r_i) + |\mathcal{R}_\text{sleep}| + |\mathcal{D}_\text{sleep}|)-\beta(\mathfrak{R}^{(t)}_\text{average fronthaul delay})
```
where
```math
\mathfrak{R}^{(t)}_\text{RSRP}(u_i) \triangleq \begin{cases}
1 & \text{RSRP}(u_i, R(u_i)) \geq \delta_\text{threshold}   \\
-2 & \text{otherwise}
\end{cases}
```
```math
\mathfrak{R}^{(t)}_\text{DU capacity}(d_i) \triangleq \begin{cases}
1 & C_\text{RU}(d_i) \leq \phi_\text{DU}\\
-2 & \text{otherwise}
\end{cases}
```
```math
\mathfrak{R}^{(t)}_\text{RU capacity}(r_i) \triangleq \begin{cases}
1 & C_\text{UE}(r_i) \leq \phi_\text{RU}\\
-2 & \text{otherwise}
\end{cases}
```
and
```math
\delta_\text{threshold} \in [-100, -80]
```
```math
\alpha \in (0,1]
```
```math
\beta \in (0,1]
```
```math
\phi_\text{DU} \in \mathbb{Z}, \phi_\text{DU} > 0
```
```math
\phi_\text{RU} \in \mathbb{Z}, \phi_\text{RU} > 0
```

## Testbed & Implementation
We will be using a custom testbed environment built in Python (can be found in `src`) to train and test the model using these parameters:

### Experiment
#### Parameters
- $n=16$ RUs evenly spaced at 100m
- $m=4$ DUs
- $k=50$ UEs moving in a random walk
- 1000m by 1000m simulation area

#### Implementation
We use PyTorch to implement the stochastic DQN. We use $I \triangleq NL + NK + 2K + L$ nodes in the input layer, $A \triangleq NL+L+N+1$ nodes on the output layer and 2 layers of $\frac{2}{3}I+A$ nodes in the two hidden layers. For further improvements we plan on using a multiple input neural network.

#### Other Models
- DQN model found in [Wang et al.](https://ieeexplore.ieee.org/document/10942980)
- Base simulation, no model