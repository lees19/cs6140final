# CS6140 Final: S4 For Sequence Modelling

## Abstract
Though many variations of RNN, CNN and Transformers have been created for sequence modeling, they still struggle to model very long sequences. S4 aims to solve this issue using a state space model (SSM) in order to capture the long range dependencies. The goal of this project is to understand all of the parts of S4 and the computational tricks used which make S4 efficent. 

## Introduction
S4 is a new deep learning architecture which has the capability of efficiently modeling long sequences. Though there are specialized versions of RNNs, CNNs, and Transformers for specific tasks with long sequences, there has yet to be a general architecture for modeling these long sequences. S4 simulates a state space model (SSM) in order to handle long range dependencies. Mathematically, an SSM is of the form: 

$$x'(t) = Ax(t) + Bu(t)$$

$$y(t) = Cx(t) + Du(t)$$

Where $A, B, C, D$ are all learnable parameters. This model takes in a $1-D$ input signal $u$, projects it to an $N-D$ latent vector $x$ and then projecting the latent vector back into a $1-D$ output signal $y$. In order to apply this model to a discrete signal, the $A, B, C$ matrices need to be discretized. We will be omitting $D$, since it can just be thought of as a skip connection. However, in order to apply the SSM to discrete sequences, it must be discretized. This can be done using the bilinear method: 

$$\bar{A} = (I - \Lambda/2 \cdot A)^{-1}(I + \Lambda/2 \cdot A)$$

$$\bar{B} = (I - \Lambda/2 \cdot A)^{-1}\Lambda B$$

$$\bar{C} = C$$

The discrete SSM is now a sequence to sequence map with input $u_k$ to output $y_k$: 

$$x_k = \bar{A}x_{k-1} + \bar{B}u_k$$

$$y_k = \bar{C}x_k$$

Which is very similar to one step of an RNN! This is the first advantage of S4: inference is extremely quick as the inference is handled like an RNN. 

But RNNs have the disadvantage of being hard to train because they are not easily parallelizable. To overcome this problem, let's unroll the RNN representation: 

$$x_0 = \bar{B}u_0$$

$$y_0 = \bar{C}\bar{B}u_0$$

and again: 

$$x_1 = \bar{AB}u_0 + \bar{B}u_1$$

$$y_1 = \bar{CAB}u_0 + \bar{CB}u_1$$

one more time: 

$$x_2 = \bar{A}^2\bar{B}u_0 + \bar{AB}u_1 + \bar{B}u_2$$
$$y_2 = \bar{CA}^2\bar{B}u_0 + \bar{CAB}u_1 + \bar{CB}u_2$$


