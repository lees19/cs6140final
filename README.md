# CS6140 Final: S4 For Sequence Modelling

## Abstract
Though many variations of RNN, CNN and Transformers have been created for sequence modeling, they still struggle to model very long sequences. S4 aims to solve this issue using a state space model (SSM) in order to capture the long range dependencies. The goal of this project is to understand all of the parts of S4 and the computational tricks used which make S4 efficent. 

## Introduction
S4 is a new deep learning architecture which has the capability of efficiently modeling long sequences. Though there are specialized versions of RNNs, CNNs, and Transformers for specific tasks with long sequences, there has yet to be a general architecture for modeling these long sequences. S4 simulates a state space model (SSM) in order to handle long range dependencies. Mathematically, an SSM is of the form: 

$$x'(t) = Ax(t) + Bu(t)$$

$$y(t) = Cx(t) + Du(t)$$

Where $A, B, C, D$ are all learnable parameters. This model takes in a $1-D$ input signal $u$, projects it to an $N-D$ latent vector $x$ and then projecting the latent vector back into a $1-D$ output signal $y$. In order to apply this model to a discrete signal, the $A, B, C$ matrices need to be discretized. We will be omitting $D$, since it can just be thought of as a skip connection. 
