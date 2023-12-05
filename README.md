# CS6140 Final: S4 For Sequence Modelling

## Abstract
Though many variations of RNN, CNN and Transformers have been created for sequence modeling, they still struggle to model very long sequences. S4 aims to solve this issue using a state space model (SSM) in order to capture the long range dependencies. The goal of this project is to understand all of the parts of S4 and the computational tricks used which make S4 efficent. 

## Introduction
S4 is a new deep learning architecture which has the capability of efficiently modeling long sequences. Though there are specialized versions of RNNs, CNNs, and Transformers for specific tasks with long sequences, there has yet to be a general architecture for modeling these long sequences. S4 simulates a state space model (SSM) in order to handle long range dependencies. Mathematically, an SSM is of the form: 

$$
\begin{align}
x'(t) = Ax(t) + Bu(t)\\
y(t) = Cx(t) + Du(t)
\end{align}
$$

Where $A, B, C, D$ are all learnable parameters. This model takes in a $1-D$ input signal $u$, projects it to an $N-D$ latent vector $x$ and then projecting the latent vector back into a $1-D$ output signal $y$. In order to apply this model to a discrete signal, the $A, B, C$ matrices need to be discretized. We will be omitting $D$, since it can just be thought of as a skip connection. However, in order to apply the SSM to discrete sequences, it must be discretized. This can be done using the bilinear method: 

$$
\begin{align}
\bar{A} = (I - \Lambda/2 \cdot A)^{-1}(I + \Lambda/2 \cdot A)\\
\bar{B} = (I - \Lambda/2 \cdot A)^{-1}\Lambda B\\
\bar{C} = C
\end{align}
$$

The discrete SSM is now a sequence to sequence map with input $u_k$ to output $y_k$: 

$$
\begin{align}
x_k = \bar{A}x_{k-1} + \bar{B}u_k\\
y_k = \bar{C}x_k
\end{align}
$$

Which is very similar to one step of an RNN! This is the first advantage of S4: inference is extremely quick as the inference is handled like an RNN. 

But RNNs have the disadvantage of being hard to train because they are not easily parallelizable. To overcome this problem, let's unroll the RNN representation: 

$$
\begin{aligned}
  x_0 &= \boldsymbol{\overline{B}} u_0 &
  x_1 &= \boldsymbol{\overline{A}} \boldsymbol{\overline{B}} u_0 + \boldsymbol{\overline{B}} u_1 &
  x_2 &= \boldsymbol{\overline{A}}^2 \boldsymbol{\overline{B}} u_0 + \boldsymbol{\overline{A}} \boldsymbol{\overline{B}} u_1 + \boldsymbol{\overline{B}} u_2 & \dots
  \\
  y_0 &= \boldsymbol{\overline{C}} \boldsymbol{\overline{B}} u_0 &
  y_1 &= \boldsymbol{\overline{C}} \boldsymbol{\overline{A}} \boldsymbol{\overline{B}} u_0 + \boldsymbol{\overline{C}} \boldsymbol{\overline{B}} u_1 &
  y_2 &= \boldsymbol{\overline{C}} \boldsymbol{\overline{A}}^2 \boldsymbol{\overline{B}} u_0 + \boldsymbol{\overline{C}} \boldsymbol{\overline{A}} \boldsymbol{\overline{B}} u_1 + \boldsymbol{\overline{C}} \boldsymbol{\overline{B}} u_2
  & \dots
\end{aligned}
$$

From the above, we see it is possible to turn the sequence map into a convolution if we let the kernel or filter be equal to: 

$$
\begin{aligned}
  \boldsymbol{\overline{K}} \in \mathbb{R}^L  = (\boldsymbol{\overline{C}}\boldsymbol{\overline{B}}, \boldsymbol{\overline{C}}\boldsymbol{\overline{A}}\boldsymbol{\overline{B}}, \dots, \boldsymbol{\overline{C}}\boldsymbol{\overline{A}}^{L-1}\boldsymbol{\overline{B}})
\end{aligned}
$$

Thus, in order to compute our SSM with a convolution: 

$$
\begin{aligned}
     y_k &= \boldsymbol{\overline{C}} \boldsymbol{\overline{A}}^k \boldsymbol{\overline{B}} u_0 + \boldsymbol{\overline{C}} \boldsymbol{\overline{A}}^{k-1} \boldsymbol{\overline{B}} u_1 + \dots + \boldsymbol{\overline{C}} \boldsymbol{\overline{A}} \boldsymbol{\overline{B}} u_{k-1} + \boldsymbol{\overline{C}}\boldsymbol{\overline{B}} u_k\\
     y &= \boldsymbol{\overline{K}} \ast u
 \end{aligned}
 $$

$\overline{K}$ is a huge filter: the size of our sequence length! In other words, if our sequence lengths are 16000, then a filter of length 16000 would be needed and all the powers of $A$ up to 16000 would need to be computed. In order to curb the cost of this convolution, we use the Fast Fourier Transform (FFT) and the discrete convolution theorem. This theorem allows us to calculate the FFT of the filter and the input sequence, multiply them pointwise and then apply an inverse FFT in order to calculate the entire convolution. 

Now, that we can generate sequences with the efficency of an RNN and train our model with the benefits of CNNs, what is holding us back? It turns out with naive initialization of the SSM, the model does quite poorly on most tasks. It is not able to memorize its past even with many training steps. There is also the problem of computing the powers of $A$. If our task involves a sequence of length 16000, then we still need to compute all 16000 powers of $A$ for the filter! Let us first address the memorization problem. In order to allow our SSM to capture long range dependencies, we initialize $A$ with a HiPPO matrix. HiPPO is a class of certain $A$ matrices which allow our hidden/latent states to memorize its past. The most important HiPPO matrix for our purposes is of the form: 

$$
\begin{aligned}
  (\text{\textbf{HiPPO Matrix}})
  \qquad
  \boldsymbol{A}_{nk} =
  \begin{cases}
    (2n+1)^{1/2}(2k+1)^{1/2} & \text{if } n > k \\
    n+1 & \text{if } n = k \\
    0 & \text{if } n < k
  \end{cases}
\end{aligned}
$$

This matrix allows the SSM to compress its past history in each state enough to roughly reconstruct its past. 

Finally, the computational bottle neck of computing the powers of the $A$ matrix must be addressed. First, we assume some special structure on our $A$ matrix. More specifically, that the $A$ matrix is a diagonal plus low rank (DPLR) matrix in complex space. 
1. Assume some special structure on our $A$ matrix
2. Introduce a generating function on the coefficients of $\overline{K}$
