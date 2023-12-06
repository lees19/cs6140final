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

Where $A, B, C, D$ are all learnable parameters. This model takes in a $1-D$ input signal $u$, projects it to an $N-D$ latent vector $x$ and then projecting the latent vector back into a $1-D$ output signal $y$. In order to apply this model to a discrete signal, the $A, B, C$ matrices need to be discretized. $D$ will be omitted, since it can just be thought of as a skip connection. However, in order to apply the SSM to discrete sequences, it must be discretized. This can be done using the bilinear method: 

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

From the above, it is possible to turn the sequence map into a convolution if the kernel or filter be equal to: 

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

$\overline{K}$ is a huge filter: the size of our sequence length! In other words, if our sequence lengths are 16000, then a filter of length 16000 would be needed and all the powers of $A$ up to 16000 would need to be computed. In order to curb the cost of this convolution, the Fast Fourier Transform (FFT) and the discrete convolution theorem. This theorem allows us to calculate the FFT of the filter and the input sequence, multiply them pointwise and then apply an inverse FFT in order to calculate the entire convolution. 

Now, that we can generate sequences with the efficency of an RNN and train our model with the benefits of CNNs, what is holding us back? It turns out with naive initialization of the SSM, the model does quite poorly on most tasks. It is not able to memorize its past even with many training steps. There is also the problem of computing the powers of $A$. If our task involves a sequence of length 16000, then we still need to compute all 16000 powers of $A$ for the filter! Let us first address the memorization problem. In order to allow our SSM to capture long range dependencies, $A$ is initialized with a HiPPO matrix. HiPPO is a class of certain $A$ matrices which allow our hidden/latent states to memorize its past. The most important HiPPO matrix for our purposes is of the form: 

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

Finally, the computational bottle neck of computing the powers of the $A$ matrix must be addressed. First, we assume our SSM has a diagonal plus low rank (DPLR) structure. This means that we assume our $A$ matrix is of the form $\Lambda - PQ^*$ for some diagonal $\Lambda$ and matrices $P, Q, B, C\in \mathbb{C}^{N\times 1}$. Without loss of genearlity, assume these matrices are vectors. With this assumption, the model is sped up in three steps. From the paper: 
> 1. Instead of computing $\overline{K}$ directly, we compute ts spectrum by evaluating its truncated generating function. This now involves a matrix inverse instead of power.
> 2. We show that the diagonal matrix case is equivalent to the computation of a Cauchy Kernel
> 3. We show the low-rank term can now be corrected by applying the Woodbury identity which reduces $(\Lambda + PQ^*)^{-1}$ in terms of $\Lambda^{-1}$, truly reducing to the diagonal case.

Step 1: 

The truncated SSM generating function at node $z$ with truncation $L$ is: 

$$
\hat{\mathcal{K}}_L(z; \boldsymbol{\overline{A}}, \boldsymbol{\overline{B}}, \boldsymbol{\overline{C}}) \in \mathbb{C} := \sum _{i=0}^{L-1} \boldsymbol{\overline{C}} \boldsymbol{\overline{A}}^i \boldsymbol{\overline{B}} z^i
$$

The generating function turns the SSM from the time domain to the frequency domain. This is also called the z-transform. From the generating function, the filter's discrete fourier transform can be obtained from evaluations of the z-transform at the roots of unity $\Omega = \{ \exp(2\pi \frac{k}{L} : k \in [L] )\}$ and finally, apply an inverse FFT. More importantly, the matrix power can be replaced with an inverse in the generating function: 

$$
\hat{\mathcal{K}}_L(z) = \sum _{i=0}^{L-1} \boldsymbol{\overline{C}} \boldsymbol{\overline{A}}^i \boldsymbol{\overline{B}} z^i = \boldsymbol{\overline{C}} (\boldsymbol{I} - \boldsymbol{\overline{A}}^L z^L) (\boldsymbol{I} - \boldsymbol{\overline{A}} z)^{-1} \boldsymbol{\overline{B}} = \boldsymbol{\widetilde{C}}  (\boldsymbol{I} - \boldsymbol{\overline{A}} z)^{-1} \boldsymbol{\overline{B}}
$$

Step 2: 

Next we assume some special structure on $A$ in order to speed up the calculation of the inverse of $A$. After substituting in our discrete SSM matrices and letting $A = \Lambda$ the authors show that the generating function can be written as: 

$$ \begin{aligned}
\boldsymbol{\hat{K}}_{\boldsymbol{\Lambda}}(z) & = c(z) \sum_i \cdot \frac{\boldsymbol{\widetilde{C}}_i \boldsymbol{B}_i} {(g(z) - \boldsymbol{\Lambda}_i)} = c(z) \cdot k _{z, \boldsymbol{\Lambda}}(\boldsymbol{\widetilde{C}}, \boldsymbol{B}) 
\end{aligned}
$$
 
where $c$ is a constant, and $g$ is a function of $z$. Thus, the inverse has been replaced with a dot product. 

Step 3: 

Finally, in order to we relax the diagonal assumption and assume that the $A$ matrix is DPLR. Using the Woodbury Identity, the authors write the inverse of our DPLR matrix in the form: 

$$ 
\begin{aligned}
(\boldsymbol{\Lambda} + \boldsymbol{P} \boldsymbol{Q}^* )^{-1} &= \boldsymbol{\Lambda}^{-1} - \boldsymbol{\Lambda}^{-1} \boldsymbol{P} (1 + \boldsymbol{Q}^* \boldsymbol{\Lambda}^{-1} \boldsymbol{P})^{-1}\boldsymbol{Q}^* \boldsymbol{\Lambda}^{-1}
\end{aligned}
$$

Finally, the authors show that the generating function for a DPLR SSM can be written as: 

$$ 
\begin{aligned} 
\boldsymbol{\hat{K}}_{DPLR}(z) &= c(z) [ k _{z, \Lambda}(\boldsymbol{\widetilde{C}}, \boldsymbol{\boldsymbol{B}}) - k _{z, \Lambda}(\boldsymbol{\widetilde{C}}, \boldsymbol{\boldsymbol{P}}) (1 + k _{z, \Lambda}(\boldsymbol{q^*}, \boldsymbol{\boldsymbol{P}}) )^{-1}  k _{z, \Lambda}(\boldsymbol{q ^ *}, \boldsymbol{\boldsymbol{B}}) ]
\end{aligned}
$$

With the DPLR assumption, it is possible to compute the discretized $A$ matrix without directly inverting the whole matrix. The authors show: 

$$
\begin{align*}
  \boldsymbol{I} + \frac{\Delta}{2} \boldsymbol{A}
  &= \boldsymbol{I} + \frac{\Delta}{2} (\boldsymbol{\Lambda} - \boldsymbol{P} \boldsymbol{Q} ^ * )
  \\ &= \frac{\Delta}{2} \left[ \frac{2}{\Delta}\boldsymbol{I}+ (\boldsymbol{\Lambda} - \boldsymbol{P} \boldsymbol{Q}^* ) \right]
  \\ &= \frac{\Delta}{2} \boldsymbol{A_0}
\end{align*}
$$

we let $A_0$ be equal to the terms in the final brackets. For the second term:  

$$
\begin{align*}
  \left( \boldsymbol{I} - \frac{\Delta}{2} \boldsymbol{A} \right)^{-1}
  &=
  \left( \boldsymbol{I} - \frac{\Delta}{2} (\boldsymbol{\Lambda} - \boldsymbol{P} \boldsymbol{Q} ^ * ) \right)^{-1}
  \\&=
  \frac{2}{\Delta} \left[ \frac{2}{\Delta} - \boldsymbol{\Lambda} + \boldsymbol{P} \boldsymbol{Q} ^ * \right]^{-1}
  \\&=
  \frac{2}{\Delta} \left[ \boldsymbol{D} - \boldsymbol{D} \boldsymbol{P} \left( 1 + \boldsymbol{Q} ^ * \boldsymbol{D} \boldsymbol{P} \right)^{-1} \boldsymbol{Q} ^ * \boldsymbol{D} \right]
  \\&= \frac{2}{\Delta} \boldsymbol{A_1}
\end{align*}
$$

where $D = (\frac{2}{\Delta} - \Lambda)^{-1}$ and $A_1$ is defined by the terms in the final bracket. 

Finally, our discrete SSM becomes

$$
\begin{align*}
  x_{k} &= \boldsymbol{\overline{A}} x_{k-1} + \boldsymbol{\overline{B}} u_k \\
  &= \boldsymbol{A_1} \boldsymbol{A_0} x_{k-1} + 2 \boldsymbol{A_1} \boldsymbol{B} u_k \\
  y_k &= \boldsymbol{C} x_k
\end{align*}
$$

To finish off the implementation of S4, we must turn the HiPPO matrix into a DPLR matrix. HiPPO must first be written out to be a normal plus low rank (NPLR) matrix. From the NPLR representation, the normal matrix can be diagonalized, and then the diagonal matrix extracted, giving us the $\Lambda$ used in the DPLR matrix!

$$
\boldsymbol{A} = \boldsymbol{V} \boldsymbol{\Lambda} \boldsymbol{V} ^ * - \boldsymbol{P} \boldsymbol{Q}^\top = \boldsymbol{V} \left( \boldsymbol{\Lambda} - \boldsymbol{V} ^ * \boldsymbol{P} (\boldsymbol{V} ^ * \boldsymbol{Q}) ^ * \right) \boldsymbol{V} ^ *
$$

