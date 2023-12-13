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

Where $A, B, C, D$ are all learnable parameters. This model takes in a $1-D$ input signal $u$, projects it to an $N-D$ latent vector $x$ and then projecting the latent vector back into a $1-D$ output signal $y$. In order to apply this model to a discrete signal, the $A, B, C$ matrices need to be discretized. $D$ will be omitted, since it can just be thought of as a skip connection. However, in order to apply the SSM to discrete sequences, it must be discretized. This can be done using the bilinear method for some step size $\Delta$: 

$$
\begin{align}
\bar{A} = (I - \Delta/2 \cdot A)^{-1}(I + \Delta/2 \cdot A)\\
\bar{B} = (I - \Delta/2 \cdot A)^{-1}\Delta B\\
\bar{C} = C
\end{align}
$$

```
def discretize(A, B, C, step):
    I = np.eye(A.shape[0])
    BL = inv(I - (step / 2.0) * A)
    Ab = BL @ (I + (step / 2.0) * A)
    Bb = (BL * step) @ B
    return Ab, Bb, C
```

The discrete SSM is now a sequence to sequence map with input $u_k$ to output $y_k$: 

$$
\begin{align}
x_k = \bar{A}x_{k-1} + \bar{B}u_k\\
y_k = \bar{C}x_k
\end{align}
$$

```
def scan_SSM(Ab, Bb, Cb, u, x0):
    def step(x_k_1, u_k):
        x_k = Ab @ x_k_1 + Bb @ u_k
        y_k = Cb @ x_k
        return x_k, y_k

    return jax.lax.scan(step, x0, u)
```

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

```
def causal_convolution(u, K, nofft=False):
    if nofft:
        return convolve(u, K, mode="full")[: u.shape[0]]
    else:
        assert K.shape[0] == u.shape[0]
        ud = np.fft.rfft(np.pad(u, (0, K.shape[0])))
        Kd = np.fft.rfft(np.pad(K, (0, u.shape[0])))
        out = ud * Kd
        return np.fft.irfft(out)[: u.shape[0]]
```

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

```
def make_HiPPO(N):
    P = np.sqrt(1 + 2 * np.arange(N))
    A = P[:, np.newaxis] * P[np.newaxis, :]
    A = np.tril(A) - np.diag(np.arange(N))
    return -A
```

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

```
@jax.jit
def cauchy(v, omega, lambd):
    """Cauchy matrix multiplication: (n), (l), (n) -> (l)"""
    cauchy_dot = lambda _omega: (v / (_omega - lambd)).sum()
    return jax.vmap(cauchy_dot)(omega)


def kernel_DPLR(Lambda, P, Q, B, C, step, L):
    # Evaluate at roots of unity
    # Generating function is (-)z-transform, so we evaluate at (-)root
    Omega_L = np.exp((-2j * np.pi) * (np.arange(L) / L))

    aterm = (C.conj(), Q.conj())
    bterm = (B, P)

    g = (2.0 / step) * ((1.0 - Omega_L) / (1.0 + Omega_L))
    c = 2.0 / (1.0 + Omega_L)

    # Reduction to core Cauchy kernel
    k00 = cauchy(aterm[0] * bterm[0], g, Lambda)
    k01 = cauchy(aterm[0] * bterm[1], g, Lambda)
    k10 = cauchy(aterm[1] * bterm[0], g, Lambda)
    k11 = cauchy(aterm[1] * bterm[1], g, Lambda)
    atRoots = c * (k00 - k01 * (1.0 / (1.0 + k11)) * k10)
    out = np.fft.ifft(atRoots, L).reshape(L)
    return out.real
```

With the DPLR assumption, it is possible to compute the discretized $A$ matrix without directly inverting the whole matrix. The authors show: 

$$
\begin{align*}
  \boldsymbol{I} + \frac{\Delta}{2} \boldsymbol{A}
  &= \boldsymbol{I} + \frac{\Delta}{2} (\boldsymbol{\Lambda} - \boldsymbol{P} \boldsymbol{Q} ^ * )\\ 
  &= \frac{\Delta}{2} \left[ \frac{2}{\Delta}\boldsymbol{I}+ (\boldsymbol{\Lambda} - \boldsymbol{P} \boldsymbol{Q}^* ) \right]\\
   &= \frac{\Delta}{2} \boldsymbol{A_0}
\end{align*}
$$

we let $A_0$ be equal to the terms in the final brackets. For the second term:  

$$
\begin{align*}
  \left( \boldsymbol{I} - \frac{\Delta}{2} \boldsymbol{A} \right)^{-1}
  &=
  \left( \boldsymbol{I} - \frac{\Delta}{2} (\boldsymbol{\Lambda} - \boldsymbol{P} \boldsymbol{Q} ^ * ) \right)^{-1}\\
  &=
  \frac{2}{\Delta} \left[ \frac{2}{\Delta} - \boldsymbol{\Lambda} + \boldsymbol{P} \boldsymbol{Q} ^ * \right]^{-1}\\
  &=
  \frac{2}{\Delta} \left[ \boldsymbol{D} - \boldsymbol{D} \boldsymbol{P} \left( 1 + \boldsymbol{Q} ^ * \boldsymbol{D} \boldsymbol{P} \right)^{-1} \boldsymbol{Q} ^ * \boldsymbol{D} \right]\\
  &= \frac{2}{\Delta} \boldsymbol{A_1}
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

```
def discrete_DPLR(Lambda, P, Q, B, C, step, L):
    # Convert parameters to matrices
    B = B[:, np.newaxis]
    Ct = C[np.newaxis, :]

    N = Lambda.shape[0]
    A = np.diag(Lambda) - P[:, np.newaxis] @ Q[:, np.newaxis].conj().T
    I = np.eye(N)

    # Forward Euler
    A0 = (2.0 / step) * I + A

    # Backward Euler
    D = np.diag(1.0 / ((2.0 / step) - Lambda))
    Qc = Q.conj().T.reshape(1, -1)
    P2 = P.reshape(-1, 1)
    A1 = D - (D @ P2 * (1.0 / (1 + (Qc @ D @ P2))) * Qc @ D)

    # A bar and B bar
    Ab = A1 @ A0
    Bb = 2 * A1 @ B

    # Recover Cbar from Ct
    Cb = Ct @ inv(I - matrix_power(Ab, L)).conj()
    return Ab, Bb, Cb.conj()
```

To finish off the implementation of S4, we must turn the HiPPO matrix into a DPLR matrix. HiPPO must first be written out to be a normal plus low rank (NPLR) matrix. From the NPLR representation, the normal matrix can be diagonalized, and then the diagonal matrix extracted, giving us the $\Lambda$ used in the DPLR matrix!

$$
\boldsymbol{A} = \boldsymbol{V} \boldsymbol{\Lambda} \boldsymbol{V} ^ * - \boldsymbol{P} \boldsymbol{Q}^\top = \boldsymbol{V} \left( \boldsymbol{\Lambda} - \boldsymbol{V} ^ * \boldsymbol{P} (\boldsymbol{V} ^ * \boldsymbol{Q}) ^ * \right) \boldsymbol{V} ^ *
$$

```
def make_NPLR_HiPPO(N):
    # Make -HiPPO
    nhippo = make_HiPPO(N)

    # Add in a rank 1 term. Makes it Normal.
    P = np.sqrt(np.arange(N) + 0.5)

    # HiPPO also specifies the B matrix
    B = np.sqrt(2 * np.arange(N) + 1.0)
    return nhippo, P, B

# After extracting the normal part, we can diagonalize to get out the DPLR terms.
# Because the normal part is actually skew-symmetric, we can extract the real and complex parts of $\boldsymbol{\Lambda}$ separately.
# This serves two purposes. First, this gives us finer-grained control over the real and imaginary parts, which can be used to improve stability. Second, this lets us use more powerful diagonalization algorithms for [Hermitian matrices](https://en.wikipedia.org/wiki/Hermitian_matrix) - in fact, the current version of JAX does not support GPU diagonalization for non-Hermitian matrices!

def make_DPLR_HiPPO(N):
    """Diagonalize NPLR representation"""
    A, P, B = make_NPLR_HiPPO(N)

    S = A + P[:, np.newaxis] * P[np.newaxis, :]

    # Check skew symmetry
    S_diag = np.diagonal(S)
    Lambda_real = np.mean(S_diag) * np.ones_like(S_diag)
    # assert np.allclose(Lambda_real, S_diag, atol=1e-3)

    # Diagonalize S to V \Lambda V^*
    Lambda_imag, V = eigh(S * -1j)

    P = V.conj().T @ P
    B = V.conj().T @ B
    return Lambda_real + 1j * Lambda_imag, P, B, V
```

Putting everything together, we finally have our S4 layer!: 
```
class S4Layer(nn.Module):
    N: int
    l_max: int
    decode: bool = False

    # Special parameters with multiplicative factor on lr and no weight decay (handled by main train script)
    lr = {
        "Lambda_re": 0.1,
        "Lambda_im": 0.1,
        "P": 0.1,
        "B": 0.1,
        "log_step": 0.1,
    }

    def setup(self):
        # Learned Parameters (C is complex!)
        init_A_re, init_A_im, init_P, init_B = hippo_initializer(self.N)
        self.Lambda_re = self.param("Lambda_re", init_A_re, (self.N,))
        self.Lambda_im = self.param("Lambda_im", init_A_im, (self.N,))
        # Ensure the real part of Lambda is negative
        # (described in the SaShiMi follow-up to S4)
        self.Lambda = np.clip(self.Lambda_re, None, -1e-4) + 1j * self.Lambda_im
        self.P = self.param("P", init_P, (self.N,))
        self.B = self.param("B", init_B, (self.N,))
        # C should be init as standard normal
        # This doesn't work due to how JAX handles complex optimizers https://github.com/deepmind/optax/issues/196
        # self.C = self.param("C", normal(stddev=1.0, dtype=np.complex64), (self.N,))
        self.C = self.param("C", normal(stddev=0.5**0.5), (self.N, 2))
        self.C = self.C[..., 0] + 1j * self.C[..., 1]
        self.D = self.param("D", nn.initializers.ones, (1,))
        self.step = np.exp(self.param("log_step", log_step_initializer(), (1,)))

        if not self.decode:
            # CNN mode, compute kernel.
            self.K = kernel_DPLR(
                self.Lambda,
                self.P,
                self.P,
                self.B,
                self.C,
                self.step,
                self.l_max,
            )

        else:
            # RNN mode, discretize

            # Flax trick to cache discrete form during decoding.
            def init_discrete():
                return discrete_DPLR(
                    self.Lambda,
                    self.P,
                    self.P,
                    self.B,
                    self.C,
                    self.step,
                    self.l_max,
                )

            ssm_var = self.variable("prime", "ssm", init_discrete)
            if self.is_mutable_collection("prime"):
                ssm_var.value = init_discrete()
            self.ssm = ssm_var.value

            # RNN Cache
            self.x_k_1 = self.variable(
                "cache", "cache_x_k", np.zeros, (self.N,), np.complex64
            )

    def __call__(self, u):
        # This is identical to SSM Layer
        if not self.decode:
            # CNN Mode
            return causal_convolution(u, self.K) + self.D * u
        else:
            # RNN Mode
            x_k, y_s = scan_SSM(*self.ssm, u[:, np.newaxis], self.x_k_1.value)
            if self.is_mutable_collection("cache"):
                self.x_k_1.value = x_k
            return y_s.reshape(-1).real + self.D * u

```

To summarize: S4 is a deep neural network architecture based on the SSM with the $A$ matrix initialized as a HiPPO matrix. However, naively applying the SSM is quite computationally expensive. The authors provide a way of reducing the number of calculations done by reducing the calculation of successive powers of $A$ by assuming certain structures of the $A$ matrix. This allows us to replace the successive powers of $A$ with inverses of the diagonal obtained from the structure placed on $A$. 

## Setup

Like self attention, S4 is a layer and can be put into a sequence block with dropout and a linear projection: 

```
class SequenceBlock(nn.Module):
    layer_cls: nn.Module
    layer: dict  # Hyperparameters of inner layer
    dropout: float
    d_model: int
    prenorm: bool = True
    glu: bool = True
    training: bool = True
    decode: bool = False

    def setup(self):
        self.seq = self.layer_cls(**self.layer, decode=self.decode)
        self.norm = nn.LayerNorm()
        self.out = nn.Dense(self.d_model)
        if self.glu:
            self.out2 = nn.Dense(self.d_model)
        self.drop = nn.Dropout(
            self.dropout,
            broadcast_dims=[0],
            deterministic=not self.training,
        )

    def __call__(self, x):
        skip = x
        if self.prenorm:
            x = self.norm(x)
        x = self.seq(x)
        x = self.drop(nn.gelu(x))
        if self.glu:
            x = self.out(x) * jax.nn.sigmoid(self.out2(x))
        else:
            x = self.out(x)
        x = skip + self.drop(x)
        if not self.prenorm:
            x = self.norm(x)
        return x
```

These blocks are then stacked on top of each other to form the layers of our neural network: 

```
class Embedding(nn.Embed):
    num_embeddings: int
    features: int

    @nn.compact
    def __call__(self, x):
        y = nn.Embed(self.num_embeddings, self.features)(x[..., 0])
        return np.where(x > 0, y, 0.0)


class StackedModel(nn.Module):
    layer_cls: nn.Module
    layer: dict  # Extra arguments to pass into layer constructor
    d_output: int
    d_model: int
    n_layers: int
    prenorm: bool = True
    dropout: float = 0.0
    embedding: bool = False  # Use nn.Embed instead of nn.Dense encoder
    classification: bool = False
    training: bool = True
    decode: bool = False  # Probably should be moved into layer_args

    def setup(self):
        if self.embedding:
            self.encoder = Embedding(self.d_output, self.d_model)
        else:
            self.encoder = nn.Dense(self.d_model)
        self.decoder = nn.Dense(self.d_output)
        self.layers = [
            SequenceBlock(
                layer_cls=self.layer_cls,
                layer=self.layer,
                prenorm=self.prenorm,
                d_model=self.d_model,
                dropout=self.dropout,
                training=self.training,
                decode=self.decode,
            )
            for _ in range(self.n_layers)
        ]

    def __call__(self, x):
        if not self.classification:
            if not self.embedding:
                x = x / 255.0  # Normalize
            if not self.decode:
                x = np.pad(x[:-1], [(1, 0), (0, 0)])
        x = self.encoder(x)
        for layer in self.layers:
            x = layer(x)
        if self.classification:
            x = np.mean(x, axis=0)
        x = self.decoder(x)
        return nn.log_softmax(x, axis=-1)

```

However, S4 operates on scalars, meaning it cannot take in more than one feature. To fix this, we simply create one stacked copy for each feature in the dataset: 
```
def cloneLayer(layer):
    return nn.vmap(
        layer,
        in_axes=1,
        out_axes=1,
        variable_axes={"params": 1, "cache": 1, "prime": 1},
        split_rngs={"params": True},
    )
```

Finally, to add the batch dimension: 
```
BatchStackedModel = nn.vmap(
    StackedModel,
    in_axes=0,
    out_axes=0,
    variable_axes={"params": None, "dropout": None, "cache": 0, "prime": None},
    split_rngs={"params": False, "dropout": True},
)
```

The dataset we will be using is the sequential MNIST data set. This is similar to the MNIST handwritten digit dataset, but the goal is to generate the rest of the digit given the first $N$ (in our case 300) pixels called context. Each data point is a black and white 28x28 image of a pixel, where the goal is to classify the intensity of the pixel (output dimension 256). Each pixel of each image is fed into the model sequentially, rather than the whole image. I am going to be using the S4 layer on 10 epochs, a batch size of 128, a hidden dimension of 64 (for the SSM), and a model dimension of 128 (for the linear unit after the SSM). I will be running this exmaple on Google Colab using their T4 GPU. 

```
!python -m train dataset=mnist layer=s4 train.epochs=10 train.bsz=128 train.lr=5e-3 train.lr_schedule=true model.layer.N=64 model.d_model=128 model.n_layers=4 model.dropout=0.0 train.weight_decay=0.05 model.prenorm=true model.embedding=true train.sample=308 
```

## Results
![im0 117](https://github.com/lees19/cs6140final/assets/43870417/64a9eabd-be3a-461f-a5e7-9bcf2e6e615d)
![im0 123](https://github.com/lees19/cs6140final/assets/43870417/467110b7-406a-49d3-9411-625974da427f)
![im0 93](https://github.com/lees19/cs6140final/assets/43870417/7ec51820-97e6-4808-9af3-7e363c83ab71)
![im0 81](https://github.com/lees19/cs6140final/assets/43870417/f2f13903-c4ec-4be0-8138-2a3765c627a8)
![im0 125](https://github.com/lees19/cs6140final/assets/43870417/49c44849-50c6-44fa-bfc4-b29dc0fe0d49)
![im0 120](https://github.com/lees19/cs6140final/assets/43870417/779bf0a1-283a-4afb-b648-808e8a975c5a)

After training the model above for 10 epochs and a model size of about 500k parameters, we obtained a best test loss of 0.55288 and a best test accuracy of 0.8922. From the generated images above, we see the model, even with only 10 epochs, is able to generate some decent images. 

## Discussion
The loss of the small model and the accuracy are quite promising. A model with less than half a million parameters and only 10 epochs, this architecture clearly has a lot of potential for sequence modeling tasks. Indeed, in the original S4 paper, the S4 model does quite well, reaching SoTA in many different long range dependency tasks. The Annotated S4 paper also cites the [MNIST generation task](https://paperswithcode.com/sota/image-generation-on-mnist) in which S4 would easily be SoTA. 

## Conclusion
In this project, I attempted to understand the inner workings of S4 and why it is efficient. I have also taken the annotated S4 implementation and cleaned up the code to specifically run with S4 and ran a small version of the model on the sequential MNIST dataset. I have used the model to generate some example images given a context of length 308. 

## References

[Original S4 Paper](https://arxiv.org/abs/2111.00396) 
[LSSL Paper, helps with understanding S4](https://arxiv.org/abs/2110.13985)
[Annotated S4](https://srush.github.io/annotated-s4/)
