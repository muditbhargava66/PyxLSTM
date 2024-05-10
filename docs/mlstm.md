# mLSTM: Matrix Long Short-Term Memory

The mLSTM (Matrix Long Short-Term Memory) is a variant of the LSTM architecture that introduces a matrix memory structure and a covariance update rule. It enhances the traditional LSTM by increasing the storage capacity and enabling parallel computation.

## Architecture

The mLSTM architecture consists of the following components:

- Input Gate ($i$): Controls the flow of input information into the memory cell.
- Forget Gate ($f$): Determines the amount of information to retain or forget from the previous memory cell state.
- Output Gate ($o$): Controls the flow of information from the memory cell to the hidden state.
- Cell State ($C$): Stores the long-term memory information as a matrix.
- Key ($k$) and Value ($v$): Represent the input information to be stored in the memory matrix.
- Query ($q$): Used to retrieve information from the memory matrix.

The mLSTM replaces the scalar memory cell state with a matrix memory structure ($C$). The input information is represented as key-value pairs, where the keys are used to update the memory matrix, and the values are stored in the matrix. The query is used to retrieve information from the memory matrix.

## Equations

The equations governing the mLSTM are as follows:

- Cell State Update:
  $C_t = f_t C_{t-1} + i_t v_t k_t^T$

- Normalizer State Update:
  $n_t = f_t n_{t-1} + i_t k_t$

- Hidden State Update:
  $h_t = o_t \odot \tanh\left(\frac{C_t q_t}{\max\left(\|n_t^T q_t\|, 1\right)}\right)$

- Input Gate:
  $i_t = \exp\left(w_i^T x_t\right)$

- Forget Gate:
  $f_t = \exp\left(w_f^T x_t\right)$

- Output Gate:
  $o_t = \sigmoid\left(W_o h_t\right)$

- Key:
  $k_t = W_k x_t$

- Value:
  $v_t = W_v x_t$

- Query:
  $q_t = W_q x_t$

where:
- $x_t$ is the input vector at time step $t$
- $W_*$ are weight matrices
- $w_*$ are weight vectors

## Parallelization

One of the key advantages of the mLSTM is its ability to perform parallel computations. The matrix memory structure allows for efficient parallel updates and retrieval of information.

The mLSTM can be implemented using matrix operations, enabling parallelization on hardware accelerators such as GPUs. This makes the mLSTM suitable for large-scale sequence modeling tasks.

## Usage

To use the mLSTM in your project, you can import the `mLSTM` class from the `xLSTM` package:

```python
from xLSTM import mLSTM

# Create an mLSTM instance
mlstm = mLSTM(input_size, hidden_size, num_layers, dropout)

# Forward pass
outputs, hidden_states = mlstm(inputs)
```

For more details on using the mLSTM, refer to the API documentation and examples.

## References

- Beck, M., Pöppel, K., Spanring, M., Auer, A., Prudnikova, O., Kopp, M., Klambauer, G., Brandstetter, J., & Hochreiter, S. (2024). xLSTM: Extended Long Short-Term Memory. arXiv preprint arXiv:2405.04517.

---
