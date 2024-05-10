# sLSTM: Scalar Long Short-Term Memory

The sLSTM (Scalar Long Short-Term Memory) is a variant of the LSTM architecture that introduces exponential gating and memory mixing. It enhances the traditional LSTM by enabling the model to revise storage decisions and improve performance on tasks requiring state tracking.

## Architecture

The sLSTM architecture consists of the following components:

- Input Gate ($i$): Controls the flow of input information into the memory cell.
- Forget Gate ($f$): Determines the amount of information to retain or forget from the previous memory cell state.
- Output Gate ($o$): Controls the flow of information from the memory cell to the hidden state.
- Cell State ($c$): Stores the long-term memory information.
- Hidden State ($h$): Represents the output of the sLSTM at each time step.

The sLSTM introduces exponential gating for the input and forget gates, which allows for more fine-grained control over the memory updates. It also incorporates memory mixing, where the hidden state from the previous time step is used to modulate the gates and cell state update.

## Equations

The equations governing the sLSTM are as follows:

- Cell State Update:
  $c_t = f_t \odot c_{t-1} + i_t \odot z_t$

- Normalizer State Update:
  $n_t = f_t \odot n_{t-1} + i_t$

- Hidden State Update:
  $h_t = o_t \odot \tanh(c_t \odot n_t^{-1})$

- Input Gate:
  $i_t = \exp(W_i x_t + b_i)$

- Forget Gate:
  $f_t = \exp(W_f x_t + b_f)$

- Output Gate:
  $o_t = \sigmoid(W_o h_t + b_o)$

- Cell Input:
  $z_t = 0$

where:
- $x_t$ is the input vector at time step $t$
- $W_*$ are weight matrices
- $b_*$ are bias vectors

## Initialization and Training

The sLSTM can be initialized with random weights and biases. The exponential gating allows for stable training, even with large values.

During training, the gradients are computed using backpropagation through time (BPTT). The sLSTM is trained using standard optimization techniques such as stochastic gradient descent (SGD) or Adam.

## Usage

To use the sLSTM in your project, you can import the `sLSTM` class from the `xLSTM` package:

```python
from xLSTM import sLSTM

# Create an sLSTM instance
slstm = sLSTM(input_size, hidden_size, num_layers, dropout)

# Forward pass
outputs, hidden_states = slstm(inputs)
```

For more details on using the sLSTM, refer to the API documentation and examples.

## References

- Beck, M., Pöppel, K., Spanring, M., Auer, A., Prudnikova, O., Kopp, M., Klambauer, G., Brandstetter, J., & Hochreiter, S. (2024). xLSTM: Extended Long Short-Term Memory. arXiv preprint arXiv:2405.04517.

---