# PyxLSTM

PyxLSTM is a Python library that provides an efficient and extensible implementation of the Extended Long Short-Term Memory (xLSTM) architecture. xLSTM enhances the traditional LSTM by introducing exponential gating, memory mixing, and a matrix memory structure, enabling improved performance and scalability for sequence modeling tasks.

## Features

- Implements the sLSTM (scalar LSTM) and mLSTM (matrix LSTM) variants of xLSTM
- Supports pre and post up-projection block structures for flexible model architectures
- Provides high-level model definition and training utilities for ease of use
- Includes scripts for training, evaluation, and text generation
- Offers data processing utilities and customizable dataset classes
- Lightweight and modular design for seamless integration into existing projects
- Extensively tested and documented for reliability and usability
- Suitable for a wide range of sequence modeling tasks, including language modeling, text generation, and more

## Installation

To install PyxLSTM, you can use pip:

```bash
pip install PyxLSTM
```

Alternatively, you can clone the repository and install it manually:

```bash
git clone https://github.com/yourusername/PyxLSTM.git
cd PyxLSTM
pip install -r requirements.txt
python setup.py install
```

## Usage

Here's a basic example of how to use PyxLSTM for language modeling:

```python
from xLSTM.model import xLSTM
from xLSTM.data import LanguageModelingDataset, Tokenizer
from xLSTM.utils import load_config, set_seed, get_device

# Load configuration
config = load_config("path/to/config.yaml")
set_seed(config.seed)
device = get_device()

# Initialize tokenizer and dataset
tokenizer = Tokenizer(config.vocab_file)
train_dataset = LanguageModelingDataset(config.train_data, tokenizer, config.max_length)

# Create xLSTM model
model = xLSTM(len(tokenizer), config.embedding_size, config.hidden_size,
              config.num_layers, config.num_blocks, config.dropout,
              config.bidirectional, config.lstm_type)
model.to(device)

# Train the model
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
train(model, train_dataset, optimizer, criterion, config, device)
```

For more detailed usage instructions and examples, please refer to the documentation.

## Code Directory Structure

```
xLSTM/
│
├── xLSTM/
│   ├── __init__.py
│   ├── slstm.py
│   ├── mlstm.py
│   ├── block.py
│   └── model.py
│ 
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── generate.py
│
├── data/
│   ├── dataset.py
│   └── tokenizer.py
│
├── utils/
│   ├── config.py
│   ├── logging.py
│   └── utils.py
│
├── tests/
│   ├── test_slstm.py  
│   ├── test_mlstm.py
│   ├── test_block.py
│   └── test_model.py
│
├── docs/
│   ├── README.md
│   ├── slstm.md
│   ├── mlstm.md
│   └── training.md
│
├── examples/
│   └── language_modeling.py
│
├── .gitignore
├── setup.py
├── requirements.txt
├── README.md
└── LICENSE
```

- xLSTM/: The main Python package containing the implementation.
  - slstm.py: Implementation of the sLSTM module.
  - mlstm.py: Implementation of the mLSTM module.
  - block.py: Implementation of the xLSTM blocks (pre and post up-projection).
  - model.py: High-level xLSTM model definition.

- scripts/: Scripts for training, evaluation, and text generation.
  - train.py: Script for training the xLSTM model.
  - evaluate.py: Script for evaluating the trained model.
  - generate.py: Script for generating text using the trained model.

- data/: Data processing utilities.
  - dataset.py: Custom dataset classes for loading and processing data.
  - tokenizer.py: Tokenization utilities.

- utils/: Utility modules.
  - config.py: Configuration management.
  - logging.py: Logging setup.
  - utils.py: Miscellaneous utility functions.

- tests/: Unit tests for different modules.
  - test_slstm.py: Tests for sLSTM module.  
  - test_mlstm.py: Tests for mLSTM module.
  - test_block.py: Tests for xLSTM blocks.
  - test_model.py: Tests for the overall xLSTM model.

- docs/: Documentation files.
  - README.md: Main documentation file.
  - slstm.md: Documentation for sLSTM.
  - mlstm.md: Documentation for mLSTM.
  - training.md: Training guide.

- examples/: Example usage scripts.
  - language_modeling.py: Example script for language modeling with xLSTM.

- .gitignore: Git ignore file to exclude unnecessary files/directories.
- setup.py: Package setup script.
- requirements.txt: List of required Python dependencies.
- README.md: Project README file.
- LICENSE: Project license file.

## Documentation

The documentation for PyxLSTM can be found in the `docs` directory. It provides detailed information about the library's components, usage guidelines, and examples.

## Contributing

Contributions to PyxLSTM are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request on the GitHub repository.

## License

PyxLSTM is released under the MIT License. See the `LICENSE` file for more information.

## Acknowledgements

We would like to acknowledge the original authors of the xLSTM architecture for their valuable research and contributions to the field of sequence modeling.

## Contact

For any questions or inquiries, please contact the project maintainer:

- Name: Mudit Bhargava
- GitHub: [@muditbhargava66](https://github.com/muditbhargava66)

I hope you find PyxLSTM useful for your sequence modeling projects!

---