# PyxLSTM

![Banner](assets/xlstm-logo-v2.png)

![Python Version](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9%20%7C%203.10-blue)
[![GitHub license](https://img.shields.io/github/license/muditbhargava66/PyxLSTM)](https://github.com/muditbhargava66/PyxLSTM/blob/main/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/pyxlstm/badge/?version=latest)](https://pyxlstm.readthedocs.io/en/latest/?badge=latest)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)
[![CodeQL](https://github.com/muditbhargava66/PyxLSTM/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/muditbhargava66/PyxLSTM/actions/workflows/github-code-scanning/codeql)
[![GitHub stars](https://img.shields.io/github/stars/muditbhargava66/PyxLSTM)](https://github.com/muditbhargava66/PyxLSTM/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/muditbhargava66/PyxLSTM)](https://github.com/muditbhargava66/PyxLSTM/network/members)
![GitHub Releases](https://img.shields.io/github/downloads/muditbhargava66/PyxLSTM/total)
![Last Commit](https://img.shields.io/github/last-commit/muditbhargava66/PyxLSTM)
[![Open Issues](https://img.shields.io/github/issues/muditbhargava66/PyxLSTM)](https://github.com/muditbhargava66/PyxLSTM/issues)
[![Open PRs](https://img.shields.io/github/issues-pr/muditbhargava66/PyxLSTM)](https://github.com/muditbhargava66/PyxLSTM/pulls)


PyxLSTM is a Python library that provides an efficient and extensible implementation of the Extended Long Short-Term Memory (xLSTM) architecture based on the research paper ["xLSTM: Extended Long Short-Term Memory"](https://arxiv.org/abs/2405.04517) by Beck et al. (2024). xLSTM enhances the traditional LSTM by introducing exponential gating, memory mixing, and a matrix memory structure, enabling improved performance and scalability for sequence modeling tasks.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Development Installation](#development-installation)
- [Usage](#usage)
- [Code Directory Structure](#code-directory-structure)
- [Running and Testing the Codebase](#running-and-testing-the-codebase)
- [Documentation](#documentation)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)
- [Star History](#star-history)
- [TODO](#todo)

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

## Development Installation

For development installation with testing dependencies:

```bash
pip install PyxLSTM[dev]
```

Alternatively, you can clone the repository and install it manually:

```bash
git clone https://github.com/muditbhargava66/PyxLSTM.git
cd PyxLSTM
pip install -r requirements.txt
pip install -e .
```

## Usage

Here's a basic example of how to use PyxLSTM for language modeling:

```python
import torch
from xLSTM.model import xLSTM
from xLSTM.data import LanguageModelingDataset, Tokenizer
from xLSTM.utils import load_config, set_seed, get_device
from xLSTM.training import train  # Assuming train function is defined in training module

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

For more detailed usage instructions and examples, please refer to the [documentation](docs/).

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
│   ├── slstm.md
│   ├── mlstm.md
│   └── training.md
│
├── examples/
│   ├── language_modeling.py
│   └── xLSTM_shape_verification.py
│
├── .gitignore
├── pyproject.toml
├── MANIFEST.in
├── requirements.txt
├── README.md
└── LICENSE
```

- **xLSTM/**: The main Python package containing the implementation.
  - slstm.py: Implementation of the sLSTM module.
  - mlstm.py: Implementation of the mLSTM module.
  - block.py: Implementation of the xLSTM blocks (pre and post up-projection).
  - model.py: High-level xLSTM model definition.

- **utils/**: Utility modules.
  - `config.py`: Configuration management.
  - `logging.py`: Logging setup.
  - `utils.py`: Miscellaneous utility functions.

- **tests/**: Unit tests for different modules.
  - `test_slstm.py`: Tests for sLSTM module.  
  - `test_mlstm.py`: Tests for mLSTM module.
  - `test_block.py`: Tests for xLSTM blocks.
  - `test_model.py`: Tests for the overall xLSTM model.

- **docs/**: Documentation files.
  - `README.md`: Main documentation file.
  - `slstm.md`: Documentation for sLSTM.
  - `mlstm.md`: Documentation for mLSTM.
  - `training.md`: Training guide.

- **.gitignore**: Git ignore file to exclude unnecessary files/directories.
- **setup.py**: Package setup script.
- **requirements.txt**: List of required Python dependencies.
- **README.md**: Project README file.
- **LICENSE**: Project license file.

## Running and Testing the Codebase

To run and test the PyxLSTM codebase, follow these steps:

1. Clone the PyxLSTM repository:
   ```bash
   git clone https://github.com/muditbhargava66/PyxLSTM.git
   ```

2. Navigate to the cloned directory:
   ```bash
   cd PyxLSTM
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the unit tests:
   ```bash
   python -m unittest discover tests
   ```
   This command will run all the unit tests located in the `tests` directory. It will execute the test files `test_slstm.py`, `test_mlstm.py`, `test_block.py`, and `test_model.py`.

If you encounter any issues or have further questions, please refer to the PyxLSTM documentation or reach out to the maintainers for assistance.

## Documentation

The documentation for PyxLSTM can be found in the [docs](docs/) directory. It provides detailed information about the library's components, usage guidelines, and examples.

## Citation

If you use PyxLSTM in your research or projects, please cite the original xLSTM paper:

```bibtex
@article{Beck2024xLSTM,
  title={xLSTM: Extended Long Short-Term Memory},
  author={Beck, Maximilian and Pöppel, Korbinian and Spanring, Markus and Auer, Andreas and Prudnikova, Oleksandra and Kopp, Michael and Klambauer, Günter and Brandstetter, Johannes and Hochreiter, Sepp},
  journal={arXiv preprint arXiv:2405.04517},
  year={2024}
}
```

Paper link: [https://arxiv.org/abs/2405.04517](https://arxiv.org/abs/2405.04517)

## Contributing

Contributions to PyxLSTM are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request on the GitHub repository.

## License

PyxLSTM is released under the [MIT License](LICENSE). See the `LICENSE` file for more information.

## Acknowledgements

We would like to acknowledge the original authors of the xLSTM architecture for their valuable research and contributions to the field of sequence modeling.

## Contact

For any questions or inquiries, please contact the project maintainer:

- Name: Mudit Bhargava
- GitHub: [@muditbhargava66](https://github.com/muditbhargava66)

We hope you find PyxLSTM useful for your sequence modeling projects!

## Star History

<a href="https://star-history.com/#muditbhargava66/PyxLSTM&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=muditbhargava66/PyxLSTM&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=muditbhargava66/PyxLSTM&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=muditbhargava66/PyxLSTM&type=Date" />
 </picture>
</a>

## TODO

- [x] Add support for Python 3.10
- [x] Add support for macOS MPS
- [x] Add support for Windows MPS
- [x] Add support for Linux MPS
- [ ] Provide more examples on time series prediction
- [ ] Include reinforcement learning examples
- [ ] Add examples for modeling physical systems
- [ ] Enhance documentation with advanced usage scenarios
- [ ] Improve unit tests for new features
- [ ] Add support for bidirectional parameter as it's not implemented in the current xLSTM model

---