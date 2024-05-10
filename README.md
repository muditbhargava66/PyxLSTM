# PyxLSTM

PyxLSTM is a Python library that provides an efficient and extensible implementation of the Extended Long Short-Term Memory (xLSTM) architecture based on the research paper ["xLSTM: Extended Long Short-Term Memory"](https://arxiv.org/abs/2405.04517) by Beck et al. (2024). xLSTM enhances the traditional LSTM by introducing exponential gating, memory mixing, and a matrix memory structure, enabling improved performance and scalability for sequence modeling tasks.

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
git clone https://github.com/muditbhargava66/PyxLSTM.git
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
│   ├── slstm.md
│   ├── mlstm.md
│   └── training.md
│
├── examples/
│   └── language_modeling.py
│
├── .gitignore
├── setup.py
├── MANIFEST.in
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

5. If all the tests pass successfully, you can proceed to run the example script:
   ```bash
   python examples/language_modeling.py --config path/to/config.yaml
   ```
   Replace `path/to/config.yaml` with the actual path to your configuration file. The configuration file should specify the dataset paths, model hyperparameters, and other settings.

   The `language_modeling.py` script will train an xLSTM model on the specified dataset using the provided configuration.

6. Monitor the training progress and metrics:
   During training, the script will display the training progress, including the current epoch, training loss, and validation loss. Keep an eye on these metrics to ensure the model is learning properly.

7. Evaluate the trained model:
   After training, you can evaluate the trained model on a test dataset using the `evaluate.py` script:
   ```bash
   python scripts/evaluate.py --test_data path/to/test_data.txt --vocab_file path/to/vocab.txt --checkpoint_path path/to/checkpoint.pt
   ```
   Replace the placeholders with the actual paths to your test data, vocabulary file, and the checkpoint file generated during training.

   The `evaluate.py` script will load the trained model from the checkpoint and evaluate its performance on the test dataset, providing metrics such as test loss and perplexity.

8. Generate text using the trained model:
   You can use the trained model to generate text using the `generate.py` script:
   ```bash
   python scripts/generate.py --vocab_file path/to/vocab.txt --checkpoint_path path/to/checkpoint.pt --prompt "Your prompt text"
   ```
   Replace the placeholders with the actual paths to your vocabulary file, checkpoint file, and provide a prompt text to initiate the generation.

   The `generate.py` script will load the trained model and generate text based on the provided prompt.

These steps should help you run and test the PyxLSTM codebase. Make sure you have the necessary dependencies installed and the required data files (train, validation, and test datasets) available.

If you encounter any issues or have further questions, please refer to the PyxLSTM documentation or reach out to the maintainers for assistance.

## Documentation

The documentation for PyxLSTM can be found in the `docs` directory. It provides detailed information about the library's components, usage guidelines, and examples.

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

PyxLSTM is released under the MIT License. See the `LICENSE` file for more information.

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

---