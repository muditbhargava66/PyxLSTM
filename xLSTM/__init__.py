"""
xLSTM: Extended Long Short-Term Memory

This package implements the xLSTM model as described in the paper:
"xLSTM: Extended Long Short-Term Memory" by Beck et al. (2024).

The xLSTM combines sLSTM (Scalar LSTM) and mLSTM (Matrix LSTM) in a novel
architecture to achieve state-of-the-art performance on various language
modeling tasks.

This __init__.py file imports and exposes the main components of the xLSTM model.

Author: Mudit Bhargava
Date: June 2024
"""

from .slstm import sLSTM, sLSTMCell
from .mlstm import mLSTM, mLSTMCell
from .block import xLSTMBlock
from .model import xLSTM

__all__ = [
    "sLSTM",
    "sLSTMCell",
    "mLSTM",
    "mLSTMCell",
    "xLSTMBlock",
    "xLSTM"
]

__version__ = "2.0.0"
__author__ = "Mudit Bhargava"