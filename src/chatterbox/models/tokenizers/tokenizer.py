import logging

import torch
from tokenizers import Tokenizer


# Special tokens
SOT = "[START]"
EOT = "[STOP]"
UNK = "[UNK]"
SPACE = "[SPACE]"
SPECIAL_TOKENS = [SOT, EOT, UNK, SPACE, "[PAD]", "[SEP]", "[CLS]", "[MASK]"]

logger = logging.getLogger(__name__)

class EnTokenizer:
    """A class for tokenizing English text using a vocabulary file. Initializes with a path to the vocab file and checks for specific tokens. Provides method to convert text to token tensors."""
    def __init__(self, vocab_file_path):
        """Initializes the tokenizer and checks for special tokens.
        Args:
        vocab_file_path (str): Path to the vocabulary file.
        Checks if start-of-text (SOT) and end-of-text (EOT) tokens are present in the vocabulary.
        Converts text to token IDs.
        Args:
        text (str): The input text.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
        Returns:
        torch.IntTensor: Tensor of token IDs.
        """
        self.tokenizer: Tokenizer = Tokenizer.from_file(vocab_file_path)
        self.check_vocabset_sot_eot()

    def check_vocabset_sot_eot(self):
        """Checks if start of text and end of text tokens are in the vocabulary.
        Args:
        None
        Returns:
        None
        Converts text to token IDs using a tokenizer and wraps them in a tensor.
        Args:
        text (str): The input text to be converted.
        Returns:
        torch.Tensor: A tensor containing the encoded token IDs.
        Preprocesses the input text by cleaning, appending language ID, replacing spaces, and then encoding it using the tokenizer.
        """
        voc = self.tokenizer.get_vocab()
        assert SOT in voc
        assert EOT in voc

    def text_to_tokens(self, text: str):
        """Converts a string to tokens.
        Args:
        text: The input string to convert.
        Returns:
        A tensor of token IDs.
        """
        text_tokens = self.encode(text)
        text_tokens = torch.IntTensor(text_tokens).unsqueeze(0)
        return text_tokens

    def encode( self, txt: str, verbose=False):
        """
        clean_text > (append `lang_id`) > replace SPACE > encode text using Tokenizer
        """
        txt = txt.replace(' ', SPACE)
        code = self.tokenizer.encode(txt)
        ids = code.ids
        return ids

    def decode(self, seq):
        """Converts a sequence to a decoded string.
        Args:
        seq (torch.Tensor or list): The input sequence to decode.
        Returns:
        str: The decoded string.
        """
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()

        txt: str = self.tokenizer.decode(seq,
        skip_special_tokens=False)
        txt = txt.replace(' ', '')
        txt = txt.replace(SPACE, ' ')
        txt = txt.replace(EOT, '')
        txt = txt.replace(UNK, '')
        return txt
