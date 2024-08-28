import collections
import os
import torch
import logging
import unicodedata
from typing import Optional, Tuple

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import BasicTokenizer

logger = logging.getLogger(__name__)
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

def load_vocab(vocab_file):
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab

def whitespace_tokenize(text):
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens

class KmerTokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        vocab_file,
        do_lower_case=False,
        never_split=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        tokenize_chinese_chars=True,
        max_len=512,
        k=6,
        stride=1,
        **kwargs
    ):
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.max_len = max_len
        self.k = k
        self.stride = stride
        super().__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )

        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(vocab_file)
            )
        self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars
            )

    def get_vocab(self):
        """Returns the vocabulary as a dictionary of token to token ID."""
        return self.vocab

    @property
    def vocab_size(self):
        return len(self.vocab)

    def _tokenize(self, text):
        kmer_size = self.k
        stride = self.stride
        split_tokens = []
        for i in range(0, len(text) - kmer_size + 1, stride):
            split_tokens.append(text[i:i + kmer_size])
        return split_tokens

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A BERT sequence has the following format:
            single sequence: [CLS] X [SEP]
        """
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]

        if len(token_ids_0) <= self.max_len - 2:
            return cls + token_ids_0 + sep
        else:
            output = []
            num_pieces = int(len(token_ids_0) // 510) + 1
            for i in range(num_pieces):
                output.extend(cls + token_ids_0[510*i:min(len(token_ids_0), 510*(i+1))] + sep)
            return output

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formated with special tokens for the model."
                )
            return list(map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0, token_ids_0))

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        
        if len(token_ids_0) < self.max_len - 2:
            return [1] + ([0] * len(token_ids_0)) + [1]
        else:
            output = []
            num_pieces = int(len(token_ids_0)//510) + 1
            for i in range(num_pieces):
                output.extend([1] + ([0] * (min(len(token_ids_0), 510*(i+1))-510*i)) + [1])
            return output
            return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            if len(token_ids_0) < self.max_len - 2:
                return len(cls + token_ids_0 + sep) * [0]
            else:
                num_pieces = int(len(token_ids_0)//510) + 1
                return (len(cls + token_ids_0 + sep) + 2*(num_pieces-1)) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """Save the tokenizer vocabulary to a directory or file."""
        index = 0
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            vocab_file = save_directory
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)
