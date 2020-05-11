import math
import os
import time
import torch

from torch.utils.data import TensorDataset
from torchtext import data


class TranslationExample:
    def __init__(self, en, fr):
        self.en = en
        self.fr = fr


def load_en_fr(root: str, mini: bool):
    dataset_path = 'datasets/europarl-v7-fr-en-mini' if mini else 'datasets/europarl-v7-fr-en'
    with open(os.path.join(root, dataset_path, 'en')) as file_en:
        lines_en = file_en.read().split('\n')

    with open(os.path.join(root, dataset_path, 'fr')) as file_fr:
        lines_fr = file_fr.read().split('\n')

    return _process(lines_en, lines_fr)


def _process(samples_en, samples_fr):
    start_time = time.time()

    # CPU
    device = None

    # For a list of possible tokenizers, look here:
    # https://github.com/pytorch/text/blob/master/torchtext/data/utils.py#L75
    # Note that tokenizer language works only for spacy
    tokenizer = 'moses'

    print('processing english...')

    EN = data.Field(sequential=True,
                    use_vocab=True,
                    init_token='<bos>',
                    eos_token='<eos>',
                    tokenize=tokenizer,
                    tokenizer_language='en')

    print('[1/3] preprocessing')
    tokens_en = [EN.preprocess(s) for s in samples_en]

    print('[2/3] building vocab')
    EN.build_vocab(tokens_en)

    print('[3/3] processing')
    processed_en = EN.process(tokens_en, device=device)
    # TODO: SHOULD USE BATCH FIRST PARAM TO FIELD.
    processed_en = torch.transpose(processed_en, 1, 0)

    print('processing french...')

    FR = data.Field(sequential=True,
                    use_vocab=True,
                    init_token='<bos>',
                    eos_token='<eos>',
                    tokenize=tokenizer,
                    tokenizer_language='fr')

    print('[1/3] preprocessing')
    tokens_fr = [FR.preprocess(s) for s in samples_fr]

    print('[2/3] building vocab')
    FR.build_vocab(tokens_fr)

    print('[3/3] processing')
    processed_fr = FR.process(tokens_fr, device=device)
    # TODO: SHOULD USE BATCH FIRST PARAM TO FIELD.
    processed_fr = torch.transpose(processed_fr, 1, 0)
    end_time = time.time()

    print(f'Processing took: {(end_time - start_time)/60:.02f}m')

    return TensorDataset(processed_en, processed_fr), EN, FR
