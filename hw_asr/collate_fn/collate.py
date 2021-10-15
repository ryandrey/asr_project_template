import logging
from typing import List
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    spec_lengths = [item['spectrogram'].size(-1) for item in dataset_items]
    text_lengths = [item['text_encoded'].size(-1) for item in dataset_items]

    bs = len(dataset_items)
    spectrogram_out = torch.zeros((bs, dataset_items[0]['spectrogram'].size(1), max(spec_lengths)))
    text_out = torch.zeros((bs, max(text_lengths)))

    for i, item in enumerate(dataset_items):
        spectrogram_out[i, :, :spec_lengths[i]] = item['spectrogram']
        text_out[i, :text_lengths[i]] = item['text_encoded']

    result_batch = {
        'spectrogram': spectrogram_out.transpose(1, 2),
        'spectrogram_length': torch.tensor(spec_lengths),
        'text_encoded': text_out,
        'text_encoded_length': torch.tensor(text_lengths),
        'text': [item['text'] for item in dataset_items]
    }
    return result_batch
