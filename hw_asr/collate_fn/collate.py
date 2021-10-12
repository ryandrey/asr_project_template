import logging
from typing import List
import torch
import torch.nn as nn
from collections import defaultdict

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    result_batch = defaultdict(list)
    for item in dataset_items:
        for key, value in item.items():
            if key == 'text_encoded' or key == 'spectrogram':
                result_batch[f"{key}_length"].append(torch.tensor([value.size(-1)]))
            result_batch[key].append(value.squeeze(0).transpose(0, -1) if torch.is_tensor(value) else value)

    for key, value in result_batch.items():
        if torch.is_tensor(value[0]):
            result_batch[key] = nn.utils.rnn.pad_sequence(value, True).transpose(1, -1)
            if result_batch[key].size(-1) == 1:
                result_batch[key] = result_batch[key].squeeze(-1)

    return result_batch
