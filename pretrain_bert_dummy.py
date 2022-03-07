# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pretrain BERT"""

from functools import partial

import numpy as np
import torch
from megatron import get_args
from megatron import get_timers
from megatron import mpu
from megatron import print_rank_0
from megatron.model import BertModel, ModelType
from megatron.training import pretrain
from megatron.utils import average_losses_across_data_parallel_group
from megatron.utils import get_ltor_masks_and_position_ids


'''
def dummy_compile_helper():
    print("NOTE(willfeng): compile_helper() is skipped!")


from megatron.data import dataset_utils as megatron__data__dataset_utils

megatron__data__dataset_utils.compile_helper = dummy_compile_helper


def dummy_fused_kernels_load(args):
    print("""
    NOTE: we don't need to call fused_kernels.load() since fused kernels are loaded
    via `cpp_python_extension` in buck build.
    """)


from megatron import fused_kernels as megatron__fused_kernels

megatron__fused_kernels.load = dummy_fused_kernels_load
'''

from megatron import arguments as megatron__arguments

_add_data_args_original = megatron__arguments._add_data_args


def add_extra_args_to_add_data_args_func(parser):
    parser = _add_data_args_original(parser)
    group = parser.add_argument_group(title="data and dataloader")
    group.add_argument(
        "--padded-vocab-size", type=int, default=1024, help="TODO(willfeng)"
    )
    return parser


megatron__arguments._add_data_args = add_extra_args_to_add_data_args_func


_add_logging_args_original = megatron__arguments._add_logging_args


# HACK(willfeng): added this to be compatible with submitit
def add_extra_args_to_add_logging_args_func(parser):
    parser = _add_logging_args_original(parser)
    group = parser.add_argument_group(title="submitit logging")
    parser.add_argument("--folder", type=str, default="", help="Folder where the jobs are stored (in subfolder)")
    return parser


megatron__arguments._add_logging_args = add_extra_args_to_add_logging_args_func


class BERTDummyDataset(torch.utils.data.Dataset):
    def __init__(self, name, num_samples, seq_length):
        self.name = name
        self.dataset_size = num_samples
        self.seq_length = seq_length

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        sample_list = np.concatenate(
            [
                np.array([idx]).reshape((1,))
                for idx in range(self.seq_length)
            ]
        ).astype(np.int64)
        return {
            "text": sample_list,
            "types": np.zeros_like(sample_list),
            "is_random": 0,
            "loss_mask": np.array([1] + [0] * (self.seq_length - 1), dtype=np.int64),
            "labels": np.zeros(self.seq_length, dtype=np.int64),
            "padding_mask": np.zeros(self.seq_length, dtype=np.int64),
        }


def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building BERT model ...')

    args = get_args()
    assert args.bert_binary_head == False, "bert_binary_head is not supported in this experiment"
    num_tokentypes = 2 if args.bert_binary_head else 0
    model = BertModel(
        num_tokentypes=num_tokentypes,
        add_binary_head=args.bert_binary_head,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process,
        init_method=torch.nn.init.zeros_,
        scaled_init_method=torch.nn.init.zeros_)
    for w in model.parameters():
        torch.nn.init.zeros_(w)

    return model


def get_batch(data_iterator):
    """Build the batch."""

    # Items and their type.
    keys = ['text', 'types', 'labels', 'is_random', 'loss_mask', 'padding_mask']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens = data_b['text'].long()
    types = data_b['types'].long()
    sentence_order = data_b['is_random'].long()
    loss_mask = data_b['loss_mask'].float()
    lm_labels = data_b['labels'].long()
    padding_mask = data_b['padding_mask'].long()

    return tokens, types, sentence_order, loss_mask, lm_labels, padding_mask


def loss_func(loss_mask, sentence_order, output_tensor):
    lm_loss_, sop_logits = output_tensor
    assert sop_logits is None, "sop_logits not supported in this experiment"

    lm_loss_ = lm_loss_.float()
    loss_mask = loss_mask.float()
    lm_loss = torch.sum(
        lm_loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()

    if sop_logits is not None:
        sop_loss = F.cross_entropy(sop_logits.view(-1, 2).float(),
                                   sentence_order.view(-1),
                                   ignore_index=-1)
        sop_loss = sop_loss.float()
        loss = lm_loss + sop_loss
        averaged_losses = average_losses_across_data_parallel_group(
            [lm_loss, sop_loss])
        return loss, {'lm loss': averaged_losses[0],
                      'sop loss': averaged_losses[1]}

    else:
        loss = lm_loss
        averaged_losses = average_losses_across_data_parallel_group(
            [lm_loss])
        return loss, {'lm loss': averaged_losses[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator').start()
    tokens, types, sentence_order, loss_mask, lm_labels, padding_mask = get_batch(
        data_iterator)
    timers('batch-generator').stop()

    if not args.bert_binary_head:
        types = None

    # Forward pass through the model.
    output_tensor = model(tokens, padding_mask, tokentype_ids=types,
                          lm_labels=lm_labels)

    return output_tensor, partial(loss_func, loss_mask, sentence_order)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0("> building train, validation, and test datasets " "for BERT ...")
    train_ds = BERTDummyDataset(
        "train", num_samples=train_val_test_num_samples[0], seq_length=args.seq_length
    )
    valid_ds = BERTDummyDataset(
        "valid", num_samples=train_val_test_num_samples[1], seq_length=args.seq_length
    )
    test_ds = BERTDummyDataset(
        "test", num_samples=train_val_test_num_samples[2], seq_length=args.seq_length
    )
    print_rank_0("> finished creating BERT datasets ...")

    return train_ds, valid_ds, test_ds


def train_model():
    pretrain(train_valid_test_datasets_provider, model_provider,
             ModelType.encoder_or_decoder,
             forward_step, args_defaults={'tokenizer_type': 'BertWordPieceLowerCase'})


if __name__ == "__main__":
    train_model()