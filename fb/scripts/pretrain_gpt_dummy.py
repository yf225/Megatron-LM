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

"""Pretrain GPT"""

from functools import partial

import numpy as np
import torch
from fairseq.benchmark.dummy_mt import DummyDataset
from megatron import get_args
from megatron import get_timers
from megatron import mpu
from megatron import print_rank_0
from megatron.model import GPTModel, ModelType
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


class GPTDummyDataset(torch.utils.data.Dataset):
    def __init__(self, name, num_samples, seq_length):
        self.name = name
        self.dataset_size = num_samples
        self.seq_length = seq_length

        self._dummy_ds = DummyDataset(
            None,
            num_items=self.dataset_size,
            item_size=seq_length,
        )

    def __len__(self):
        return len(self._dummy_ds)

    def __getitem__(self, idx):
        sample_list = np.concatenate(
            [
                np.array(self._dummy_ds[idx]).reshape((1,))
                for idx in range(self.seq_length + 1)
            ]
        ).astype(np.int64)
        return {"text": sample_list}


def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0("building GPT model ...")
    model = GPTModel(
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process,
    )
    return model


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    # tokenizer = get_tokenizer()  # TODO(willfeng): fix?

    # Items and their type.
    keys = ["text"]
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b["text"].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()
    assert tokens.shape[0] == args.micro_batch_size
    assert tokens.shape[1] == args.seq_length

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        "###########",  # tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss,
    )

    return tokens, labels, loss_mask, attention_mask, position_ids


def loss_func(loss_mask, output_tensor):
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {"lm loss": averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    # args = get_args()
    timers = get_timers()

    # Get the batch.
    timers("batch-generator").start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)
    timers("batch-generator").stop()

    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0("> building train, validation, and test datasets " "for GPT ...")
    train_ds = GPTDummyDataset(
        "train", num_samples=train_val_test_num_samples[0], seq_length=args.seq_length
    )
    valid_ds = GPTDummyDataset(
        "valid", num_samples=train_val_test_num_samples[1], seq_length=args.seq_length
    )
    test_ds = GPTDummyDataset(
        "test", num_samples=train_val_test_num_samples[2], seq_length=args.seq_length
    )
    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
    )
