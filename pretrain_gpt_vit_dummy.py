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


class EpochListening:
    """Mixin for receiving updates whenever the epoch increments."""

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        """
        Whether we can reuse the :class:`fairseq.data.EpochBatchIterator` for
        this dataset across epochs.

        This needs to return ``False`` if the sample sizes can change across
        epochs, in which case we may need to regenerate batches at each epoch.
        If your dataset relies in ``set_epoch`` then you should consider setting
        this to ``False``.
        """
        return True

    def set_epoch(self, epoch):
        """Will receive the updated epoch number at the beginning of the epoch."""
        pass


class FairseqDataset(torch.utils.data.Dataset, EpochListening):
    """A dataset that provides helpers for batching."""

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch suitable for forwarding with a Model
        """
        raise NotImplementedError

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        raise NotImplementedError

    def num_tokens_vec(self, indices):
        """Return the number of tokens for a set of positions defined by indices.
        This value is used to enforce ``--max-tokens`` during batching."""
        raise NotImplementedError

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        raise NotImplementedError

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        return np.arange(len(self), dtype=np.int64)

    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return False

    def attr(self, attr: str, index: int):
        return getattr(self, attr, None)

    def prefetch(self, indices):
        """Prefetch the data required for this epoch."""
        raise NotImplementedError

    def get_batch_shapes(self):
        """
        Return a list of valid batch shapes, for example::

            [(8, 512), (16, 256), (32, 128)]

        The first dimension of each tuple is the batch size and can be ``None``
        to automatically infer the max batch size based on ``--max-tokens``.
        The second dimension of each tuple is the max supported length as given
        by :func:`fairseq.data.FairseqDataset.num_tokens`.

        This will be used by :func:`fairseq.data.FairseqDataset.batch_by_size`
        to restrict batch shapes. This is useful on TPUs to avoid too many
        dynamic shapes (and recompilations).
        """
        return None

    def batch_by_size(
        self,
        indices,
        max_tokens=None,
        max_sentences=None,
        required_batch_size_multiple=1,
    ):
        """
        Given an ordered set of indices, return batches according to
        *max_tokens*, *max_sentences* and *required_batch_size_multiple*.
        """
        from fairseq.data import data_utils

        fixed_shapes = self.get_batch_shapes()
        if fixed_shapes is not None:

            def adjust_bsz(bsz, num_tokens):
                if bsz is None:
                    assert max_tokens is not None, "Must specify --max-tokens"
                    bsz = max_tokens // num_tokens
                if max_sentences is not None:
                    bsz = min(bsz, max_sentences)
                elif (
                    bsz >= required_batch_size_multiple
                    and bsz % required_batch_size_multiple != 0
                ):
                    bsz -= bsz % required_batch_size_multiple
                return bsz

            fixed_shapes = np.array(
                [
                    [adjust_bsz(bsz, num_tokens), num_tokens]
                    for (bsz, num_tokens) in fixed_shapes
                ]
            )

        try:
            num_tokens_vec = self.num_tokens_vec(indices).astype('int64')
        except NotImplementedError:
            num_tokens_vec = None

        return data_utils.batch_by_size(
            indices,
            num_tokens_fn=self.num_tokens,
            num_tokens_vec=num_tokens_vec,
            max_tokens=max_tokens,
            max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
            fixed_shapes=fixed_shapes,
        )

    def filter_indices_by_size(self, indices, max_sizes):
        """
        Filter a list of sample indices. Remove those that are longer than
        specified in *max_sizes*.

        WARNING: don't update, override method in child classes

        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)

        Returns:
            np.array: filtered sample array
            list: list of removed indices
        """
        if isinstance(max_sizes, float) or isinstance(max_sizes, int):
            if hasattr(self, "sizes") and isinstance(self.sizes, np.ndarray):
                ignored = indices[self.sizes[indices] > max_sizes].tolist()
                indices = indices[self.sizes[indices] <= max_sizes]
            elif (
                hasattr(self, "sizes")
                and isinstance(self.sizes, list)
                and len(self.sizes) == 1
            ):
                ignored = indices[self.sizes[0][indices] > max_sizes].tolist()
                indices = indices[self.sizes[0][indices] <= max_sizes]
            else:
                indices, ignored = data_utils._filter_by_size_dynamic(
                    indices, self.size, max_sizes
                )
        else:
            indices, ignored = data_utils._filter_by_size_dynamic(
                indices, self.size, max_sizes
            )
        return indices, ignored

    @property
    def supports_fetch_outside_dataloader(self):
        """Whether this dataset supports fetching outside the workers of the dataloader."""
        return True


class DummyDataset(FairseqDataset):
    def __init__(self, batch, num_items, item_size):
        super().__init__()
        self.batch = batch
        self.num_items = num_items
        self.item_size = item_size

    def __getitem__(self, index):
        return index

    def __len__(self):
        return self.num_items

    def collater(self, samples):
        return self.batch

    @property
    def sizes(self):
        return np.array([self.item_size] * self.num_items)

    def num_tokens(self, index):
        return self.item_size

    def size(self, index):
        return self.item_size

    def ordered_indices(self):
        return np.arange(self.num_items)

    @property
    def supports_prefetch(self):
        return False

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
