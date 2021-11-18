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

"""Pretrain VIT"""

import torch
import torch.nn.functional as F
from functools import partial
from megatron import get_args, get_timers, mpu, print_rank_0
# from megatron.data.vit_dataset import build_train_valid_datasets
from megatron.model import ModelType
from megatron.model.vit_model import VitModel
from megatron.training import pretrain
from megatron.utils import average_losses_across_data_parallel_group

cpu_datatype = torch.float

def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0("building VIT model ...")
    args = get_args()

    model = VitModel(num_classes=args.num_classes,
                     pre_process=pre_process,
                     post_process=post_process)
    return model

def get_batch(data_iterator):
    """Build the batch."""
    args = get_args()

    # Broadcast data.
    keys = ["image", "label"]
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = mpu.broadcast_data(keys, data, cpu_datatype)

    images = data_b["image"]
    labels = data_b["label"]
    assert images.shape[0] == args.micro_batch_size

    return images, labels

def loss_func(labels, output_tensor):
    logits = output_tensor.contiguous().float()
    loss = F.cross_entropy(logits, labels)

    outputs = torch.argmax(logits, -1)
    correct = (outputs == labels).float()
    accuracy = torch.mean(correct)

    averaged_loss = average_losses_across_data_parallel_group([loss, accuracy])

    return loss, {"loss": averaged_loss[0], "accuracy": averaged_loss[1]}

def forward_step(data_iterator, model):
    """Forward step."""
    timers = get_timers()

    # Get the batch.
    timers("batch-generator").start()
    (
        images,
        labels,
    ) = get_batch(data_iterator)
    timers("batch-generator").stop()

    # Forward model. lm_labels
    output_tensor = model(images)

    return output_tensor, partial(loss_func, labels)


class VitDummyDataset(torch.utils.data.Dataset):
    def __init__(self, crop_size=224):
        self.crop_size = crop_size

    def __len__(self):
        return 10000000

    def __getitem__(self, index):
        return {
            "image": torch.randn(3, self.crop_size, self.crop_size).to(cpu_datatype),
            "label": torch.tensor(1.0).to(cpu_datatype),
        }

def build_train_valid_datasets_dummy(crop_size=224):

    # # training dataset
    # train_data_path = os.path.join(data_path[0], "train")
    # normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # process = [
    #     transforms.RandomResizedCrop(crop_size),
    #     transforms.RandomHorizontalFlip(),
    # ]
    # if color_jitter:
    #     process += [
    #         transforms.ColorJitter(
    #             brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
    #         )
    #     ]
    # fp16_t = transforms.ConvertImageDtype(torch.half)
    # process += [ImageNetPolicy(), transforms.ToTensor(), normalize, fp16_t]
    # transform_train = transforms.Compose(process)
    # train_data = datasets.ImageFolder(
    #     root=train_data_path, transform=transform_train
    # )
    train_data = VitDummyDataset()

    # # validation dataset
    # val_data_path = os.path.join(data_path[0], "val")
    # transform_val = transforms.Compose(
    #     [
    #         transforms.Resize(crop_size),
    #         transforms.CenterCrop(crop_size),
    #         transforms.ToTensor(),
    #         normalize,
    #         fp16_t
    #     ]
    # )
    # val_data = datasets.ImageFolder(
    #     root=val_data_path, transform=transform_val
    # )
    val_data = VitDummyDataset()

    test_data = VitDummyDataset()

    return train_data, val_data, test_data


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0(
        "> building train, validation, and test datasets " "for VIT ..."
    )
    train_ds, valid_ds, test_ds = build_train_valid_datasets_dummy()
    print_rank_0("> finished creating VIT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'dataloader_type': 'cyclic'}
    )
