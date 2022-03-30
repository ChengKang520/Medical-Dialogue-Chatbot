

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Import libraries
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
# https://www.kaggle.com/parthplc/t5-fine-tuning-tutorial#T5-text-to-text-transformer


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Import libraries
import argparse
import os
import json
import time
import logging
import random

import nltk
nltk.download('punkt')

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)

t5_model = 't5-small'  #  't5-small' #  't5-large' #  't5-base'
gpu_number = 1
train_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)


def get_dataset(tokenizer, type_path, args):
    return MedicalDialogueDataset(tokenizer=tokenizer, data_dir=args.data_dir, type_path=type_path,  max_len=args.max_seq_length)

## Model
class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams):
        super(T5FineTuner, self).__init__()
        # self.hparams = hparams
        self.save_hyperparameters(hparams)
        self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
        self.tokenizer = T5Tokenizer.from_pretrained(hparams.tokenizer_name_or_path)

    def is_logger(self):
        return self.trainer.global_rank <= 0

    def forward(
            self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, label=None
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=label,
        )

    def _step(self, batch):
        label = batch["target_ids"]
        label[label[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            label=label,
            decoder_attention_mask=batch['target_mask']
        )

        loss = outputs[0]

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    # def training_epoch_end(self, outputs):
    #     avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
    #     tensorboard_logs = {"avg_train_loss": avg_train_loss}
    #     return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        tensorboard_logs = {"val_loss": loss}
        return {"val_loss": loss, "log": tensorboard_logs}

    # def validation_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
    #     self.log('val_loss', avg_loss, logger=True)
    #     tensorboard_logs = {"val_loss": avg_loss}
    #     return {"val_loss": avg_loss, "log": tensorboard_logs}  # , 'progress_bar': tensorboard_logs

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    # def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
    #     if self.trainer.use_tpu:
    #         # xm.optimizer_step(optimizer)
    #         ha = 1
    #     else:
    #         optimizer.step()
    #     optimizer.zero_grad()
    #     self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}
        return tqdm_dict

    def train_dataloader(self):
        train_dataset = get_dataset(tokenizer=self.tokenizer, type_path="train", args=self.hparams)
        dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, drop_last=True, shuffle=True,
                                num_workers=4)
        t_total = (
                (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
                // self.hparams.gradient_accumulation_steps
                * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = get_dataset(tokenizer=self.tokenizer, type_path="val", args=self.hparams)
        return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4)


logger = logging.getLogger(__name__)

class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
            # Log results
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info("***** Test results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
            # Log and save results to file
            output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
            with open(output_test_results_file, "w") as writer:
                for key in sorted(metrics):
                    if key not in ["log", "progress_bar"]:
                        logger.info("{} = {}\n".format(key, str(metrics[key])))
                        writer.write("{} = {}\n".format(key, str(metrics[key])))






import csv
from dataclasses import dataclass

from enum import Enum
from typing import List, Optional
from transformers import PreTrainedTokenizer


@dataclass(frozen=True)
class InputExample:
    example_id: str
    context: str
    answer: str
    label: str
    """
      A single training/test example for multiple choice
      Args:
          example_id: Unique id for the example.
          contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
          answer : str containing answer for which we need to generate question
          label: string containg questions
      """


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


class DataProcessor:
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class SwagProcessor(DataProcessor):
    """Processor for the SWAG data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        print('******************** Load Train')
        return self._create_examples(os.path.join(data_dir, "train_MedDG_trans.json"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        print('******************** Load Validation')
        return self._create_examples(os.path.join(data_dir, "dev_MedDG_trans.json"), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        raise ValueError(
            "For swag testing, the input file does not contain a label column. It can not be tested in current code"
            "setting!"
        )
        return self._create_examples(os.path.join(data_dir, "test_MedDG_trans.json"), "test")

    def _create_examples(self, filename, type: str):
        """加载数据
        单条格式：(标题, 正文)
        """
        examples = []
        with open(filename, mode='rb') as json_file:
            student_loaded = json.load(json_file)
            msg_contents = student_loaded[0]
            msg_labels_binary = student_loaded[1]
            msg_labels_multi = student_loaded[2]
            flag_num = 0
            msg_labels_temp = ''
            for dialogue_seg_ind in range(len(msg_contents)):
                dialogue_seg = msg_contents[dialogue_seg_ind]
                flag_num += 1
                if flag_num== 84:
                    ha = 1
                try:
                    msg_doctor, msg_patient = dialogue_seg.split('*')
                except:
                    msg_doctor = dialogue_seg
                    msg_patient = ' '

                msg_labels_temp = msg_labels_multi[dialogue_seg_ind]
                examples.append(InputExample(
                    example_id=flag_num,
                    context=msg_doctor,
                    answer=msg_patient,
                    label=msg_labels_temp)
                )

        return examples


class MedicalDialogueDataset(Dataset):
    def __init__(self, tokenizer, data_dir, type_path, max_len=512):
        self.data_dir = data_dir
        self.type_path = type_path
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []

        self.proc = SwagProcessor()

        self._build()

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}

    def __len__(self):
        return len(self.inputs)

    def _build(self):
        if self.type_path == 'train':
            examples = self.proc.get_train_examples(self.data_dir)
        else:
            examples = self.proc.get_dev_examples(self.data_dir)

        i_num = 0
        for example in tqdm(examples):
            self._create_features(example)

    def _create_features(self, example):
        input_ = "Doctor: %s </s> Patients: %s" % (example.context, example.answer)
        target_ = "%s" % (str(example.label))
        # self.targets.append(target_)

        AA = 'it not cool that ping pong is not included in rio 2016'
        # tokenize inputs
        tokenized_inputs = self.tokenizer.batch_encode_plus(
            [input_], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt", truncation=True
        )
        self.inputs.append(tokenized_inputs)

        # tokenize targets
        tokenized_targets = self.tokenizer.batch_encode_plus(
            [target_], max_length=128, pad_to_max_length=True, return_tensors="pt", truncation=True
        )
        self.targets.append(tokenized_targets)



if __name__=='__main__':
    tokenizer = T5Tokenizer.from_pretrained(t5_model)
    dataset = MedicalDialogueDataset(tokenizer, data_dir='./MedDG/input/', type_path='train')


    args_dict = dict(
        data_dir="", # path for data files
        output_dir="", # path to save the checkpoints
        model_name_or_path=t5_model,
        tokenizer_name_or_path=t5_model,
        max_seq_length=512,
        learning_rate=3e-4,
        weight_decay=0.0,
        adam_epsilon=1e-8,
        warmup_steps=0,
        train_batch_size=16,
        eval_batch_size=8,
        num_train_epochs=20,
        gradient_accumulation_steps=16,
        n_gpu=gpu_number,
        # early_stop_callback=False,
        fp_16=False, # if you want to enable 16-bit training then install apex and set this to true
        opt_level='apex', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
        max_grad_norm=1.0, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
        seed=42,
    )

    args_dict.update({'data_dir': './MedDG/input/',
                      'output_dir': './MedDG/output/',
                      'num_train_epochs': train_epochs})  ##  '/home/kangchen/Rehrearsal_TransferLearning/Data_Bank/process_translate/'
    args = argparse.Namespace(**args_dict)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.output_dir, monitor="val_loss", mode="min", save_top_k=5
    )

    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=args.n_gpu,  # args.n_gpu  None
        max_epochs=args.num_train_epochs,
        precision = 16 if args.fp_16 else 32,
        amp_backend=args.opt_level,
        gradient_clip_val=args.max_grad_norm,
        checkpoint_callback=checkpoint_callback,
        callbacks=[LoggingCallback()],
    )


    model = T5FineTuner(args)
    trainer = pl.Trainer(**train_params)
    trainer.fit(model)
