
# Creating tkinter GUI
import tkinter
from tkinter import *


# Import libraries
import argparse
import os
import random
import nltk
# nltk.download('punkt')

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = T5Tokenizer.from_pretrained('t5-small')


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
            self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    def _step(self, batch):
        labels = batch["target_ids"]
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=labels,
            decoder_attention_mask=batch['target_mask']
        )
        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log('avg_val_loss', avg_train_loss, logger=True)
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs}  # , 'progress_bar': tensorboard_logs

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        tensorboard_logs = {"val_loss": loss}
        return {"val_loss": loss, "log": tensorboard_logs}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log('avg_val_loss', avg_loss, logger=True)
        tensorboard_logs = {"avg_val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}  # , 'progress_bar': tensorboard_logs

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

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
        if self.trainer.use_tpu:
            # xm.optimizer_step(optimizer)
            ha = 1
        else:
            optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}
        return tqdm_dict


# Function to convert
def listToString(s):
    # initialize an empty string
    str1 = " "
    # return string
    return (str1.join(s))


def send():
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)
    if msg != '':
        msg_doctor, msg_patient = msg.split('*')
        ChatBox.config(state=NORMAL)
        ChatBox.insert(END, "Doctor: " + msg_doctor + '\n\n')
        ChatBox.config(foreground="#446665", font=("Verdana", 12))

        # ************************************************************
        # predict the text
        msg_ = "Doctor: %s </s> Patients: %s" % (msg_doctor, msg_patient)
        input_ = "%s" % (msg_)
        tokenized_inputs = tokenizer.batch_encode_plus(
            [input_], max_length=200, pad_to_max_length=True, return_tensors="pt", truncation=True
        )
        outs = model.model.generate(input_ids=tokenized_inputs['input_ids'],
                                    attention_mask=tokenized_inputs['attention_mask'],
                                    max_length=100)
        res = [tokenizer.decode(ids) for ids in outs]
        # ************************************************************
        ChatBox.insert(END, "Patients: " + msg_patient + '\n' + listToString(res) + '\n\n')
        ChatBox.config(state=DISABLED)
        ChatBox.yview(END)


# ********************************************
# Validate
# ********************************************
args_dict = dict(
    data_dir="",  # path for data files
    output_dir="",  # path to save the checkpoints
    model_name_or_path='t5-small',
    tokenizer_name_or_path='t5-small',
    max_seq_length=512,
    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    train_batch_size=1,
    eval_batch_size=2,
    num_train_epochs=2,
    gradient_accumulation_steps=16,
    n_gpu=None,
    # early_stop_callback=False,
    fp_16=False,  # if you want to enable 16-bit training then install apex and set this to true
    opt_level='O1',
    # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    max_grad_norm=1.0,  # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    seed=42,
)

args_dict.update({'data_dir': './MEdDG/input/',
                  'output_dir': './MEdDG/output/',
                  'num_train_epochs': 1})
args = argparse.Namespace(**args_dict)
model = T5FineTuner(args)
ckpt = torch.load('./MEdDG/output/epoch=28-step=11251.ckpt', map_location=torch.device(device))
model.load_state_dict(ckpt['state_dict'])
model.eval()

if __name__ == '__main__':
    # ***************************************************************************************
    root = Tk()
    root.title("Chatbot")
    root.geometry("400x500")
    root.resizable(width=FALSE, height=FALSE)

    # Create Chat window
    ChatBox = Text(root, bd=0, bg="white", height="8", width="50", font="Arial", )

    ChatBox.config(state=DISABLED)

    # Bind scrollbar to Chat window
    scrollbar = Scrollbar(root, command=ChatBox.yview, cursor="heart")
    ChatBox['yscrollcommand'] = scrollbar.set

    # Create Button to send message
    SendButton = Button(root, font=("Verdana", 12, 'bold'), text="Send", width="12", height=5,
                        bd=0, bg="#f9a602", activebackground="#3c9d9b", fg='#000000',
                        command=send)

    # Create the box to enter message
    EntryBox = Text(root, bd=0, bg="white", width="29", height="5", font="Arial")
    # EntryBox.bind("<Return>", send)

    # Place all components on the screen
    scrollbar.place(x=376, y=6, height=386)
    ChatBox.place(x=6, y=6, height=386, width=370)
    EntryBox.place(x=128, y=401, height=90, width=265)
    SendButton.place(x=6, y=401, height=90)
    root.mainloop()



