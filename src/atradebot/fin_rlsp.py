#RL finetune on stock price
# from: https://huggingface.co/docs/trl/quickstart


import torch
from transformers import GPT2Tokenizer
from argparse import ArgumentParser
import os
import sys
import time
from pathlib import Path
import shutil

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import transformers
from datasets import load_dataset
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig, prepare_model_for_kbit_training, LoraConfig, get_peft_model
from tqdm import tqdm
import numpy as np
from argparse import ArgumentParser
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import mean_squared_error 
import re
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
from atradebot import main, fin_train, backtest, utils
import copy
import pandas as pd
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from atradebot import fin_train

OUTFOLDER = 'exp3_rl'

def calc_reward(alloc, date, add=30):
    future_date = utils.business_days(date, add)
    cash = 0 #value
    for stock, qnt in alloc.items():
        qnt = int(qnt)
        hdata = utils.get_price_date(date, future_date, stock)
        if qnt > 0: #buy now and value is in future
            cash += hdata['Close'][-1]*qnt
        elif qnt < 0: #sell now value is now
            cash += hdata['Close'][0]*abs(qnt)
    return cash


def train_rl_model(args):
    # 1. load a pretrained model
    #https://github.com/lvwerra/trl/blob/main/examples/multi-adapter-rl/rl_finetuning.py
    model_name = "achang/fin_alloc_small0"#"gpt2"
    token_id = "gpt2"
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(token_id, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    # 2. initialize trainer
    data = load_dataset(args.dhub)

    def build_dataset(data_point):
        prompt = fin_train.generate_prompt(data_point, mode='eval')
        sample = tokenizer(
            prompt,
            truncation=True,
            max_length=512,
            padding="max_length",
        )
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    data = data.shuffle().map(build_dataset)
    data.set_format(type="torch")

    config = PPOConfig(
        learning_rate=1e-5,
        batch_size=1,
        mini_batch_size=1,
        gradient_accumulation_steps=4,
        optimize_cuda_cache=True,
        log_with="tensorboard",
        project_kwargs={'logging_dir':OUTFOLDER},
    )

    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])
    ppo_trainer = PPOTrainer(
        config,
        model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=data['train'],
        data_collator=collator,
    )

    # 4. generate model response
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 64,
    }
    
    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        question_tensors = batch["input_ids"]
        response_tensors = ppo_trainer.generate(question_tensors, return_prompt=False, **generation_kwargs)

        # Compute reward score
        rewards = []
        for batch_id, output in enumerate(response_tensors):
            response = fin_train.get_response(output.cpu().numpy(), tokenizer)
            alloc = fin_train.get_str2alloc(response)
            date = tokenizer.decode(question_tensors[batch_id])
            date = fin_train.get_str2date(date)
            reward = calc_reward(alloc, date, add=20)#apply alloc and check gain after x days
            rewards.append(torch.tensor(reward, dtype=torch.float64))

        # rewards = [torch.tensor(0, dtype=torch.float64) for i in range(len(question_tensors))]

        # Run PPO step
        stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
        print(rewards)
        ppo_trainer.log_stats(stats, batch, rewards)


    model.push_to_hub("achang/stock_rl_alloc_small0")


def get_parser(raw_args=None):
    parser = ArgumentParser(description="model")
    parser.add_argument('--reload', action="store_true",
                        help='to reload checkpoint')
    parser.add_argument('--mode', type=str, default='eval',
                        help='train or eval')    
    parser.add_argument('-d', '--dhub', type=str,
                        default='achang/stock_alloc', help='get from hub folder name for task dataset')
    parser.add_argument('-m', '--mhub', type=str,
                        default='achang/fin_alloc_0', help='push to hub folder model')
    args = parser.parse_args(raw_args)
    return args


if __name__ == "__main__":
    args = get_parser()
    train_rl_model(args)
