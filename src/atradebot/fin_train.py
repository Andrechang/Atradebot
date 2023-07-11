#model to generate news for a stock
#from: https://huggingface.co/dfurman/falcon-40b-chat-oasst1/blob/main/finetune_falcon40b_oasst1_with_bnb_peft.ipynb
#https://huggingface.co/blog/falcon
# https://colab.research.google.com/drive/1n5U13L0Bzhs32QO_bls5jwuZR62GPSwE?usp=sharing


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
from atradebot import main, news_util

os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = "cuda" if torch.cuda.is_available() else "cpu"

HUB_MODEL = "achang/fin_alloc" # Change here to use your own model
HUB_DATA = 'achang/stock_alloc'# Change here to use your own data
OUTFOLDER = 'exp3'
IGNORE_INDEX = -100
MICRO_BATCH_SIZE = 4  # change to 4 for 3090
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
#from databricks/dolly-v2-3b
INSTRUCTION_KEY = "### Instruction:"
RESPONSE_KEY = "### Response:"
END_KEY = "### End"

def generate_prompt(data_point, mode='train'):
    # from https://github.com/tloen/alpaca-lora
    prompt = ""
    if data_point["instruction"]:
        prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
            ### Instruction: {data_point["instruction"]}
            ### Input: {data_point["input"]}\n ### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
            ### Instruction: {data_point["instruction"]}\n ### Response:"""
    if mode == 'train':
            prompt += f""" {data_point["output"]} ### End"""
    return prompt 


def get_model(peft_model_id = HUB_MODEL):
    config = PeftConfig.from_pretrained(peft_model_id)
    model_id = config.base_model_name_or_path
    #load model and tokenizer with quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id,
                                            padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=bnb_config, 
        device_map="auto",
        trust_remote_code=True)
    #setup lora
    model = PeftModel.from_pretrained(model, peft_model_id)
    model.eval()
    return model, tokenizer



def get_response(sequence, tokenizer):
    response_key_token_id = tokenizer.encode(RESPONSE_KEY)
    end_key_token_id = tokenizer.encode(END_KEY)
    response_positions = np.where(sequence == response_key_token_id)[0]
    if len(response_positions) == 0:
        print(f"Could not find response key {response_key_token_id} in: {sequence}")
    else:
        response_pos = response_positions[0]

    end_positions = np.where(sequence == end_key_token_id)[0]
    if len(end_positions) > 0:
        end_pos = end_positions[0]
        decoded = tokenizer.decode(sequence[response_pos + 1 : end_pos]).strip()
        return decoded    
    else:
        return ""

def train_model(args):
    if args.mode == 'train':            
        model_id = "databricks/dolly-v2-3b"
        config = LoraConfig(
            r=8, 
            lora_alpha=32, 
            target_modules=["query_key_value"], #gpt_neox
            lora_dropout=0.05, 
            bias="none", 
            task_type="CAUSAL_LM"
        )
        #load model and tokenizer with quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id,
                                                padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            quantization_config=bnb_config, 
            device_map="auto",
            trust_remote_code=True)
        #setup lora
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, config)
    else:
        model, tokenizer = get_model(HUB_MODEL)

    #get data
    data = load_dataset(HUB_DATA)

    data = data.shuffle().map(
        lambda data_point: tokenizer(
            generate_prompt(data_point, mode=args.mode),
            truncation=True,
            padding="max_length",
        )
    )
    data = data["train"].train_test_split(test_size=0.1)

    #train: sequence of text and predict next token
    training_args = transformers.TrainingArguments(
        # auto_find_batch_size=True,
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=150,
        learning_rate=2e-5,
        fp16=True,
        save_total_limit=4,
        report_to='tensorboard',
        logging_dir=os.path.join(OUTFOLDER, "logs"),
        logging_steps=4,
        output_dir=os.path.join(OUTFOLDER, "outputs"),
        save_strategy='epoch',
        optim="paged_adamw_8bit",
        lr_scheduler_type = 'cosine',
        # evaluation_strategy = 'epoch',
        warmup_ratio = 0.01,
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=data['train'],
        eval_dataset=data['test'],
        args=training_args,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    if args.mode == 'train':
        trainer.train(resume_from_checkpoint = args.reload)
        
        metrics = trainer.evaluate(eval_dataset=data['test'])
        print(metrics)
        #save model
        model.push_to_hub(HUB_MODEL)
        model.save_pretrained(f'{OUTFOLDER}/best_model')

    else:
        model.to(device)
        data.set_format("torch")
        eval_dataloader = DataLoader(data["test"], batch_size=1)

        all_preds, all_targets = [], []
        for in_data in tqdm(eval_dataloader):
            in_data['input_ids'] = in_data['input_ids'].to(device)
            with torch.cuda.amp.autocast():
                outputs = model.generate(
                    input_ids = in_data['input_ids'],
                    attention_mask = in_data['attention_mask'], 
                    max_new_tokens=128,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id)
            for output in outputs:
                response = get_response(output.cpu().numpy(), tokenizer)
                pred = re.findall(r"[-+]?(?:\d*\.*\d+)", response)
                target = re.findall(r"[-+]?(?:\d*\.*\d+)", in_data['output'][0])
                print("sample: ", response)
                print("target: ", target)
                if len(pred) > 3:
                    pred = pred[:3]
                if len(pred) < 3:
                    continue
                all_preds += pred
                all_targets += target
            
        all_targets = [eval(i) for i in all_targets]
        all_preds = [eval(i) for i in all_preds]
        print("MSE: ", mean_squared_error(all_targets, all_preds))



class FinForecastStrategy:
    def __init__(self, start_date, end_date, data, stocks, cash=10000, model_id="achang/fin_forecast"):
        """
        model:
        data: achang/stock_forecast
        input: 
            Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
                ## Instruction: what is the forecast for ... 
                ## Input: date, news title, news snippet
        output:
                ## Response: forecast percentage change for 1mon, 5mon, 1 yr
        """

        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        self.prev_date = start_date #previous date for rebalance
        self.stocks = {i: 0 for i in stocks}
        
        self.cash = [cash*0.5, cash*0.3, cash*0.1, cash*0.1] #amount to invest in each rebalance
        self.cash_idx = 0

        self.start_date = start_date
        self.end_date = end_date
        delta = end_date - start_date
        self.days_interval = delta.days/len(self.cash) #rebalance 4 times
        self.data = data['Adj Close']

        self.model, self.tokenizer = get_model(model_id)
        self.model.to(device)

    def model_run(self, date, num_news=5):
        """get news before date and forecast using model

        Args:
            date (pandas.Timestamp): date = pd.Timestamp("2021-07-22")
            num_news (int, optional): number of news to get. Defaults to 5.

        Returns:
            dict{stock: list of pred }: prediction for each stock. Prediction format [1 mon, 5 mon, 1 yr]
        """

        end = date.date()
        start = main.business_days(end, -5)
        all_stocks = {}
        for stock in self.stocks:
            news, _, _ = news_util.get_google_news(stock=stock, num_results=num_news, time_period=[start, end])
            assert len(news) > 0, "no news found, google search blocked error"
            all_pred = []
            for new in news:
                in_dict = {'instruction': f"what is the forecast for {new['stock']}", 
                        'input':f"{new['date']} {new['title']} {new['snippet']}"}
                prompt = generate_prompt(in_dict, mode='eval')
                in_data = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length")
                in_data['input_ids'] = in_data['input_ids'].to(device)
                with torch.cuda.amp.autocast():
                    outputs = self.model.generate(input_ids = in_data['input_ids'], 
                        attention_mask = in_data['attention_mask'],
                        max_new_tokens=128,
                        pad_token_id= self.tokenizer.eos_token_id,
                        eos_token_id= self.tokenizer.eos_token_id
                    )

                response = get_response(outputs[0].cpu().numpy(), self.tokenizer)
                pred = re.findall(r"[-+]?(?:\d*\.*\d+)", response)
                print(f"{new['stock']} forecast: {response} \n {pred}")
                if len(pred) > 3:
                    pred = pred[:3]
                if len(pred) < 3:
                    continue
                
                pred = [eval(i) for i in pred] #convert to str to float
                all_pred.append(pred)

            #average forecast
            avg_pred = np.mean(np.array(all_pred), axis=0)
            all_stocks[stock] = avg_pred
        return all_stocks

    def model_allocation(self, date, amount_invest, prices, portfolio, sell_mode=False):
        """get allocation for each stock based on model predictions
        Args:
            date (pandas.Timestamp): date = pd.Timestamp("2021-07-22")
            amount_invest (int): amount to invest
            prices (pandas.Series): list of stock prices
            portfolio (dict{stock: number of shares}): current portfolio
            sell_mode (bool, optional): whether to sell stocks. Defaults to False.
        Returns:
            dict{stock: number to buy/sell}: allocation for each stock
        """        
        all_stocks = self.model_run(date)
        #pick top increasing forecast
        future_mode = 0 #choose timeline 1mon, 5mon, 1yr
        alloc = sorted(all_stocks.items(), key=lambda x: x[1][future_mode], reverse=True) #most increase first 
        # alloc: tuple(stock, forecast)
        weights = [0.6, 0.3, 0.1] #weight for each stock
        amounts = [int(amount_invest * weight) for weight in weights]
        allocation = {i[0]: int(amounts[idx]/prices[i[0]]) for idx, i in enumerate(alloc[:3])}#get top3 stocks

        #get top stocks to sell
        if sell_mode:
            alloc_sell = sorted(all_stocks.items(), key=lambda x: x[1][future_mode], reverse=False) #most decrease first 
            weights_sell = 0.5 #weight to sell holdings 
            for sell in alloc_sell:
                if portfolio[sell[0]][date] > 0:
                    amounts_sell = int(portfolio[sell[0]][date]*weights_sell)
                    allocation[sell[0]] = -amounts_sell
                    break

        leftover = amount_invest - sum([prices[i[0]]*allocation[i[0]] for i in alloc[:3]])
        return allocation, leftover
        #TODO balance with max_sharpe_allocation
        

    def generate_allocation(self, date, portfolio):
        """generate allocation for each stock using average cost investment method
        Args:
            date (pandas.Timestamp): date = pd.Timestamp("2021-07-22")

        Returns:
            dict{stock: number to buy/sell}: allocation for each stock
        """        
        delta = date.date() - self.prev_date
        idx = self.data.index.get_loc(str(date.date()))
        if date.date() == self.prev_date: #first day
            # allocation, leftover = max_sharpe_allocation(self.data[0:idx], amount_invest=self.cash[self.cash_idx])
            allocation, leftover = self.model_allocation(date, amount_invest=self.cash[self.cash_idx], 
                prices=self.data.iloc[idx], portfolio=portfolio, sell_mode=False)
            self.cash_idx += 1                        
            return allocation
        elif delta.days > self.days_interval: #rebalance 
            # allocation, leftover = max_sharpe_allocation(self.data[0:idx], amount_invest=self.cash[self.cash_idx])
            allocation, leftover = self.model_allocation(date, amount_invest=self.cash[self.cash_idx], 
                prices=self.data.iloc[idx], portfolio=portfolio, sell_mode=True)
            self.prev_date = date.date()
            self.cash_idx += 1
            return allocation
        else:
            return self.stocks


def get_parser(raw_args=None):
    parser = ArgumentParser(description="model")
    parser.add_argument('--reload', action="store_true",
                        help='to reload checkpoint')
    parser.add_argument('--mode', type=str, default='eval',
                        help='train or eval')
    args = parser.parse_args(raw_args)
    return args

if __name__ == "__main__":
    args = get_parser()
    train_model(args)


