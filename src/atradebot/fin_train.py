# train a neural net model on Yahoo Finance data API, Google News

import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
from datasets import load_dataset
from tqdm import tqdm
from argparse import ArgumentParser
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import mean_squared_error 
import re
from datetime import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = "cuda" if torch.cuda.is_available() else "cpu"

OUTFOLDER = 'exp'
IGNORE_INDEX = -100
MICRO_BATCH_SIZE = 4  # change to 4 for 3090
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE

INSTRUCTION_KEY = "### Instruction:"
RESPONSE_KEY = "### Response:"
END_KEY = "### End"


def generate_prompt(data_point, mode='train'):
    """generate prompt for training or eval
    https://github.com/tloen/alpaca-lora

    :param data_point: text data with instruction, input, output
    :type data_point: dict{"instruction": str, "input": str, "output": str}
    :param mode: generate prompt for train or eval, defaults to 'train'
    :type mode: str, optional
    :return: prompt to fed model
    :rtype: str
    """    
    prompt = ""
    if data_point["instruction"]:
        prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. \
            Write a response that appropriately completes the request.
            ### Instruction: {data_point["instruction"]}
            ### Input: {data_point["input"]}\n ### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
            ### Instruction: {data_point["instruction"]}\n ### Response:"""
    if mode == 'train':
            prompt += f""" {data_point["output"]} ### End"""
    return prompt 

def get_str2date(response:str):
    """get date from response

    :param response: string following the fin_train.py generate_prompt format
    :type response: str
    :return: date
    :rtype: datetime.date
    """    
    dx = response.find("### Input:")
    match = re.search(r'\d{4}-\d{2}-\d{2}', response[dx:])
    date = datetime.strptime(match.group(), '%Y-%m-%d').date()
    return date

def get_response(sequence, tokenizer):
    """get response from model output sequence

    :param sequence: output sequence from model
    :type sequence: str
    :param tokenizer: model tokenizer
    :type tokenizer: Tokenizer
    :return: only response part of the sequence
    :rtype: str
    """    
    decoded_str = tokenizer.decode(sequence)
    resp = decoded_str.find(RESPONSE_KEY)
    end = decoded_str[resp:].find(END_KEY)
    if resp == -1 and end != -1:
        return decoded_str[: resp + end]
    elif resp == -1 and end == -1:
        return ''
    else:
        return decoded_str[resp + len(RESPONSE_KEY) : resp + end]

def get_str2alloc(response:str):
    """get allocation for backtesting from response

    :param response: response from model
    :type response: str
    :return: allocation generated from response
    :rtype: dict{stock str: stocks buy/sell int}
    """    
    nums = re.findall(r"[-+]?(?:\d*\.*\d+)", response)
    keys = re.findall(r'\'(.*?)\'', response)
    alloc = {}
    for i, k in enumerate(keys):
        alloc[k] = nums[i]
    return alloc


def get_slm_model(model_name):
    """get a small language model

    :param model_name: id of model from huggingface
    :type model_name: str
    :return:  model, tokenizer
    :rtype:  ModelForCausalLM, Tokenizer
    """    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def train_model(args):
    if args.mode == 'train':
        model, tokenizer = get_slm_model("gpt2")
    else:
        model, tokenizer = get_slm_model(args.mhub)
        model.eval()

    #get data
    data = load_dataset(args.dhub)

    data = data.shuffle().map(
        lambda data_point: tokenizer(
            generate_prompt(data_point, mode=args.mode),
            truncation=True,
            max_length=512,
            padding="max_length",
        )
    )
    data = data["train"].train_test_split(test_size=0.1)

    #train: sequence of text and predict next token
    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=50,
        learning_rate=2e-5,
        fp16=True,
        save_total_limit=4,
        report_to='tensorboard',
        logging_dir=os.path.join(OUTFOLDER, "logs"),
        logging_steps=4,
        output_dir=os.path.join(OUTFOLDER, "outputs"),
        save_strategy='epoch',
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
        model.push_to_hub(args.mhub)
        model.save_pretrained(f'{OUTFOLDER}/best_model')

    else:
        model.to(device)
        data.set_format("torch")
        eval_dataloader = DataLoader(data["test"], batch_size=1)

        all_preds, all_targets = [], []
        for in_data in tqdm(eval_dataloader):
            in_data['input_ids'] = in_data['input_ids'].to(device)
            in_data['attention_mask'] = in_data['attention_mask'].to(device)
            with torch.cuda.amp.autocast():
                outputs = model.generate(
                    input_ids = in_data['input_ids'],
                    attention_mask = in_data['attention_mask'], 
                    max_new_tokens=32,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id)
            for output in outputs:
                response = get_response(output.cpu().numpy(), tokenizer)
                target = in_data['output'][0] 
                print("sample: ", response)
                print("target: ", target)
                all_preds += response
                all_targets += target
        
        # metric_forecast_task(all_targets, all_preds) #for forecast models

def metric_forecast_task(all_targets, all_preds):
    atgt, apred = [], []
    for target, response in zip(all_targets, all_preds):
        pred = re.findall(r"[-+]?(?:\d*\.*\d+)", response)
        tgt = re.findall(r"[-+]?(?:\d*\.*\d+)", target)     
        if len(pred) > 3:
            pred = pred[:3]
        if len(pred) < 3:
            continue
        atgt += tgt
        apred += pred
    atgt = [eval(i) for i in atgt]
    apred = [eval(i) for i in apred]
    print("MSE: ", mean_squared_error(atgt, apred))

def get_parser(raw_args=None):
    parser = ArgumentParser(description="train model")
    parser.add_argument('--reload', action="store_true",
                        help='to reload checkpoint')
    parser.add_argument('--mode', type=str, default='eval',
                        help='train or eval')   
    parser.add_argument('-d', '--dhub', type=str,
                        default='achang/stocks_one_nvda_v2', help='get from hub folder name for task dataset')
    parser.add_argument('-m', '--mhub', type=str,
                        default='achang/fin_falcon7b_one_nvda_v2', help='push to hub folder model')
    args = parser.parse_args(raw_args)
    return args


if __name__ == "__main__":
    args = get_parser()
    train_model(args)
