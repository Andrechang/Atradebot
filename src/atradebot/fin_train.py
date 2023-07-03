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

os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = "cuda" if torch.cuda.is_available() else "cpu"

HUB_MODEL = "achang/fin_forecast"
HUB_DATA = 'achang/stock_forecast'
OUTFOLDER = 'exp2'
IGNORE_INDEX = -100
MICRO_BATCH_SIZE = 4  # change to 4 for 3090
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
#from databricks/dolly-v2-3b
INSTRUCTION_KEY = "### Instruction:"
RESPONSE_KEY = "### Response:"
END_KEY = "### End"

def generate_prompt(data_point):
    # from https://github.com/tloen/alpaca-lora
    if data_point["instruction"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
            ### Instruction: {data_point["instruction"]}
            ### Input: {data_point["input"]}
            ### Response: {data_point["output"]} ### End"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
            ### Instruction: {data_point["instruction"]}
            ### Response: {data_point["output"]} ### End"""

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
        return tokenizer.decode(sequence[response_pos + 1 : ])

def main(args):
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
            generate_prompt(data_point),
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
        num_train_epochs=70,
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
        for in_data in tqdm(eval_dataloader):
            in_data['input_ids'] = in_data['input_ids'].to(device)
            with torch.cuda.amp.autocast():
                outputs = model.generate(input_ids = in_data['input_ids'], 
                    max_new_tokens=256,
                    top_p = 0.92,
                    top_k = 0,
                    do_sample = True,
                    early_stopping=True,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id)
            for output in outputs:
                print("sample: ", get_response(output.cpu().numpy(), tokenizer))

            break

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
    main(args)


