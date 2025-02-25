# finetune llama-3 https://medium.com/@alexandros_chariton/how-to-fine-tune-llama-3-2-instruct-on-your-own-data-a-detailed-guide-e5f522f397d7
# https://github.com/AI4Finance-Foundation/FinGPT/tree/master/fingpt/FinGPT_Forecaster


from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    logging,
)
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)
import os, torch
from datasets import load_dataset, Dataset
from argparse import ArgumentParser
import bitsandbytes as bnb
from functools import partial
from atradebot.data.llm_prompt import format_chat_template

torch_dtype = torch.float16
attn_implementation = "eager"
BASEMODEL = "meta-llama/Llama-3.2-3B-Instruct"
LORAMODEL = "achang/llama-3.2-3b_lora"
DATASET_NAME = "achang/test_stock"
RESPONSE_TOKENS = '<|start_header_id|>assistant<|end_header_id|>\n\n'

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:  # needed for 16 bit
        lora_module_names.remove('lm_head')
        return list(lora_module_names)

def tokenize_function(max_length, tokenizer, example):
    
    tgt_id = example['text'].find(RESPONSE_TOKENS)
    prompt_ids = tokenizer.encode(example['text'][:tgt_id].strip(), padding=False,
        max_length=max_length, truncation=True)
    
    target_ids = tokenizer.encode(example['text'][tgt_id:].strip(), padding=False, 
                                  max_length=max_length, truncation=True, add_special_tokens=False)
    
    input_ids = prompt_ids + target_ids
    exceed_max_length = len(input_ids) >= max_length
    if input_ids[-1] != tokenizer.eos_token_id and not exceed_max_length: # Add EOS Token
        input_ids.append(tokenizer.eos_token_id)
    label_ids = [tokenizer.pad_token_id] * len(prompt_ids) + input_ids[len(prompt_ids):]
    return {
        "input_ids": input_ids,
        "labels": label_ids,
        "exceed_max_length": exceed_max_length
    }

def train_model(args):
    # QLoRA config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=True,
    )
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        BASEMODEL,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation=attn_implementation
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASEMODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Dataset 
    dataset = load_dataset(DATASET_NAME, split="train")
    dataset = dataset.shuffle(seed=65) # Only use 1000 samples for quick demo
    dataset = dataset.train_test_split(test_size=0.2)
    dataset = dataset.map(partial(format_chat_template, tokenizer),num_proc= 4,)

    # Tokenize the data
    tokenized_dataset = dataset.map(partial(tokenize_function, args.max_length, tokenizer))
    tokenized_dataset = tokenized_dataset.filter(lambda x: not x['exceed_max_length'])
    tokenized_dataset = tokenized_dataset.remove_columns(['instruction', 'response', 'text', 'exceed_max_length'])

    modules = find_all_linear_names(model)
    # LoRA config #TODO: https://arxiv.org/pdf/2501.06252 SVF is better than lora
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=modules
    )
    model = get_peft_model(model, peft_config)
    #Hyperparamter   
    training_arguments = TrainingArguments(
        output_dir='./output_dir',
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        optim="paged_adamw_32bit",
        num_train_epochs=1,
        eval_strategy="steps",
        eval_steps=0.2,
        warmup_steps=5,
        logging_strategy="steps",
        learning_rate=2e-4,
        max_grad_norm=2,
        fp16=True,
        report_to="tensorboard"
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer)

    trainer.train()
    trainer.model.push_to_hub(LORAMODEL)


def test_model(args):
    model = AutoModelForCausalLM.from_pretrained(
            BASEMODEL,
            trust_remote_code=True, 
            device_map="auto",
            torch_dtype=torch.float16,
    )
    # model = PeftModel.from_pretrained(bmodel, new_model) # TODO: maybe dont need train
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(BASEMODEL)
    dataset = load_dataset(DATASET_NAME, split="train").select(range(10))
    dataset = dataset.map(partial(format_chat_template, tokenizer), num_proc= 4)

    prompt = dataset[0]["text"]
    ii = prompt.find(RESPONSE_TOKENS)
    prompt = prompt[:ii]
    inputs = tokenizer(prompt, return_tensors='pt', padding=False)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    
    res = model.generate(
            **inputs, max_length=4096, do_sample=True,
            eos_token_id=tokenizer.eos_token_id, use_cache=True)
    output = tokenizer.decode(res[0], skip_special_tokens=True)
    print(prompt)
    print("Output: ======================")
    print(output)

    

def get_parser(raw_args=None):
    parser = ArgumentParser(description="train model")
    parser.add_argument('--mode', type=str, default='eval',
                        help='train or eval')   
    parser.add_argument("--max_length", default=1024, type=int)
    args = parser.parse_args(raw_args)
    return args

if __name__ == "__main__":
    args = get_parser()
    if args.mode == 'train':
        train_model(args)
    else:
        test_model(args)