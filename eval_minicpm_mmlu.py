from modelscope import AutoModelForCausalLM as Mamclm, AutoTokenizer as Mat
from modelscope.msdatasets import MsDataset
import torch
import numpy as np
import time

print("-"*50)
print("Retrieve Data")

choices = ["A", "B", "C", "D"]
categories = [
    "high_school_european_history",
    "business_ethics",
    "clinical_knowledge",
    "medical_genetics",
    "high_school_us_history",
    "high_school_physics",
    "high_school_world_history",
    "virology",
    "high_school_microeconomics",
    "econometrics",
    "college_computer_science",
    "high_school_biology",
    "abstract_algebra",
    "professional_accounting",
    "philosophy",
    "professional_medicine",
    "nutrition",
    "global_facts",
    "machine_learning",
    "security_studies",
    "public_relations",
    "professional_psychology",
    "prehistory",
    "anatomy",
    "human_sexuality",
    "college_medicine",
    "high_school_government_and_politics",
    "college_chemistry",
    "logical_fallacies",
    "high_school_geography",
    "elementary_mathematics",
    "human_aging",
    "college_mathematics",
    "high_school_psychology",
    "formal_logic",
    "high_school_statistics",
    "international_law",
    "high_school_mathematics",
    "high_school_computer_science",
    "conceptual_physics",
    "miscellaneous",
    "high_school_chemistry",
    "marketing",
    "professional_law",
    "management",
    "college_physics",
    "jurisprudence",
    "world_religions",
    "sociology",
    "us_foreign_policy",
    "high_school_macroeconomics",
    "computer_security",
    "moral_scenarios",
    "moral_disputes",
    "electrical_engineering",
    "astronomy",
    "college_biology",
]

## Data can be found at ~/.cache/modelscope/mmlu/marketing
# ds : dict = MsDataset.load('mmlu', subset_name='marketing', split='test')
# print("Type of dataset: ", type(ds))
# print("Length of dataset: ", len(ds))
# print("Example of dataset: ", next(iter(ds)))
# print("Example using index: ", ds[0])
# print("Length of example", len(ds[0]))  # {"input" : "...", "A": "...", "B": "...", "C": "...", "D": "...", "target: "...}


# Load model
print("-"*50)
print("Loading model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "OpenBMB/MiniCPM-2B-dpo-fp32"
tokenizer = Mat.from_pretrained(model_path, trust_remote_code=True)
model = Mamclm.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True)
print("Model loaded...")
print("-"*50)


# Try chat
# start_time = time.time()
# input = next(iter(ds))["input"]
# responds, history = model.chat(tokenizer, input, temperature=0.8, top_p=0.8)
# end_time = time.time()
# print(f"{[end_time - start_time]}: {responds}")

def format_subject(subject:str):
    l = subject.split("_")
    return " ".join([x.capitalize() for x in l])

def format_example(df, index, include_answer=True):
    prompt = df[index]["input"]
    k = len(df[index]) - 2 # number of choices
    for j in range(k): # iterate over 4 choices
        prompt += "\n{}: {}".format(choices[j], df[index][choices[j]])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df[index]["target"])
    return prompt

def generate_prompt(df, subject, k=1):
    """
    Prompt generation for multiple choice questions with answers. 
    Form few-shot prompts in the recommended format
    """
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = len(df) # total number of questions
    for i in range(k):
        prompt += format_example(df, i)
    return prompt


def baseline(model, tokenizer):
    """ 
    Perform baseline test and get score 
    """
    for subject in categories:
        df_test = MsDataset.load('mmlu', subset_name=subject, split='test') # for testing
        df_dev = MsDataset.load('mmlu', subset_name=subject, split='train') # for prompt tuning
        print(f"Subject: {subject}, Test Length: {len(df_test)}, Dev Length: {len(df_dev)}")

        model.eval()
        
        cors = [] # correct responses?
        all_probs = [] # 
        answers = choices[: len(df_test[0]) - 2 ] # ["A", "B", "C", "D"]
        
        correct = 0
        for i in range(len(df_test)):
            # Create proper prompt
            k = 0 # number of examples, more examples for more prompting
            prompt_end = format_example(df_test, i, include_answer=False)
            train_prompt = generate_prompt(df_dev, subject, k)
            prompt = train_prompt + prompt_end
            
            print("------Prompt------")
            print(prompt)
            print("------Prompt------")

            # Logic
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)
            outputs = model.generate(input_ids, max_length=250, attention_mask=attention_mask)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            correct_answer = df_test[i]["target"]
            
            print("---Response---")
            print(f"Response: {response}, Correct Answer: {correct_answer}")
            print("---Response---")
            print("-" * 50)

            if (response.strip().lower() == correct_answer.strip()):
                correct += 1
        
        
        print(f"Total correct for {subject}: {correct}")      
                
                  
        # TODO Show result of one subject for now
        break
    
            
def prompt_tuning():
    """
    Perform CoT prompt tuning and get score
    """
    
def fine_tuning():
    """
    Perform fine-tuning and get score
    """
    
def delta_fine_tuning():
    """
    Perform delta fine-tuning and get score
    """
    
    
if __name__ == "__main__":
    baseline(model, tokenizer)
    # prompt_tuning()
    # fine_tuning()
    # delta_fine_tuning()
    
    
    
exit()
import argparse
import json
import os
import time

import pandas as pd
import tensor_parallel as tp
import torch
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM

TASKS = [
        'abstract_algebra',
        'anatomy',
        'astronomy',
        'business_ethics',
        'clinical_knowledge',
        'college_biology',
        'college_chemistry',
        'college_computer_science',
        'college_mathematics',
        'college_medicine',
        'college_physics',
        'computer_security',
        'conceptual_physics',
        'econometrics',
        'electrical_engineering',
        'elementary_mathematics',
        'formal_logic',
        'global_facts',
        'high_school_biology',
        'high_school_chemistry',
        'high_school_computer_science',
        'high_school_european_history',
        'high_school_geography',
        'high_school_government_and_politics',
        'high_school_macroeconomics',
        'high_school_mathematics',
        'high_school_microeconomics',
        'high_school_physics',
        'high_school_psychology',
        'high_school_statistics',
        'high_school_us_history',
        'high_school_world_history',
        'human_aging',
        'human_sexuality',
        'international_law',
        'jurisprudence',
        'logical_fallacies',
        'machine_learning',
        'management',
        'marketing',
        'medical_genetics',
        'miscellaneous',
        'moral_disputes',
        'moral_scenarios',
        'nutrition',
        'philosophy',
        'prehistory',
        'professional_accounting',
        'professional_law',
        'professional_medicine',
        'professional_psychology',
        'public_relations',
        'security_studies', 
        'sociology',
        'us_foreign_policy',
        'virology',
        'world_religions']

choices = ["A", "B", "C", "D"]

def compute_metric(output_filename):
    with open(output_filename, 'r') as f:
        run_results = json.load(f)
    total_acc = 0
    total_num = 0
    for task in run_results:
        acc = 0
        pred_answers = run_results[task]['pred_answers']
        gold_answers = run_results[task]['gold_answers']
        for pred, gold in zip(pred_answers, gold_answers):
            if pred == gold: acc += 1
        print("ACC-%s: %.4f" % (task, acc/len(gold_answers)))
        total_acc += acc
        total_num += len(gold_answers)
    print("ACC-all: %.4f" % (total_acc/total_num))


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt

def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


# def custom_stopping_criteria(input_ids, score, **kwargs):
#     stop_ids = [29871, 13, 13] # \n\n 
#     return input_ids[-len(stop_ids)]

def prepare_input(tokenizer, prompts):
    input_tokens = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding=True)
    input_tokens = {k:input_tokens[k] for k in input_tokens if k in ["input_ids", "attention_mask"]}
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to('cuda')

    return input_tokens

def load(ckpt_dir, model_type):
    n_gpus = torch.cuda.device_count()

    if model_type == 'llama':
        # we use tensor parallel for loading llama
        tokenizer = LlamaTokenizer.from_pretrained(ckpt_dir, use_fast=False, padding_side="left")
        
        model = LlamaForCausalLM.from_pretrained(ckpt_dir, low_cpu_mem_usage = True, torch_dtype=torch.float16)
        model = tp.tensor_parallel(model, [i for i in range(n_gpus)])

        tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        tokenizer.bos_token_id = 1
    else:
        # mpt-30b's tokenizer only has the fast version
        use_fast = "mosaicml/mpt-30b" in ckpt_dir
        # however, tensor parallel for running falcon will occur bugs
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=use_fast, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(ckpt_dir, device_map = 'balanced_low_0', torch_dtype=torch.bfloat16, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                tokenizer.pad_token_id = 0


    model.eval()

    return model, tokenizer

def batch_split(prompts, batch_num):
    batch_prompts = []
    mini_batch = []
    for prompt in prompts:
        mini_batch.append(prompt)
        if len(mini_batch) == batch_num:
            batch_prompts.append(mini_batch)
            mini_batch = []
    if len(mini_batch) != 0:
        batch_prompts.append(mini_batch)
    return batch_prompts

def batch_infer(model, tokenizer, prompts):
    batch_size = 8
    answers = []
    for batch_input in tqdm(batch_split(prompts, batch_size)):
        encode_inputs = prepare_input(tokenizer, batch_input)
        outputs = model.generate(**encode_inputs, max_new_tokens=1, pad_token_id=tokenizer.pad_token_id)
        answers.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    answers = [answer[-1] for answer in answers]
    return answers

def main(ckpt_dir: str, param_size: str, model_type: str):
    
    run_results = {}
    output_filename = 'run_results_%s_%sb.json' % (model_type, param_size)
    
    model, tokenizer = load(ckpt_dir, model_type)
    start_time = time.time()
    for task in TASKS:
        print('Testing %s ...' % task)
        records = []
        dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", task + "_dev.csv"), header=None)[:args.ntrain]
        test_df = pd.read_csv(os.path.join(args.data_dir, "test", task + "_test.csv"), header=None)
        for i in range(test_df.shape[0]):
            # get prompt and make sure it fits
            k = args.ntrain
            prompt_end = format_example(test_df, i, include_answer=False)
            train_prompt = gen_prompt(dev_df, task, k)
            prompt = train_prompt + prompt_end
            while len(tokenizer.tokenize(prompt)) + 1> 2048: # bos token
                prompt_split = prompt.split("\n\n")
                prompt_split.pop(1)
                prompt = '\n\n'.join(prompt_split)
            label = test_df.iloc[i, test_df.shape[1]-1]
            records.append({'prompt':prompt, 'answer':label})

        pred_answers = batch_infer(model, tokenizer, [record['prompt'] for record in records])
        gold_answers = [record['answer'] for record in records]
        run_results[task] = {'pred_answers':pred_answers, 'gold_answers':gold_answers}
    with open(output_filename, 'w') as f:
        json.dump(run_results, f, ensure_ascii=False, indent=2)
    
    compute_metric(output_filename)
    end_time = time.time()
    print("total run time %.2f" % (end_time - start_time))
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--param_size', type=str, required=True)
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--ntrain', type=int, default=5)
    args = parser.parse_args()
    
    main(args.ckpt_dir, args.param_size, args.model_type)