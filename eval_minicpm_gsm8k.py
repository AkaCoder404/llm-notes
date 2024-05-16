from transformers import AutoTokenizer, AutoModelForCausalLM
from modelscope import AutoTokenizer as Mat, AutoModelForCausalLM as MamfcLM
from modelscope.msdatasets import MsDataset
import torch
import datasets
from tqdm import tqdm
import re

# Loading the gsm8k dataset from modelscope
gsm8k_ms_ds = MsDataset.load("gsm8k")
test_ds : MsDataset = gsm8k_ms_ds["test"]
train_ds: MsDataset = gsm8k_ms_ds["train"]
print("Test Dataset", len(test_ds)) # use for evaluating
print("Train Dataset", len(train_ds)) # use this as dev for few-shot prompting

# Loading the model
model_name = "OpenBMB/MiniCPM-2B-dpo-fp32"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = Mat.from_pretrained(model_name, trust_remote_code=True)
model = MamfcLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True)


def format_example(ds, index, include_answer=True):
  """
  [Question]
  Answer:
  """
  prompt = ds[index]["question"]
  prompt += "\nAnswer:"
  if include_answer:
    prompt += " {}\n\n".format(ds[index]["answer"])
  return prompt

def generate_prompt(ds, k=1):
  """
  Prompt generation for question-answer questions with gsm8k
  
  """
  # prompt = "Act as a grade school math teacher and score the following problem solution.\nQuestion:"
  prompt = "Below is an instruction that describes a task. \nWrite a response that appropriately completes the request.\n\nQuestion:\n"
  for i in range(k):
    prompt += format_example(ds, i)
  return prompt

def extract_first_answer(text):
    """
    Tool to extract the first answer block from the text.
    """
    # Define the pattern to match the answer block ending with #### followed by a number
    pattern = r"(.*?#### \d+)"
    
    # Search for the pattern in the text
    match = re.search(pattern, text, re.DOTALL)
    
    # If a match is found, return the matched text
    if match:
        return match.group(1)
    else:
        return -1
    
# Running
temperature = 0.8
top_p = 0.95
max_new_tokens = 300
count = 0
correct = 0
no_answer = 0

# Increase the batch size to speed up the processing
batch_size = 4  # Set the desired batch size
num_prompts = len(test_ds)  # Get the total number of prompts
num_batches = (num_prompts + batch_size - 1) // batch_size  # Calculate the number of batches

for batch_idx in tqdm(range(num_batches)):
    # Get the indices for the current batch
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, num_prompts)
    batch_indices = range(start_idx, end_idx)
    
    # Ensure the tokenizer has a padding token
    if tokenizer.pad_token is None:
        # Option 1: Set the eos_token as the padding token (if available)
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        # Option 2: Add a new padding token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            # Resize the model's token embeddings
            model.resize_token_embeddings(len(tokenizer))


    # Inference
    prompt_start = generate_prompt(train_ds, k=5) # k=1 for one example
    # Prepare the inputs for the current batch
    batch_prompts = [prompt_start + format_example(test_ds, idx, include_answer=False) for idx in batch_indices]
    # TODO max_length = 512, default is 1024
    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    # Generate responses for the current batch
    outputs = model.generate(input_ids=input_ids,
                                attention_mask=attention_mask,
                                max_new_tokens=max_new_tokens,
                                top_p=top_p,
                                temperature=temperature)
    batch_responses = [tokenizer.decode(output[input_ids[i].shape[0]:], skip_special_tokens=True) for i, output in enumerate(outputs)]

    # Process the responses for the current batch
    for idx, response in zip(batch_indices, batch_responses):
        # Get the first answer block from the response
        answer_block = extract_first_answer(response)
        if answer_block == -1:
            no_answer += 1

        try:
            # Compare final answer with the actual answer (EM score)
            predicted_answer = answer_block.split("#### ")[1].strip() if answer_block != -1 else -1
            actual_answer = extract_first_answer(test_ds[idx]["answer"]).split("#### ")[1].strip()
            if predicted_answer != -1 and predicted_answer == actual_answer:
                correct += 1
        except Exception as e:
            print("Error: ", e)
            no_answer += 1

        # Write the prompt, response, and the answer to a file
        with open("results_baseline3.txt", "a") as f:
            f.write(f"Predicted Answer: {predicted_answer}\nActual Answer: {actual_answer}\n")
            f.write("-" * 50 + "\n")

        count += 1

exit()

# Let's try to calculate the score
for i in tqdm(range(len(test_ds))):
    graded_solutions = [] # 

    prompt_start = generate_prompt(train_ds)
    prompt_end = format_example(test_ds, i, include_answer=False)
    prompt = prompt_start + prompt_end

    # TODO Batch the prompt for faster processing
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    outputs = model.generate(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        max_new_tokens=max_new_tokens,
                                        top_p=top_p,
                                        temperature=temperature)
    response = tokenizer.decode(outputs[0][input_ids.shape[1]:],
                                    skip_special_tokens=True)
    
    # Get the first answer block from the response
    answer_block = extract_first_answer(response)
    if answer_block == -1:
        no_answer += 1
    
    # Compare final answer with the actual answer (EM score)
    predicted_answer = answer_block.split("#### ")[1].strip() if answer_block != -1 else -1
    actual_answer = extract_first_answer(test_ds[i]["answer"]).split("#### ")[1].strip()
    if predicted_answer != -1 and predicted_answer == actual_answer:
        correct += 1
        
    # Write the prompt, response and the answer to a file
    with open("results_baseline1.txt", "a") as f:
        f.write(f"Predicted Answer: {predicted_answer}\nActual Answer: {actual_answer}\n")
        f.write("-" * 50 + "\n")
        
    count += 1