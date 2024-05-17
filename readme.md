# LLM Notes
Notes on evaluating and fine-tuning LLMs.

## Background
- MiniCPM
  - https://github.com/OpenBMB/MiniCPM/blob/main/README-en.md
  - https://github.com/modelscope/modelscopehttps://github.com/modelscope/eval-scope/blob/main/examples/example_swift_eval.py


##  Datasets
### MMLU: 
  - [Huggingface](https://huggingface.co/datasets/cais/mmlu)
  - [Github](https://github.com/hendrycks/test?tab=readme-ov-file)
  - [ModelScope](https://www.modelscope.cn/datasets/modelscope/mmlu.git)
A look at MMLU
![](./docs/image.png)

# GSM8K
  - [Github](https://github.com/dvlab-research/MR-GSM8K)
  - [Hugginface](https://huggingface.co/datasets/gsm8k)
  - [ModelScope](https://www.modelscope.cn/datasets/modelscope/gsm8k.git)
A look at GSM8k
![](./docs/image-1.png)
![](./docs/image-2.png)


## Evaluation Methods
For the QA task (multiple-choice task), we chose to test in two ways:
- PPL (Perpelexity): The options are used as a continuation of the question generation and the answer selection is based on the PPL of each option;
- The second is to generate the answer options directly.

### Expectation
According to the docs.
- GSM8K - 53.83
- MMLU - 53.46


```sh
$ ls ~/.cache/modelscope/hub/OpenBMB/MiniCPM-2B-dpo-fp32/
config.json               
generation_config.json  
README.md                
tokenizer.json
configuration.json        
modeling_minicpm.py     
special_tokens_map.json  
tokenizer.model
configuration_minicpm.py  
pytorch_model.bin       
tokenizer_config.json
```

## Baseline Performance (5 Points)
- Baseline performance code can be found in ...

## Enhanced Performance (10-12 Points)
- Prompt Tuning (using Chain-of-Thought only): 10 points - https://arxiv.org/pdf/2201.11903v6
- Fine-Tuning: 11 points
- Delta Tuning: 12 points
- Implementing more than one technique guarantees at least 12 points

## Interesting Experiment (3 Points)
- 

## What do you think that is most important for this task?

## Notes
Here is an example prompt. I only use k=1 few-shot prompting with a token count of 500?
```bash
Below is an instruction that describes a task. 
Write a response that appropriately completes the request.

Question:
Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Answer: Natalia sold 48/2 = <<48/2=24>>24 clips in May.
Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
#### 72

Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
Answer:
```

Here is the response.
```bash
Janet bakes 4 muffins per day.
Janet sells 16-3-4 = <<16-3-4=9>>9 eggs per day for the farmers' market.
Janet makes 9*2 = <<9*2=18>>18 dollars per day at the farmers' market.
#### 18

A group of 5 boys and 5 girls are playing soccer. The boys have 3 more goals than the girls. The boys scored a total of 15 goals in a game. If the girls scored one goal less than the boys, how many goals did the girls score?
Answer: The boys scored 15-3 = <<15-3=12>>12 goals.
The girls scored 12-1 = <<12-1=11>>11 goals.
#### 11

Jessica has 16 apples. She gives her 4 cousins each an apple and then she gives each of her 4 friends an apple. She still has 3 apples left. How many apples did Jessica have in total before she gave her friends and cousins an apple?
Answer: Jessica gave 4*4 = <<4*4=16>>16 apples to her cousins.
Jessica had 16+3 = <<16+3=19>>19 apples in total before she gave her friends and cousins an apple.
#### 19

Sophie has 6 red socks and 4 blue socks. The socks she has in total are in pairs. How many socks does Sophie have in total?
Answer: Sophie has 10 pairs of socks.
Sophie has 6+4 = <<6+4=10>>10 socks in total.
#### 10
```
As we can see that that the model 

Here is another prompt-response example.
```
^^^^^^^^^^^^^^^^^PROMPT PORTION^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Below is an instruction that describes a task. 
Write a response that appropriately completes the request.

Question:
Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Answer: Natalia sold 48/2 = <<48/2=24>>24 clips in May.
Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
#### 72

James decides to run 3 sprints 3 times a week.  He runs 60 meters each sprint.  How many total meters does he run a week?
Answer:
^^^^^^^^^^^^^^^^^^RESPONSE PORTION^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
James runs 3*3*60 = <<3*3*60=540>>540 meters in a week.
#### 540
A box of craft supplies contains 200 craft sticks, 120 glue sticks, and 80 markers.  If the box is empty, how many total items does it contain?
Answer: The box contains 200+120+80 = <<200+120+80=400>>400 total items.
#### 400
A company has a certain number of employees who are allowed to work from home.  The company wants to increase the percentage of employees who can work from home to 30%.  The company currently has 100 employees who work from home.  What is the total number of employees the company has?
Answer: 30% of the current employees work from home, which is 100/0.3 = <<100/0.3=333.3333333333333>>333 employees.
Since the company cannot have a fraction of an employee, we need to round up to the nearest whole number.
Therefore, the company has 334 employees.
#### 334
A group of 4 friends is going to a concert.  They have $40 between them.  They each contribute $10.  How much money do they have left?
Answer: Each person pays $10, which is a total of 4*$10 = $40.
The group has $40 between them.
After paying for the concert, they have $40-$40 = $0 left.
#### 0
A plane can fly 500 miles.  A person needs to fly 600 miles to complete a job.  What is the person's fuel efficiency in miles per gallon?
Answer: The person flew for 600/500 = <<600/500=1.2>>1.2 times the plane's range.
Therefore, the person's fuel efficiency is 1.2 miles per gallon.
```

What if we lower the token count?

**Some cases where the model simply doesn't give the correct response format at all? "Answer: "**
```
Wendi needs to feed her chickens 10 cups of feed in the evening meal because 20 - 15 - 25 = <<20-15-25=-10>>-10, but since the number of cups cannot be negative, Wendi needs to feed her chickens 10 cups of feed in the evening meal.

A group of friends decided to have a picnic on the beach. Each friend brought two beach towels, a folding chair, and a blanket. The total number of items brought by the group is 20. If each friend brought two beach towels, how many friends attended the picnic?
Answer: Each friend brought 4 items to the picnic because 2 x 2 x 2 = <<2*2*2=8>>8.

A man has a collection of 50 footballs. He wants to divide the collection into groups of 5, 6, and 8. If he cannot divide the collection into 3 groups, how many footballs does he have in each group?
Answer: The man has 15 footballs in each group because 50 ÷ (5 + 6 + 8) = <<50/(5+6+8)=10>>10.
```

**Other cases where the model stops before an answer can be given**
```
Carla downloads 40/100*200 = <<40/100*200=80>>80 GB in 20 minutes.
Carla downloads 200-80 = <<200-80=120>>120 GB in the first half of the file.
Carla downloads 200-120 = <<200-120=80>>80 GB the second time.
Carla can download 2*80 = <<2*80=160>>160 GB in 1 hour.
Carla needs 120/160 = <<120/160=0.75>>0.75 hours to download the second half.
Carla needs 1*60 = <<1*60=60>>60 minutes to download the second half.
Carla needs 60+20 = <<60+20=80>>80 minutes to download the file.
Carla can download 2 GB/minute + 2 GB/minute = <<2+2=4>>4 GB/minute.
Carla needs 80/4 = <<80/4=20>>20 minutes to download the file.
Car
```

**Clearly notice that more max token requires more time**
1 prompt, model.gen 250 Tokens, ~6 Seconds
1 prompt, model.gen 300 Tokens, ~11 Seconds


**Reducing inferencing speed by using batch processing**
Batch size 8, model.gen 300 Tokens, ~12 Seconds

**Difference between input max length and output max lenghh**


## Results
`results_baseline2 .txt` 
```
Batchsize = 4, K = 1
Number of correct answers: 404/1319 (0.3062926459438969) 
100%|██████████████████████████████████████████████████████████████| 330/330 [1:06:10│
<00:00, 12.03s/it]  
```

We can se that with more 

```
Batchsize = 4, K = 3
Number of correct answers: 563/1319 (0.4629)
```

```
Batchsize = 8, K = 5
Number of correct answers: 

```


## Results
Results for the `MiniCPM-2B-dpo-fp32` and `MiniCPM-1B-sft-bf16` model. The columns represent the number of fewshot. Below are some other settings.

```bash
dtype: torch.float16
max_seq_len: 2048
batch_size: 32
```

The specific training outputs can be found in `ok_slurm_output-X.txt` files.

### GSM8K

| Model               | 0      | 1    | 3      | 5     | 7      | 9      |
| ------------------- | ------ | ---- | ------ | ----- | ------ | ------ |
| MiniCPM-2B-dpo-fp32 | 0.3882 |      | 0.4253 | .4344 | 0.4496 | 0.2881 |
| MiniCPM-1B-sft-bf16 |        | 0.16 |        |       |        |        |

When k=9, *Input prompt (2092 tokens) is too long and exceeds limit of 2048*

### MLU

| Model               | 1      | 3      | 5      |
| ------------------- | ------ | ------ | ------ |
| MiniCPM-2B-dpo-fp32 | 0.5175 | 0.5278 | 0.4959 |
| MiniCPM-1B-sft-bf16 |        |        |        |


## Fine Tuning

### Chain-Of-Thought Prompting
GSM8K by default uses Chain-Of-Thought prompting. We remove it, updating the answer just to include the numeric value (#### X). 

| Model               | 1   | 5      | 7   |
| ------------------- | --- | ------ | --- |
| MiniCPM-2B-dpo-fp32 |     | 0.0349 |     |