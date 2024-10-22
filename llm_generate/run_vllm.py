import re
import vllm
from vllm.lora.request import LoRARequest
import pandas as pd
import argparse
import os
parser = argparse.ArgumentParser(description='gamma inference')
parser.add_argument('--model_dir', type=str, default = "/home/chenning/opensource_models/llama/Meta-Llama-3.1-8B-Instruct")
parser.add_argument('--lora_dir', type=str, default = '/kaggle/input/0920-llama-expert-frog-30/checkpoint-1150/checkpoint-1150')
parser.add_argument('--input_file', type=str, default = './tmp/submission.parquet')
parser.add_argument('--output_file', type=str, default = './tmp/submission.parquet')
parser.add_argument('--batch_size', type=int, default = 4)
parser.add_argument('--gpu', type=str, default = 0)
parser.add_argument('--max_length', type=int, default = 2200)
parser.add_argument('--max_tokens', type=int, default = 1024)

parser.add_argument('--seed', type=int, default = 777)
parser.add_argument('--num_responses', type=int, default=1)
parser.add_argument('--spread_max_length',action='store_true', default=False)
parser.add_argument('--temperature',type=float, default = 0)
parser.add_argument('--top_p',type=float, default = 0.9)
parser.add_argument('--tensor_parallel_size', type=int, default = 1)

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
# os.environ["CUDA_VISIBLE_DEVICES"]='2,3'

df = pd.read_parquet(args.input_file)
sql_lora_path = args.lora_dir
llm_path = args.model_dir
print(f'lora path: {sql_lora_path}')
print(f'llm path: {llm_path}')
print(args)
llm = vllm.LLM(
    llm_path,
    tensor_parallel_size=args.tensor_parallel_size, 
    # gpu_memory_utilization=0.95, 
    trust_remote_code=True,
    dtype="half", 
    enforce_eager=False,
    max_model_len=args.max_length,
    disable_log_stats=True,
    enable_lora=True,
    max_lora_rank=64
)
tokenizer = llm.get_tokenizer()


responses = llm.generate(
    df["prompt_response"].values,
    vllm.SamplingParams(
        n=args.num_responses,  # Number of output sequences to return for each prompt.
        top_p=args.top_p,  # Float that controls the cumulative probability of the top tokens to consider.
        temperature=args.temperature,  # randomness of the sampling
        seed=args.seed, # Seed for reprodicibility
        skip_special_tokens=False,  # Whether to skip special tokens in the output.
        max_tokens=args.max_tokens,  # Maximum number of tokens to generate per output sequence.
    ),
    lora_request=LoRARequest("sql_adapter", 1, sql_lora_path),
    use_tqdm = True
)

if args.num_responses != 1:
    responses = [
        [response.outputs[i].text for i in range(args.num_responses)]
            for response in responses
    ]
else:
    responses = [x.outputs[0].text for x in responses]
df["text"] = responses

df.to_parquet(args.output_file, index=False)