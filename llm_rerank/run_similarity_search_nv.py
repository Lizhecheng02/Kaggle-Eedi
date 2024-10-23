import os
os.environ["CUDA_VISIBLE_DEVICES"]="7"


from transformers import AutoTokenizer, AutoModel,AutoConfig
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import argparse
import torch.nn.functional as F

from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

parser = argparse.ArgumentParser(description='gamma inference')
parser.add_argument('--model_dir', type=str, default = "/home/chenning/opensource_models/nvidia/NV-Embed-v2")
parser.add_argument('--lora_path', type=str, default = "../recall_model_save/nv_round1_qlora_rerun/epoch_19_model/adapter.bin")
parser.add_argument('--input_file', type=str, default = 'submission.parquet')
parser.add_argument('--output_file', type=str, default = 'submission.parquet')
parser.add_argument('--batch_size', type=int, default = 2)
parser.add_argument('--gpu', type=int, default = 0)
parser.add_argument('--max_length', type=int, default = 512)
parser.add_argument('--seed', type=int, default = 777)
parser.add_argument('--recall_num', type=int, default = 25)

parser.add_argument('--num_beam', type=int, default=1)
parser.add_argument('--spread_max_length',action='store_true', default=False)
parser.add_argument('--temp',type=float, default = 1.03)
parser.add_argument('--use_lora',action='store_true', default=False)
# recall_num
args = parser.parse_args()
# 

result_df = pd.read_parquet(args.input_file)
m_map = pd.read_csv("../datas/misconception_mapping.csv")



per_gpu_batch_size = args.batch_size
# Sentences we want sentence embeddings for

model_path = args.model_dir
# device = "cuda:0"
print(model_path)
print(args)
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
config.text_config._name_or_path=model_path

# load model with tokenizer
model = AutoModel.from_pretrained(model_path, config=config, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")

# lora_dir = '../recall_model_save/nv_round1_qlora_rerun/epoch_0_model'
# model = PeftModel.from_pretrained(model, lora_dir)
if args.use_lora:
    lora_path = args.lora_path
    config = LoraConfig(
                r=64,
                lora_alpha=128,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                bias="none",
                lora_dropout=0.05,  # Conventional
                task_type="CAUSAL_LM",
            )
    # config = LoraConfig(
    #             r=32,
    #             lora_alpha=64,
    #             target_modules=[
    #                 "q_proj",
    #                 "k_proj",
    #                 "v_proj",
    #                 "o_proj",
    #                 # "gate_proj",
    #                 # "up_proj",
    #                 # "down_proj",
    #             ],
    #             bias="none",
    #             lora_dropout=0.05,  # Conventional
    #             task_type="CAUSAL_LM",
    #         )
    model = get_peft_model(model, config)
    d = torch.load(lora_path)
    model.load_state_dict(d, strict=False)
    model = model.merge_and_unload()
model.eval()
max_length = 32768

task_name_to_instruct = {"example": "Your task is to retrieve the possible misconception that might lead students to choose the incorrect answer in Mathematics multiple-choice questions.",}

query_prefix = "Instruct: " + task_name_to_instruct["example"]+"\n Query: "
# query_prefix = ""
passage_prefix = query_prefix


# task_name_to_instruct = {"example": "Given a question, retrieve passages that answer the question",}
# query_prefix = "Instruct: " + task_name_to_instruct["example"]+"\n Query: "
# passage_prefix = ""



def gen_embeds(model, texts, prefix, batch_size=2):
    
    all_ctx_vector = []
    for mini_batch in tqdm(range(0, len(texts[:]), batch_size)):
        mini_context = texts[mini_batch:mini_batch + batch_size]
        embeddings = model.encode(mini_context, instruction=prefix, max_length=max_length)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        all_ctx_vector.append(embeddings.detach().cpu().numpy())
    all_ctx_vector = np.concatenate(all_ctx_vector, axis=0)
    return all_ctx_vector

def gen_embeds2(model, texts, prefix, batch_size=2):
    
    all_ctx_vector = []
    for idx in tqdm(range(len(texts))):
        batch_embeddings = []
        for mini_batch in range(0, len(texts[idx]), batch_size):
            mini_context = texts[idx][mini_batch:mini_batch + batch_size]
            embeddings = model.encode(mini_context, instruction=prefix, max_length=max_length)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            # 将当前文本列表的嵌入添加到批次嵌入列表中
            batch_embeddings.append(embeddings.detach().cpu().numpy())
        batch_embeddings = np.concatenate(batch_embeddings, axis=0)
        all_ctx_vector.append(batch_embeddings)
#     all_ctx_vector = np.concatenate(all_ctx_vector, axis=0)
    return np.array(all_ctx_vector)

if args.num_beam != 1:
    preds_vectors = gen_embeds2(model, result_df["text"].to_list(), query_prefix, batch_size=per_gpu_batch_size)
else:
    preds_vectors = gen_embeds(model, result_df["text"].to_list(), query_prefix, batch_size=per_gpu_batch_size)
print(preds_vectors.shape)
labels_vectors = gen_embeds(model, m_map["MisconceptionName"].to_list(), passage_prefix, batch_size=per_gpu_batch_size)

print(preds_vectors.shape)
print(labels_vectors.shape)


from sklearn.metrics.pairwise import cosine_similarity

if args.num_beam != 1:
    pred_labels = []
    for idx in range(len(preds_vectors)):
        preds_cos_sim_arr = cosine_similarity(preds_vectors[idx], labels_vectors)
        preds_cos_sim_arr = np.sum(preds_cos_sim_arr, axis=0).reshape(1,-1)
        preds_sorted_indices = np.argsort(-preds_cos_sim_arr, axis=1)
        pred_labels.append(preds_sorted_indices[:, :args.recall_num].reshape(1,-1).tolist())
    
else:
    preds_cos_sim_arr = cosine_similarity(preds_vectors, labels_vectors)
    preds_sorted_indices = np.argsort(-preds_cos_sim_arr, axis=1)

    pred_labels = preds_sorted_indices[:, :args.recall_num].tolist()

result_df["MisconceptionId"] = pred_labels
result_df["MisconceptionId"] = result_df["MisconceptionId"].apply(lambda x: ' '.join(map(str, x)))
result_df = result_df.rename(columns={"id":"QuestionId_Answer"})

submission = result_df[["QuestionId_Answer", "MisconceptionId"]]

# submission = submission.sort_values("QuestionId_Answer", ascending=True).reset_index(drop=True)
submission.head()
submission.to_csv(args.output_file, index=False)