## LLM generate

### stage1


```
sh scripts/train_llama_instruct.sh
```

### 构造第二阶段训练集

```
create_stage2_data.ipynp
```

### stage2

```
sh scripts/train_llama_instruct_stage2.sh
```
