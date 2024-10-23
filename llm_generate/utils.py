import pandas as pd, numpy as np
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
def process(input_str):
    return json.loads(input_str)


def load_json(data):
    data.loc[:, 'prompt'] = data['prompt'].apply(process)
    data.loc[:, 'response_a'] = data['response_a'].apply(process)
    data.loc[:, 'response_b'] = data['response_b'].apply(process)
    return data

def prompt_1(data):
    '''
    #ConstructName (Most granular level of knowledge related to question) : \n{}
    #QuestionText : \n {}
    #CorrectAnswer: \n {}
    #InCorrectAnswer: \n {}
    '''
    data['prompt_response'] = "#Prompt\n#ConstructName (Most granular level of knowledge related to question):  \n" + data['ConstructName'] +\
                              "\n#QuestionText: \n" + data['QuestionText'] + "\n#CorrectAnswer: \n" + data['CorrectAnswerText'] +\
                              "\n#InCorrectAnswer: \n" + data['InCorrectAnswerText']
    return data

def prompt_2(data):

    def apply_template(row):
        PROMPT  = """Question: {Question}
            Incorrect Answer: {IncorrectAnswer}
            Correct Answer: {CorrectAnswer}
            Construct Name: {ConstructName}
            Subject Name: {SubjectName}

            Your task: Identify the misconception behind Incorrect Answer."""
        return PROMPT.format(
                 ConstructName=row["ConstructName"],
                 SubjectName=row["SubjectName"],
                 Question=row["QuestionText"],
                 IncorrectAnswer=row['InCorrectAnswerText'],
                 CorrectAnswer=row['CorrectAnswerText'])
    

    data['prompt_response'] = data.apply(apply_template, axis=1)
    return data


def prompt_3(data):

    def apply_template(row):
        PROMPT  = """#Construct Name: {ConstructName}\n#Question: {Question}\n#Correct Answer: {CorrectAnswer}\n#Incorrect Answer: {IncorrectAnswer}"""
        return PROMPT.format(
                 ConstructName=row["ConstructName"],
                 Question=row["QuestionText"],
                 CorrectAnswer=row['CorrectAnswerText'],
                 IncorrectAnswer=row['InCorrectAnswerText'])

    data['prompt_response'] = data.apply(apply_template, axis=1)
    return data


def get_text_length(text):
    '''
    不用空格分隔的文本, text length = len
    不用空格分隔的一般tokenizer后长度类似，所以还可以缩小
    空格分隔的，len(text.split(" "))
    '''
    length1 = len(text)
    length2 = len(text.split(" "))
    #远超过
    if length1 >= length2 * 30 and length1>= 300:
        return length1 * 0.75
    return length2




def make_prompt(data, if_train, prompt_type, max_length):
    #seperate prompt-response
    # data = data.explode(['prompt','response_a','response_b']).reset_index(drop = True)

    #prepare label
    
    data = data.fillna('None')   
    if prompt_type == 1:
        data = prompt_1(data)
    elif prompt_type == 2:
        data = prompt_2(data)
    elif prompt_type == 3:
        data = prompt_3(data)
    return data
def load_split_data(data_path, prompt_type, max_length, if_train, split, split_by_prompt, if_drop_duplicate, keep):
    """
    prompt_type: [1, 2, 3]
    if_train: True or False
    """
    if "csv" in data_path:
        data = pd.read_csv(data_path)
        #safely load data
        data = load_json(data)
    elif "json" in data_path:
        data = pd.read_json(data_path)

    # 删除MisconceptionId=-1的数据（即无label的）
    data = data[data['MisconceptionId'] != -1].reset_index(drop=True)
        
    # if split:
    #     #split train and valid
    #     idx = data.id.unique()
    #     valid_idx = [idx[i] for i in range(len(idx)) if i % 20 == 0]
        
    #     valid = data.loc[data.id.isin(valid_idx),].reset_index(drop = True)
    #     train = data.loc[~data.id.isin(valid_idx),].reset_index(drop = True)
            
    #     train = make_prompt(train, if_train, prompt_type, max_length)
    #     valid = make_prompt(valid, if_train, prompt_type, max_length)
    #     if if_drop_duplicate:
    #         train = train.drop_duplicates(subset = ['id'], keep =keep).reset_index(drop = True)
    #         valid = train.drop_duplicates(subset = ['id'], keep =keep).reset_index(drop = True)
    #     return train, valid
    
        
    data = make_prompt(data, if_train, prompt_type, max_length)
    # if if_drop_duplicate:
    #         data = data.drop_duplicates(subset = ['id'], keep =keep).reset_index(drop = True)
    return data, None

    
    