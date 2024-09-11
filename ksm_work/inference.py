import os
from tqdm import tqdm

import pandas as pd 
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AwqConfig
from datasets import load_dataset

import argparse 
from transformers import set_seed
from datasets import Dataset, DatasetDict
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

import re

# Define your evaluation function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(logits, dim=-1)
    return {
        'accuracy': (predictions == labels).float().mean().item()
    }
    
def generate_dict(df):
    prompt_q = "Provide me an answer to the following question."
    prompt_ca = "You are given the following multiple choices and supposed to output the index of the correct answer."
    prompt_ex = "For instance, if the choices are ['a', 'b', 'c', 'd'] and the answer is 'b' the correct output should be 1."
    prompt_ex2= "For example, if the choices are ['a, a', 'a,b', 'b,a', 'b,b'] and the correct answer is 'b,a', the correct output should be 2."
    prompt_ex3="""When the sentence is long, it is divided using "", or ', and each part is labeled from 0 to 3. Provide the correct number that represents the correct part."""
    
    instruction_list = [[prompt_q + ' ' + prompt_ca + ' ' + prompt_ex + ' ' + prompt_ex2 + ' ' + prompt_ex3] for _ in range(len(df))]
    question_list = df['문제']
    choices_list = df['선택지']
    dataset_dict = {'instruction': instruction_list, 'question': question_list, 'choices': choices_list}
    dataset = Dataset.from_dict(dataset_dict)
    
    return dataset
    
def create_datasets(df, tokenizer, apply_chat_template=False):
    """ 
    Customized function for converting dataframes to huggingface datasets 
    """
    def preprocess(samples):
        batch = []
        PROMPT_DICT = {
            "prompt_input": (
                "Below is an instruction that describes a task, paired with an input that provides further context.\n"
                "아래는 작업을 설명하는 명령어와 추가적 맥락을 제공하는 입력이 짝을 이루는 예제입니다.\n\n"
                "Write a response that appropriately completes the request.\n요청을 적절히 완료하는 응답을 작성하세요.\n\n"
                "### Instruction(명령어):\n{instruction}\n\n### Input(입력):\n{input}\n\n### Response(응답):"
            ),
            "prompt_no_input": (
                "Below is an instruction that describes a task.\n"
                "아래는 작업을 설명하는 명령어입니다.\n\n"
                "Write a response that appropriately completes the request.\n명령어에 따른 요청을 적절히 완료하는 응답을 작성하세요.\n\n"
                "### Instruction(명령어):\n{instruction}\n\n### Response(응답):"
            ),
        }
        for instruction, question, choices in zip(samples["instruction"], samples["question"], samples["choices"]):
            user_input = question + '<|sep|>' + choices  
            conversation = PROMPT_DICT['prompt_input'].replace('{instruction}', instruction[0]).replace('{input}', user_input)
            batch.append(conversation)

        return {"content": batch}

    dataset = generate_dict(df)
    
    raw_datasets = DatasetDict()
    raw_datasets["test"] = dataset

    raw_datasets = raw_datasets.map(
        preprocess,
        batched=True,
        remove_columns=raw_datasets["test"].column_names,
    )

    test_data = raw_datasets["test"]
    print(
        f"Size of the test set: {len(test_data)}"
    )
    print(f"A sample of test dataset: {test_data[1]}")

    return test_data


if __name__ == "__main__":

    # set base directory 
    BASE_DIR = os.path.dirname(__file__)
    
    # Confirm which GPUs are visible
    print(f"Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', required=False, default=42, help='add seed number')
    parser.add_argument('--response_split', required=False, default='\nThe answer is', help='add response splitter')
    parser.add_argument('--model_path', required=False, default='', help='add pretrained model path')

    args = parser.parse_args()
   
    # set seed for reproducibility
    set_seed(args.seed)
    
    model_path = os.path.join(BASE_DIR, args.model_path)
    # model_path = args.model_path

    quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        torch_dtype=torch.float16,
        # quantization_config=quantization_config,
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    df = pd.read_csv(os.path.join(BASE_DIR, 'data/test_preprocessing_v2.csv'), encoding='utf-8')
    
    test_dataset = create_datasets(
        df,
        tokenizer,
        apply_chat_template=False
    )
        
    device = "cuda" if torch.cuda.is_available else "cpu"
    model = model.to(device)
    
    # inference 
    df_submission = pd.DataFrame()
    id_list, answer_list, response_list = list(), list(), list()
    
    for i, test_data in enumerate(tqdm(test_dataset)): 
        text = test_data['content']
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        
        # Remove 'token_type_ids' if present 
        model_inputs.pop('token_type_ids', None)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=9,
                eos_token_id=tokenizer.eos_token_id, 
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(response)
        
        # create submission.csv 
        if args.response_split in response:
            # response_split이 존재하는 경우, split으로 답변 부분 추출
            answer = response.split(args.response_split)[1].strip()
        else:
            # response_split이 존재하지 않는 경우, response 전체를 그대로 answer에 담음
            answer = response.strip()  # 필요시 strip()으로 공백 제거

        def extract_number_or_keep(text):
            match = re.search(r'\d+', text)  # 숫자를 검색
            if match:
                return match.group()  # 숫자가 있으면 숫자만 반환
            else:
                return text  # 숫자가 없으면 원본 텍스트 반환


        # extract_number_or_keep 함수를 적용
        answer = extract_number_or_keep(answer)
        
        id_list.append(i)
        answer_list.append(answer)
        response_list.append(response)

        
    df_submission['id'] = id_list
    df_submission['answer'] = answer_list
    df_submission['response'] = response_list
    df_submission.to_csv(os.path.join(BASE_DIR, 'submission_Synatra-7B-Instruct_single_prompt.csv'), index=False)
