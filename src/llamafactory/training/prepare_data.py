
import json
from datasets import Dataset

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>. "
    "In the answer, only output the calculated number and yes or no, without including the process or any other explanations."
)

system_prompt = SYSTEM_PROMPT

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def make_conversation(example):
        #print(example)
    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["question"]},
        ],
        'solution': example["answer"],
    }

def get_finqa(dataset_path):
    with open(dataset_path, "r", encoding="utf8") as f:
        infs = json.load(f)
    
    data = Dataset.from_list(infs)
        
    data = data.map(make_conversation)

    return data



