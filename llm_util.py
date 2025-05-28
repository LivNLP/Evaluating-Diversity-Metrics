import re
import json
import ollama
import numpy as np
from typing import List
from openai import OpenAI
from evals.eval_diversity import eval_self_bleu, eval_entropy_distinct, vendi_score, eval_self_avgcosine, intdiv_score, eval_chamfer_distance
client = OpenAI(api_key = "")




def remove_numeric_prefixes(text):
    # Regular expression to match numeric prefixes followed by a period
    pattern = r'\b\d+\.\s*'
    # Replace found patterns with an empty string
    cleaned_text = re.sub(pattern, '', text.strip("\n "))
    
    return cleaned_text

def read_data(data_path):
    data = []
    sentences = []
    inputs = []
    with open(data_path, 'r', encoding='utf-8') as file:
        for line in file:
            obj = json.loads(line)
            data.append(obj)
    return data

def evaluate_diversity(fake_data, q=1):
    sentences = []
    for obj in fake_data:
        sentences.append(obj['sentences'])
    self_bleu = eval_self_bleu(sentences)
    print("Self BLEU: ", self_bleu)
    self_cosSim = eval_self_avgcosine(sentences)
    print("Self Cosine Similarity: ", self_cosSim)
    chamfer = eval_chamfer_distance(sentences)
    print("Chamfer Distance: ", chamfer)
    diversity_metrics = eval_entropy_distinct(sentences)
    print("Entropy & Distinct: ", diversity_metrics)
    
    for i in [0.5,1,"inf"]:
        vendi = vendi_score(sentences, q=i)
        print(f"Vendi{i}: ", vendi)
    intdiv = intdiv_score(sentences, q=q)
    print("IntDiv: ", intdiv)
    return self_bleu


def vendi_all(fake_data):
    sentences = []
    for obj in fake_data:
        sentences.append(obj['sentences'])
    for i in [0.1, 0.5, 1, 2, "inf"]:
        vendi = vendi_score(sentences, q=i)
        print(f"Vendi: ", vendi)


def separate_eval_data(sentences, metric):
    score = 0
    # metric: self_bleu, self_rough, entropy, distinct, vendi_ngram, vendi_simcse
    # self metric need to be subtracted from 1
    if  "self_bleu" in metric or "self_rough" in metric:
        score = 1 - eval_self_bleu(sentences)[metric]
    elif "self_cos" in metric:
        score = 1 - eval_self_avgcosine(sentences)
    elif "chamfer" in metric:
        score = eval_chamfer_distance(sentences)
    elif "distinct" in metric or "entropy" in metric:
        score = eval_entropy_distinct(sentences)[metric]
    elif "vendi" in metric:
        q_type = metric.split("_")[-1]
        metric_name = metric[:-len(q_type)-1]
        q = float(q_type) if q_type != "inf" else q_type
        score = vendi_score(sentences, q=q)[metric_name]
    elif "intdiv" in metric:
        score = intdiv_score(sentences)[metric]  
    return score        


def cronbach_alpha(scores):
    scores = np.array(scores)
    J = scores.shape[1]
    var_scores = np.var(scores, axis=1, ddof=1) 
    total_var = np.var(np.mean(scores, axis=1), ddof=1)  
    alpha = (J / (J - 1)) * (1 - np.sum(var_scores) / total_var)
    
    return alpha

# We have already know the overall scores for each dataset right? So we just read them   
def consistency_evaluation(data_path, metric):
    scores = []
    data = read_data(data_path)[:100]
    # Overall score
    sentences = []
    for obj in data:
        sentences.append(obj['sentences'])

    separate_scores = [separate_eval_data([sentence], metric) for sentence in sentences]
    alpha = cronbach_alpha(separate_scores)
    print("Cronbach's α: ", alpha)
    return alpha
    
    
    


def upload_batch(file_path):
    batch_file = client.files.create(file=open(file_path, "rb"),purpose="batch")
    batch_job = client.batches.create(input_file_id=batch_file.id,endpoint="/v1/chat/completions",completion_window="24h")
    print("Batch job created with id: ", batch_job.id)
    return batch_job.id

def retrieve_batch(batch_id, openai_file_path):
    batch_job = client.batches.retrieve(batch_id)
    result_file_id = batch_job.output_file_id
    result = client.files.content(result_file_id).content

    with open(openai_file_path, 'wb') as file:
        file.write(result)
    return result

def openai_chat(
    model: str=None,
    temperature: float=1.0,
    max_tokens: int=512,
    top_p: float=1.0,
    frequency_penalty: float=0,
    presence_penalty: float=0,
    prompt: str=None,
    n: int=1,
    messages: List[dict]=None,
    stop: List[str]=None,
    **kwargs,
) -> List[str]:
    assert prompt is not None or messages is not None, "Either prompt or messages should be provided."
    if messages is None:
        messages = [{"role":"system","content":"You are an AI assistant that helps people find information."},
                {"role":"user","content": prompt}]
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        n=n,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop,
        **kwargs,
    )
    response = [c.message.content.strip() for c in completion.choices]
    
    return response
