import os
import sys
import warnings
import json
import pandas as pd
import numpy as np
from collections import defaultdict
sys.path.append(r'')
warnings.filterwarnings("ignore")
from tqdm import tqdm
from nlgeval.pycocoevalcap.bleu.bleu import Bleu
from nlgeval.pycocoevalcap.rouge.rouge import Rouge
import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
from bert_score import score
from vendi_score import text_utils
from sklearn.metrics.pairwise import cosine_distances


TOKENIZER = AutoTokenizer.from_pretrained("huggingface/roberta_large")
MODEL = AutoModel.from_pretrained("huggingface/roberta_large")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL = MODEL.to(device)  # Move the model to GPU

def address_predict(concept_file,predict_file):
    references = []
    concept_dict = defaultdict(list)
    previous_set = None
    with open(concept_file, 'r', encoding="utf-8") as fr1, open(predict_file, 'r', encoding="utf-8") as fr2:
        for concept, pred in zip(fr1.readlines(), fr2.readlines()):
            concepts = concept.strip().split()
            sorted_line = ' '.join(sorted(concepts))
            concept_dict[sorted_line].append(pred.strip())
    nested_list = [v for v in concept_dict.values()]

    return nested_list



def eval_self_bertscore(sentences_groups):
    hyp_list, ref_list = [], []
    for sentences in sentences_groups:
        for i in range(len(sentences)):
            hyp_list.append(sentences[i])
            ref_list.append(sentences[:i]+sentences[i+1:])
    
    _, _, F1 = score(hyp_list, ref_list, lang="en", rescale_with_baseline=True)
    bertscore = F1.mean().item()
    print("The F1 self-bertscore is ", bertscore)
    return bertscore

def eval_self_bleu(sentences_groups):
    hyp_list, ref_list = [], []
    for sentences in sentences_groups:
        for i in range(len(sentences)):
            hyp_list.append(sentences[i]) 
            ref_list.append('\t'.join(sentences[:i]+sentences[i+1:]))
    
    self_metrics = compute_metrics(hyp_list=hyp_list, ref_list=ref_list)
    self_metrics = {f'self_{k}': v for k, v in self_metrics.items()}
    self_mean_bleu = np.mean([self_metrics[f'self_bleu_{i}'] for i in range(1, 5)])
    self_metrics['self_bleu'] = self_mean_bleu

    return self_metrics

# Calculate the average cosine similarity between each sentence pair in a group
def calculate_cosine(texts, model=MODEL, tokenizer=TOKENIZER):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move input data to GPU
    
    # Get the embeddings
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output

    # Calculate cosine similarities and sum them up for each sentence
    embeddings_norm = embeddings / embeddings.norm(dim=1, keepdim=True)
    
    # Compute similarity matrix
    similarity_matrix = torch.matmul(embeddings_norm, embeddings_norm.T)
    
    # Exclude self-comparisons
    mask = torch.ones_like(similarity_matrix) - torch.eye(len(embeddings), device=embeddings.device)
    total_similarities = (similarity_matrix * mask).sum().item()
    num_comparisons = mask.sum().item()
    
    # Calculate average similarity
    average_similarity = total_similarities / num_comparisons
    return average_similarity


def eval_self_avgcosine(sentences_groups):
    scores = []
    for sentences in sentences_groups:
        temp = calculate_cosine(sentences, model=MODEL, tokenizer=TOKENIZER)
        scores.append(temp)
    return np.mean(scores)


def chamfer_distance(texts, model=MODEL, tokenizer=TOKENIZER):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move input data to GPU
    
    # Get the embeddings
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
    
    # Calculate cosine similarities and sum them up for each sentence
    embeddings_norm = embeddings / embeddings.norm(dim=1, keepdim=True)
    embeddings_np = embeddings_norm.cpu().numpy()
    cosine_dist_matrix = cosine_distances(embeddings_np)
    np.fill_diagonal(cosine_dist_matrix, np.inf)
    min_distances = np.min(cosine_dist_matrix, axis=1)
    chamfer_distance_value = np.mean(min_distances)
    return chamfer_distance_value         


def eval_chamfer_distance(sentences_groups):
    scores = []
    for sentences in sentences_groups:
        temp = chamfer_distance(sentences, model=MODEL, tokenizer=TOKENIZER)
        scores.append(temp)
    return np.mean(scores)

    
    
def eval_entropy_distinct(predictions):
    diversity_metrics = {}

    counter = [defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)]
    for pred in predictions:
        for gg in pred:
            g = gg.rstrip('2').split()
            for n in range(4):
                for idx in range(len(g)-n):
                    ngram = ' '.join(g[idx:idx+n+1])
                    counter[n][ngram] += 1
        
    for n in range(4):
        entropy_score = 0
        total = sum(counter[n].values()) + 1e-10
        for v in counter[n].values():
            entropy_score += - (v+0.0) /total * (np.log(v+0.0) - np.log(total))
        diversity_metrics[f'entropy_{n+1}'] = entropy_score
    diversity_metrics['entropy'] = sum([diversity_metrics[f'entropy_{n+1}'] for n in range(4)])/4
    
    
    for n in range(4):
        total = sum(counter[n].values()) + 1e-10
        diversity_metrics[f'distinct_{n+1}'] = (len(counter[n].values())+0.0) / total
    diversity_metrics['distinct'] = sum([diversity_metrics[f'distinct_{n+1}'] for n in range(4)]) / 4
    return diversity_metrics

    

def compute_metrics(hyp_list, ref_list, no_overlap=False):
    refs = {idx: lines.strip().split('\t') for (idx, lines) in enumerate(ref_list)}
    hyps = {idx: [lines.strip()] for (idx, lines) in enumerate(hyp_list)}
    assert len(refs) == len(hyps)

    ret_scores = {}
    if not no_overlap:
        scorers = [
            (Bleu(4), ["bleu_1", "bleu_2", "bleu_3", "bleu_4"]),
            (Rouge(), "rouge_l"),
        ]
        for scorer, method in scorers:
            score, scores = scorer.compute_score(refs, hyps)
            if isinstance(method, list):
                for sc, scs, m in zip(score, scores, method):
                    ret_scores[m] = sc
            else:
                ret_scores[method] = score
        del scorers
    return ret_scores
from collections import defaultdict


def sorted_sentence(sen_list):
    sorted_sentences = sorted(sen_list, key=lambda x: x['score'], reverse=True)
    sorted_sentences_list = [sentence['sentence'] for sentence in sorted_sentences]
    return sorted_sentences_list


def sort_predict(concept_file,predict_file):
    concept_dict = defaultdict(list)
    previous_set = None
    res = []
    temp_lst = []
    original_sentences = []
    with open(concept_file, 'r') as fr1, open(predict_file, 'r') as fr2:
        for concept, pred in zip(fr1.readlines(), fr2.readlines()):
            concepts = ' '.join(sorted(concept.strip().split()))
            
            original_sentences.append(pred.split('\tScore: ')[0].strip())
            if concepts != previous_set:
                if previous_set is not None:
                    sort_sens = sorted_sentence(temp_lst)
                    #print(sort_sens)
                    res.extend(sort_sens)
                temp_lst = []
                previous_set = concepts
            temp = {}
            temp['sentence'] = pred.split('\tScore: ')[0].strip()
            temp['score'] = float(pred.split('\tScore: ')[1].strip())       
            temp_lst.append(temp)
        if previous_set is not None:
            res.extend(sorted_sentence(temp_lst))
    with open(predict_file+".addressed",'w',encoding="utf-8") as fw:
        for line in res:
            fw.write(line+"\n")
    
    with open(predict_file+".original",'w',encoding="utf-8") as fw:
        for line in original_sentences:
            fw.write(line+"\n")


def vendi_score(texts, q, model=MODEL, tokenizer=TOKENIZER):
    ngram = []
    simcse = []
    
    for sentences in texts:
        ngram_temp = text_utils.ngram_vendi_score(sentences, ns=[1, 2, 3, 4], q=q)
        simcse_temp = text_utils.embedding_vendi_score(sentences, model=model, tokenizer=tokenizer, q=q)
        
        ngram.append(ngram_temp)
        simcse.append(simcse_temp)
    ngram_vs = np.mean(ngram)
    simcse_vs = np.mean(simcse)
    vendi = {f"vendi_ngram": ngram_vs, f"vendi_simcse": simcse_vs}
    return vendi

def intdiv_score(texts, q=1,  model=MODEL, tokenizer=TOKENIZER):
    ngram = []
    simcse = []
    for sentences in texts:
        ngram_temp = text_utils.ngram_intdiv_score(sentences, ns=[1, 2, 3, 4], q=q)
        #simcse_temp = text_utils.embedding_intdiv_score(sentences, model=model, tokenizer=tokenizer,q=q)
        ngram.append(ngram_temp)
        #simcse.append(simcse_temp)
    ngram_vs = np.mean(ngram)
    #simcse_vs = np.mean(simcse)
    intdiv = {"intdiv": ngram_vs}

    return intdiv

def round_dict_values(d, decimal_places=3):
    return {k: round(v, decimal_places) for k, v in d.items()}

def evaluate_and_save_results(data_path, start_loc, window_size, methods, output_excel_path):
    results = []
    for i in range(start_loc, 5):
        for j in range(window_size, 8):
            for method in methods:
                file_path = f"{data_path}/spatial_{i}_{j}_{method}.json"
                print()
                print(f"Evaluating {file_path}")
                
                sentences = []
                with open(file_path, "r", encoding='utf-8') as file:
                    for line in file:
                        obj = json.loads(line)
                        sentences.append(obj['sentences'])
                
                self_bleu = eval_self_bleu(sentences)
                diversity_metrics = eval_entropy_distinct(sentences)
                vendi = vendi_score(sentences)
                
                # Round the values to 3 decimal places
                self_bleu = round_dict_values(self_bleu)
                diversity_metrics = round_dict_values(diversity_metrics)
                vendi = round_dict_values(vendi)
                
                result = {
                    "file": file_path,
                    "self_bleu": self_bleu,
                    "diversity_metrics": diversity_metrics,
                    "vendi": vendi
                }
                results.append(result)
    
    # Save results to Excel
    rows = []
    for result in results:
        row = {
            "file": result["file"],
            **result["self_bleu"],
            **result["diversity_metrics"],
            **result["vendi"]
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_excel(output_excel_path, index=False)


def get_sentence_embedding(texts):
    inputs = TOKENIZER(texts, padding=True, truncation=True, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move input data to GPU
    
    # Get the embeddings
    with torch.no_grad():
        embeddings = MODEL(**inputs, output_hidden_states=True, return_dict=True).pooler_output
    return embeddings

def top_k_std(sentences_groups, k=4):
    group_stds = []
    for group in sentences_groups:
        embeddings = get_sentence_embedding(group)
        std_per_dimension = torch.std(embeddings, dim=0)
        overall_std = std_per_dimension.mean().item()
        group_stds.append((group, overall_std))
    sorted_groups = sorted(group_stds, key=lambda x: x[1], reverse=True)[:k]
    
    return sorted_groups
    


if __name__ == "__main__":
    sentences_groups = [[
        "The quick brown fox jumps over the lazy cat.",
        "The quick brown fox jumps over the lazy dog.",
        "The quick brown fox jumps over the lazy mouse.",
        "The quick brown fox jumps over the lazy girl."
    ]]
    print(eval_chamfer_distance(sentences_groups))
    
    sentences_groups = [["The goat jumped over the dog to feed it with a bottle.", "To go on vacation, the dog fed the goat by jumping into a bottle.", "A hungry goat said that it will jump into a bottle and feed itself like a dog.", "The dog took a deep breath and jumped inside the feeding bottle while the goat watched."]]
    print(eval_chamfer_distance(sentences_groups))