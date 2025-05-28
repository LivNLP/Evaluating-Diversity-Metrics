import json
import os
import spacy
import random
import argparse 
import openai
from typing import List 
from llm_util import openai_chat, upload_batch, separate_eval_data
from tqdm import tqdm

# Function to parse command line arguments
def get_args():
    parser = argparse.ArgumentParser() 
    
    # Add arguments for file paths and OpenAI configurations
    parser.add_argument("--model_name", type=str, default="qwen2")
    parser.add_argument("--candidate_path", type=str, default="bad_pairs_qwen2/shuffle.jsonl") 
    parser.add_argument("--ref_path", type=str, default="")
    parser.add_argument("--template_path", type=str, default="templates/commongen_eval.md")
    parser.add_argument("--dataset_name", type=str, default="commongen")
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=1497) 
    parser.add_argument("--generator", type=str)
    
    # OpenAI Configurations
    
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=300)
    
    args = parser.parse_args()
    
    # Generate output path and generator name dynamically
    args.generator = args.model_name +"_"+ os.path.basename(args.candidate_path)[:-6]
    args.output_path = "quality_eval/"+args.generator+ ".jsonl" 
    
    return args

# Initialize Spacy NLP model globally for reuse
nlp = None 

def analyze_words(pos_words, sentence):
    """
    Analyzes words in a sentence and returns matching lemmatized words from a given list.
    """
    global nlp 
    
    if nlp is None:
        nlp = spacy.load('en_core_web_lg') 
    
    doc = nlp(sentence)
    found_words = [tok.lemma_ for tok in doc if tok.lemma_ in pos_words]  # Lemmatization and POS matching
    
    return list(set(found_words))  # Return unique items

def load_quality_template(args, concept_list, candidate_A, candidate_B):
    """
    Load template and fill in placeholders for concepts and candidates.
    """
    with open(args.template_path, 'r', encoding='utf-8') as file:
        instruction_template = file.read()
        
    # Format the template with provided candidates and concept list
    instruction = instruction_template.format(
        concept_list=concept_list,
        candidate_A=candidate_A,
        candidate_B=candidate_B
    )
    
    return [{"role": "user", "content": instruction}]

def parse_result(result_str):
    if "neither" in result_str.lower():
        return "neither"
    elif "A" in result_str:
        return "A"
    elif "B" in result_str:
        return "B"
    elif "tie" in result_str:
        return "tie"
    else:
        return "Not Matched"

def batch_inputs(args, results):
    """
    Convert input prompts into API batch format for uploading to OpenAI.
    """
    tasks = [
        {
            "custom_id": f"task-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o",
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "messages": results[i]['prompt'],
            }
        } for i in range(len(results))
    ]
    return tasks


def upload_to_openai(args, results):
    """
    Upload batched tasks to OpenAI API.
    """
    batchs = batch_inputs(args, results)
    upload_path = args.generator +".jsonl"
    with open(upload_path, 'w', encoding='utf-8') as output_file:
        for batch in batchs:
            output_file.write(json.dumps(batch) + "\n")
    return upload_batch(upload_path)  # Return the uploaded batch ID


def retrieve_batches(args, openai_file):
    results = []
    with open(args.output_path, "r") as f:
        results = json.load(f)
    with open(openai_file, 'r', encoding='utf-8') as f:
        for item, line in zip(results, f):
            response = json.loads(line)["response"]["body"]["choices"][0]["message"]["content"]
            r = parse_result(response)
            result = parse_result(response)
            item["result"] = result
            item["winner"] = item["assignment"].get(result, result)
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)
            
    return results
        
def gpt_eval(results, args):
    """
    Evaluate candidates using OpenAI and update results with winners.
    """
    for ind, item in tqdm(enumerate(results), total=len(results)):
        chat_record = item["prompt"]
    
        # Call the OpenAI chat API
        response = openai_chat(
            model="gpt-4o",
            messages=chat_record,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )[0]
        
        # Parse the result and assign the winner
        result = parse_result(response)
        item["result"] = result
        item["winner"] = item["assignment"].get(result, result)
            
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)  
    
    return results
 

def place_generation(args):
    """
    Prepare data for evaluation by assigning candidates A and B randomly.
    """
    with open(args.ref_path, 'r', encoding='utf-8') as file:
        ref_data = [json.loads(line) for line in file.readlines()]
    with open(args.candidate_path, 'r', encoding='utf-8') as file:
        candidate_data = [json.loads(line) for line in file.readlines()]
    
    id_to_references = {x["src"]: x["sentences"] for x in ref_data}
    candidates = [c for c in candidate_data if c["src"] in id_to_references]
    references = [id_to_references[c["src"]] for c in candidates]
    
    # Adjust end index if necessary
    if args.end_idx < 0:
        args.end_idx = len(candidates)
    
    print(f"# examples in candidates: {len(candidates)}; We take {args.end_idx - args.start_idx} for evaluation.")
    
    results = []
    for item, human_annotations in zip(candidates[args.start_idx:args.end_idx], references[args.start_idx:args.end_idx]):
        d = {
            "human_ref": human_annotations,
            "src": item["src"],
            "model_output": item["sentences"],
            "generator": args.generator,
            "assignment": {},
            "result": "N/A"
        }
        
        # Randomly assign candidate A and B
        if random.random() < 0.5:
            A, B = random.choice(d["model_output"]), random.choice(d["human_ref"])
            d["assignment"] = {"A": d["generator"], "B": "human"}
        else:
            A, B = random.choice(d["human_ref"]), random.choice(d["model_output"])
            d["assignment"] = {"A": "human", "B": d["generator"]}
        
        d["prompt"] = load_quality_template(args, d["src"], A, B)
        results.append(d)
    
    return results


def evaluate_quality(args):
    """
    Evaluate quality metrics and print results.
    """
    row, cover_ratios, results = {}, [], []
    win_count, human_win_count, tie_count = 0, 0, 0
    lens = []
    
    # Load results
    with open(args.output_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    # Compute coverage and winner statistics
    for item in results:
        for sent in item['model_output']:
            lens.append(len(sent.split()))
            if args.dataset_name == "commongen":
                concept_set = item["src"].split()
                found_words = analyze_words(concept_set, sent)
                cover_ratios.append(len(found_words) == len(concept_set))
        
        # if item["winner"] == item["generator"]:
        #     win_count += 1    
        # elif item["winner"] == "human":
        #     human_win_count += 1
        # else:
        #     tie_count += 1
    
    row["len"] = f"{sum(lens) / len(lens):.2f}"
    row["win_tie"] = f"{(win_count + tie_count) / len(results) * 100:.2f}"
    
    if args.dataset_name == "commongen":
        row['cover'] = f"{sum(cover_ratios) / len(cover_ratios) * 100:.2f}"
        row["overall"] = (float(row["win_tie"]) * float(row["cover"])) / 100
    
    print(row)
            
def evaluate_accuracy(data_path, metric):
    total_allcase = 0
    correct_allcase = 0
    data = []

    with open(data_path, 'r', encoding='utf-8') as file:
        for line in file:
            obj = json.loads(line)
            set1_score = separate_eval_data([obj['set1']], metric)
            set2_score = separate_eval_data([obj['set2']], metric)
            traditional_diversity = 0 if set1_score >= set2_score else 1
            llm_diversity = obj['llm_diversity']
            
            if llm_diversity != 2:
                data.append((traditional_diversity, llm_diversity))

    for traditional_diversity, llm_diversity in data:
        total_allcase += 1
        if traditional_diversity == llm_diversity:
            correct_allcase += 1

    return correct_allcase / total_allcase, data

# Only those has equal quality but different diversity
def evaluate_diversity(data_path):

    bad_subset = []
    with open(data_path, 'r', encoding='utf-8') as file:
        for line in file:
            obj = json.loads(line)
            if obj['llm_diversity'] != 2:
                bad_subset.append(obj)

    evaluate_path = data_path[:-5] + "_evaluation.json"
    with open(evaluate_path, 'w', encoding='utf-8') as file:
        for obj in bad_subset:
            file.write(json.dumps(obj) + '\n')

    # filter the sentence sets with the top-4 highest diversity
    candidate_metrics = ["self_bleu_3", "self_bleu_4", "vendi_ngram_0.5", "vendi_ngram_1", "vendi_ngram_inf","distinct_4", "entropy_2",
         "chamfer","self_cos", "vendi_simcse_0.5", "vendi_simcse_1", "vendi_simcse_inf",]
    # candidate_metrics = [ "self_bleu_3", "vendi_ngram_1","distinct_4", "entropy_2", "self_cos","vendi_simcse_1"]
    for metric in candidate_metrics:
        accuracy, data = evaluate_accuracy(evaluate_path, metric)
        print(f"Accuracy for {metric}: {accuracy}")

      
        
def quality_evaluation(args):
    random.seed(42)
    results = place_generation(args)
    print(f"We have {len(results)} examples to evaluate!")
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)
    upload_to_openai(args, results)
    
    #results = gpt_eval(results, args)
    #evaluate_quality(args)
    
if __name__ == "__main__":
    # Evaluate diversity
    evaluate_diversity("data/qwen_evaluation.jsonl")
    
    # Evaluate quality
    # args = get_args()
    # print(args.output_path)
    # # quality_evaluation(args)
    # # results = retrieve_batches(args, "batch_output.jsonl")
    # evaluate_quality(args)