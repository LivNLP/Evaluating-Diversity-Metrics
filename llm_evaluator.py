import os
import glob
import json
import random
import argparse
from collections import defaultdict
from itertools import combinations
from llm_util import openai_chat, upload_batch

# Default scoring maps
QUALITY = {"original": 1, "para_a": 1, "para_b": 1, "para_c": 1,
            "shuffle": 2, "shuffle_noun": 2, "nonsensical": 2}
DIVERSITY = {"original": 1, "para_a": 2, "para_b": 3, "para_c": 4,
             "shuffle": 2, "shuffle_noun": 7, "nonsensical": 6}
CHOICE_MAP = {"(A)": 0, "(B)": 1, "A": 0, "B": 1, "(C)": 2, "C": 2}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate and upload semeval-style comparisons via OpenAI GPT4 evaluator"
    )
    # Paths and I/O
    parser.add_argument("--data_dir", default="semeval/default",
                        help="Directory containing .jsonl source files")
    parser.add_argument("--template", default="templates/template.txt",
                        help="Path to the system prompt template")
    parser.add_argument("--src_out", default="output/source.jsonl",
                        help="Output path for sampled source entries")
    parser.add_argument("--batch_out", default="output/batch.jsonl",
                        help="Output path for OpenAI batch requests")
    parser.add_argument("--result_src", default="output/source.jsonl",
                        help="Input source file for retrieving results")
    parser.add_argument("--result_batch", default="output/batch_results.jsonl",
                        help="OpenAI batch result file")
    parser.add_argument("--result_out", default="output/all_results.jsonl",
                        help="Path to write merged results")
    parser.add_argument("--pref_in", default="output/all_results.jsonl",
                        help="Source file for preference aggregation")
    parser.add_argument("--pref_out", default="output/preference.jsonl",
                        help="Output path for final preference data")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of same-source samples to draw")
    # Set definitions
    parser.add_argument("--good_sets", default="original,para_a,para_b,para_c",
                        help="Comma-separated names of good sets")
    parser.add_argument("--bad_sets", default="shuffle_nouns,nonsensical,shuffle",
                        help="Comma-separated names of bad sets")
    # Single-run test with defaults
    parser.add_argument("--test_src", type=str,
                        default="Walk Dog Take Park Couple",
                        help="Source string for single generation test")
    parser.add_argument("--test_set1", type=json.loads,
                        default=json.dumps([
                            "The couple decided to take a walk in the park without taking their dog.",
                            "The couple takes their dog for a walk in the park.",
                            "Every evening, the couple takes a walk in the park with their dog.",
                            "The dog enjoys when the couple takes it for a walk in the park."
                        ], ensure_ascii=False),
                        help="JSON-formatted list of strings for set1")
    parser.add_argument("--test_set2", type=json.loads,
                        default=json.dumps([
                            "A couple take their dog for a walk in the park every morning.",
                            "Every morning, the couple and their dog take a moment and walk in the park.",
                            "Every evening, the couple takes a walk in the park with their dog.",
                            "In the park, a walk is taken every evening by the couple with their dog."
                        ], ensure_ascii=False),
                        help="JSON-formatted list of strings for set2")
    return parser.parse_args()


def single_generation(chat_record, temperature=0.0):
    openai_args = {
        "model": "gpt-4o",
        "messages": chat_record,
        "temperature": temperature,
        "max_tokens": 600,
        "n": 5,
        "response_format": {"type": "json_object"},
    }
    response = openai_chat(**openai_args)
    return average_vote(response)


def average_vote(answers):
    parsed = [json.loads(a) for a in answers]
    keys = [
        "Quality_Score_Set1", "Diversity_Score_Set1",
        "Quality_Score_Set2", "Diversity_Score_Set2"
    ]
    return {k: sum(d[k] for d in parsed) / len(parsed) for k in keys}


def load_data(data_dir):
    data = defaultdict(lambda: defaultdict(list))
    for filepath in glob.glob(os.path.join(data_dir, '*.jsonl')):
        with open(filepath, encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                src = entry.get('src')
                data[src][filepath].extend(entry.get('sentences', []))
    results = []
    for src, files in data.items():
        rec = {'src': src}
        for fp, sents in files.items():
            key = os.path.splitext(os.path.basename(fp))[0]
            rec[key] = sents
        results.append(rec)
    return results


def load_template(sample, template_path):
    with open(template_path, encoding='utf-8') as f:
        system_prompt = f.read()
    test_case = {
        'Source': sample['src'],
        'Set 1': sample['set1'],
        'Set 2': sample['set2']
    }
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(test_case, ensure_ascii=False)}
    ]


def same_source(data, n, good_sets, bad_sets):
    pairs = []
    for entry in data:
        src = entry['src']
        # pairs of bad sets
        for a, b in combinations(bad_sets, 2):
            pairs.append({
                'src': src,
                'set1': entry.get(a, []),
                'set2': entry.get(b, []),
                'set1_label': a,
                'set2_label': b
            })
    print(f"Generated {len(pairs)} candidate pairs")
    return random.sample(pairs, min(n, len(pairs)))


def batch_inputs(data_list, template_path):
    tasks = []
    for i, sample in enumerate(data_list):
        record = load_template(sample, template_path)
        tasks.append({
            'custom_id': f'task-{i}',
            'method': 'POST',
            'url': '/v1/chat/completions',
            'body': {
                'model': 'gpt-4o',
                'temperature': 1,
                'max_tokens': 600,
                'n': 5,
                'response_format': {'type': 'json_object'},
                'messages': record
            }
        })
    return tasks


def upload_to_openai(data_list, src_out, batch_out, template_path):
    os.makedirs(os.path.dirname(src_out), exist_ok=True)
    os.makedirs(os.path.dirname(batch_out), exist_ok=True)
    tasks = batch_inputs(data_list, template_path)
    with open(batch_out, 'w', encoding='utf-8') as bf:
        for t in tasks:
            bf.write(json.dumps(t, ensure_ascii=False) + '\n')
    with open(src_out, 'w', encoding='utf-8') as sf:
        for entry in data_list:
            sf.write(json.dumps(entry, ensure_ascii=False) + '\n')


def retrieve_result(src_in, batch_in, output_path):
    combined = []
    with open(src_in, encoding='utf-8') as sf, open(batch_in, encoding='utf-8') as bf:
        for src_line, result_line in zip(sf, bf):
            obj = json.loads(src_line)
            choices = json.loads(result_line)['response']['body']['choices']
            answers = [c['message']['content'] for c in choices]
            scores = average_vote(answers)
            obj.update({
                'Quality_Set1': scores['Quality_Score_Set1'],
                'Quality_Set2': scores['Quality_Score_Set2'],
                'Diversity_Set1': scores['Diversity_Score_Set1'],
                'Diversity_Set2': scores['Diversity_Score_Set2'],
                'llm_quality': 0 if scores['Quality_Score_Set1'] > scores['Quality_Score_Set2'] else
                               1 if scores['Quality_Score_Set1'] < scores['Quality_Score_Set2'] else 2,
                'llm_diversity': 0 if scores['Diversity_Score_Set1'] > scores['Diversity_Score_Set2'] else
                                1 if scores['Diversity_Score_Set1'] < scores['Diversity_Score_Set2'] else 2
            })
            combined.append(obj)
    with open(output_path, 'w', encoding='utf-8') as outf:
        for rec in combined:
            outf.write(json.dumps(rec, ensure_ascii=False) + '\n')


def main():
    args = parse_args()

    # Single-run test
    if args.test_src and args.test_set1 and args.test_set2:
        sample = {'src': args.test_src, 'set1': args.test_set1, 'set2': args.test_set2}
        record = load_template(sample, args.template)
        scores = single_generation(record)
        print(json.dumps(scores, ensure_ascii=False, indent=2))
        return

    good_sets = args.good_sets.split(',')
    bad_sets = args.bad_sets.split(',')

    data = load_data(args.data_dir)
    sampled = same_source(data, args.num_samples, good_sets, bad_sets)

    upload_to_openai(sampled, args.src_out, args.batch_out, args.template)
    print(f"Batches written to {args.batch_out}, sources to {args.src_out}")

if __name__ == '__main__':
    main()
