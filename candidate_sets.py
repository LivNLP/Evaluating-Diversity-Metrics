
import sys
sys.path.append(r'')
import json
import random
import pandas as pd
from llm_util import read_data, evaluate_diversity, vendi_all
from evals.eval_diversity import eval_self_bleu, eval_entropy_distinct, vendi_score
from aligner import Aligner
from itertools import combinations, permutations
import nltk
import random
from nltk import pos_tag, word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
aligner = Aligner()





def shuffle_nouns_and_pronouns(sentence, n):
    # Tokenize the sentence
    words = word_tokenize(sentence)
    # Perform POS tagging
    pos_tags = pos_tag(words)
    
    # Extract the indices of nouns and pronouns
    noun_pronoun_indices = [i for i, (word, tag) in enumerate(pos_tags) if tag.startswith('NN') or tag in ('PRP', 'PRP$')]
    
    # Determine the number of nouns and pronouns to shuffle
    num_to_shuffle = min(n, len(noun_pronoun_indices))
    if num_to_shuffle == 0:
        return sentence  # No nouns or pronouns to shuffle
    
    # Select the noun and pronoun indices to shuffle
    indices_to_shuffle = random.sample(noun_pronoun_indices, num_to_shuffle)
    
    # Extract the nouns and pronouns to shuffle
    words_to_shuffle = [words[i] for i in indices_to_shuffle]
    random.shuffle(words_to_shuffle)
    
    # Place the shuffled words back in their positions
    for i, shuffled_word in zip(indices_to_shuffle, words_to_shuffle):
        words[i] = shuffled_word
    
    # Detokenize the sentence back to a string
    shuffled_sentence = TreebankWordDetokenizer().detokenize(words)
    return shuffled_sentence




def other_random(data_path):
    fake_data = []
    real_data = read_data(data_path)
    for idx in range(len(real_data)):
        fake_sentences = []
        src = real_data[idx]["src"]
        original_sentences = real_data[idx]["sentences"]
        temp_lst = []
        for sentence in original_sentences:
            ordered_src, token_sen, index, _ = aligner.align(src.split(), sentence, multi=False, distance=1)

            tokens = token_sen.split()
            new_seq = [tokens[idx] for idx in range(len(tokens)) if idx not in index]
            source = src.split()
            random.shuffle(source)
            new_seq = source + new_seq
            temp_lst.append(' '.join(new_seq).replace(' .','.').replace(' ,',','))
        obj = {"src": src, "sentences": temp_lst}
        fake_data.append(obj)
    with open(f"shuffle_concept.jsonl", 'w', encoding='utf-8') as file:
        for i in range(len(fake_data)):
            file.write(json.dumps(fake_data[i]) + '\n')
        
# paraphrase_generation [A, B, C, D] -> [A, A*, C, D]
def combination_1(origin_data, para_data):
    result_list = []
    for origninal_obj, para_obj in zip(origin_data, para_data):
        src = origninal_obj['src']
        origin_sentences = origninal_obj['sentences']
        para_sentences = para_obj['sentences']
        temp_combination_list = []
        for i in range(len(para_sentences)):
            current_pair = [para_sentences[i], origin_sentences[i]]
            remaining_A = origin_sentences[:i] + origin_sentences[i+1:]
            
            for combo in combinations(remaining_A, 2):
                temp_lst = current_pair + list(combo)
                temp_combination_list.append(temp_lst)
                
        result_list.append({"src": src, "sentences": temp_combination_list})
    
    # Write each combination to different files
    write_to_files(result_list)
        

# paraphrase_generation [A, B, C, D] -> [A, A*, B, B*]
def combination_2(origin_data, para_data):
    result_list = []
    for origninal_obj, para_obj in zip(origin_data, para_data):
        src = origninal_obj['src']
        origin_sentences = origninal_obj['sentences']
        para_sentences = para_obj['sentences']
        temp_combination_list = []
        for combo in combinations(range(4), 2):
            current_pair = [origin_sentences[combo[0]], para_sentences[combo[0]],  origin_sentences[combo[1]], para_sentences[combo[1]]]
            temp_combination_list.append(current_pair)
            
        result_list.append({"src": src, "sentences": temp_combination_list})

    # Write each combination to different files
    write_to_files(result_list)


# [A.B.C.D]->[A, A', A'', B]
def combination_3(origin_data, para_data1, para_data2):
    result_list = []
    for origninal_obj, para_1, para_2 in zip(origin_data, para_data1, para_data2):
        src = origninal_obj['src']
        
        origin_sentences = origninal_obj['sentences']
        para_sentences1 = para_1['sentences']
        para_sentences2 = para_2['sentences']
        print(src, origin_sentences, para_sentences1, para_sentences2)
        temp_combination_list = []
        for combo in permutations(range(4), 2):
            
            current_pair = [origin_sentences[combo[0]], para_sentences1[combo[0]],  para_sentences2[combo[0]], origin_sentences[combo[1]]]
            temp_combination_list.append(current_pair)
        result_list.append({"src": src, "sentences": temp_combination_list})

    # Write each combination to different files
    write_to_files(result_list)

def combination_4(origin_data, para_data):
    result_list = []
    for origninal_obj, para_obj in zip(origin_data, para_data):
        src = origninal_obj['src']
        origin_sentences = origninal_obj['sentences']
        para_sentences = para_obj['sentences']
        temp_combination_list = []
        # Choose 2 to replace
        for combo in combinations(range(4), 2):
            temp = origin_sentences.copy()
            temp[combo[0]] = para_sentences[combo[0]]
            temp[combo[1]] = para_sentences[combo[1]]
            temp_combination_list.append(temp)
            
        result_list.append({"src": src, "sentences": temp_combination_list})
    #print(result_list)
    write_to_files(result_list)


def combination_5(origin_data, para_data):
    start_loc = 0
    window_size = 9
    result_list = []
    for origninal_obj, para_obj in zip(origin_data, para_data):
        src = origninal_obj['src']
        origin_sentences = origninal_obj['sentences']
        para_sentences = para_obj['sentences']
        temp_combination_list = []
        for combo in combinations(range(4), 2):
            temp = origin_sentences.copy()
            temp[combo[0]] = para_sentences[combo[0]]
            
            words = para_sentences[combo[0]].split()
            perturbed_words = words[:]
            
            window = words[start_loc:start_loc + window_size]
            random.shuffle(window)
            perturbed_words[start_loc:start_loc + window_size] = window
            temp[combo[1]] = ' '.join(perturbed_words)
            
            temp_combination_list.append(temp)
            
        result_list.append({"src": src, "sentences": temp_combination_list})
    #print(result_list)
    write_to_files(result_list)


def write_to_files(result_list):
    # Write each combination to different files
    n = len(result_list[0]['sentences'])
    for i in range(n):
        temp_result = []
        with open(f"temp/comb_a_{i}.jsonl", 'w', encoding='utf-8') as output_file:
            for obj in result_list:
                temp = {"src": obj['src'], "sentences": obj['sentences'][i]}
                temp_result.append(temp)
                output_file.write(json.dumps(temp) + "\n")
        
        print("Writing the combination to file", i)
        # Evaluate the diversity of the generated data
        print(f"Evaluating the Combination {i}")
        results = evaluate_diversity(temp_result)

        
if __name__ == '__main__':    
    
    #Fake data test
    origin_data = [{"src": "A B C D", "sentences": [1, 2,3,4]}]
    para_data = [{"src": "A B C D", "sentences": [5,6,7,8]}]
    

    paths = ["data/original_dataset/good_pairs_gpt/original.jsonl", "data/original_dataset/good_pairs_gpt/para_a.jsonl", "data/original_dataset/good_pairs_gpt/para_b.jsonl", "data/original_dataset/good_pairs_gpt/para_c.jsonl",
             "data/original_dataset/bad_pairs_gpt/nonsensical.jsonl", "data/original_dataset/bad_pairs_gpt/shuffle.jsonl", "data/original_dataset/bad_pairs_gpt/shuffle_nouns.jsonl"]
    for path in paths:
        print('-------------------')
        print(f"Data Path: {path}")    
        origin_data = read_data(path)
        evaluate_diversity(origin_data)
    
    

    
    
    

