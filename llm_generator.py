import requests
import random
import argparse
import json
import ollama
from llm_util import openai_chat, remove_numeric_prefixes, upload_batch, evaluate_diversity, read_data
from tqdm import tqdm



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', default="openai", type=str)
    parser.add_argument('--output_folder', default="outputs", type=str)
    parser.add_argument('--data_path', default="data/original_dataset/good_pairs_gpt/original.jsonl", type=str)    
    parser.add_argument('--model_name', default="gpt-3.5-turbo", type=str)
    parser.add_argument('--batch_size', default=1, type=int) 
    parser.add_argument('--top_p',default=1, type=float)
    parser.add_argument('--temperature',default=1, type=float)
    parser.add_argument('--repetition_penalty',default=1, type=float)
    parser.add_argument('--max_tokens',default=1024, type=int)

    return parser.parse_args()



def ollama_generation(args, chat_record):
    ollama_args = {
                "model": args.model_name,
                "messages": chat_record,
                "format":"json",
                "options": {"temperature":args.temperature,}
            }
    
    try:
        response = ollama.chat(**ollama_args)
        # print(response)
        generation = json.loads(response['message']["content"])
        if 'paraphrases' in generation.keys():
            sentences = generation['paraphrases']
            return sentences
        # elif 'sentence' in generation.keys():
        #     sentences = generation['sentence']
        #     return sentences
        else:
            print(response)
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON response: {e}")
        generation = None  # You can return None or handle this case as needed
         
    
        #return generation

def openai_generation(args, chat_record):
    openai_args = {
                "model": args.model_name,
                "messages": chat_record,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "response_format":{ "type": "json_object" },
            }
    response = openai_chat(**openai_args)[0]
    sentences = json.loads(response)['sentences']
    print(sentences)
    return sentences


def orignal_template(sample):
    input_text = sample['src'].replace(" ","', '")
    instruction = """# Instruction
Given a set of specific words, write four short and simple sentences that contains all the required words. The sentence should describe a common scene in daily life, and the concepts should be used in a natural way. 

# Example:
## Example 1:
    { "words": ["watch", "pool", "swim", "kid"], "sentences": [ "The kid watches the pool before going for a swim.", "The kid swam while watching the pool for danger.", "The kid's watch falls into the pool during the swim.", "The kid swims in the pool when I watch him." ] }
    
## Example 2:
{ "words": ["fish", "man", "cook", "eat"], "sentences": [ "The man cooked and ate the fish.", "The fishing man eats the raw, without cooking it.", "The fish was cooked, but the man refuse to eat it.", "The man was cooking when he saw the fish eating." ] }
Output the result in JSON format.\n\n"""
    task = """# Your Task\n {"words": [' """+ input_text + """ ']}"""

    chat_record = [{"role": "system","content":instruction},{"role": "user", "content": task}]    
    return chat_record

# Generate simple but different sentences
def simple_but_different_template(sample):
    input_text = sample['src'].replace(" ",", ")
    instruction = """# Instruction
Given a set of specific words, write four simple sentences that contains all the required words. Each sentence should describe a different commonsense scenario. The scenarios described should vary significantly in meaning. Each sentence must clearly illustrate how slight variations in the arrangement or context of the same words can result in different meanings, all within the framework of everyday life and commonsense.

    # Example:
    ## Example 1:
    { "words": ["watch", "pool", "swim", "kid"], "sentences": [ "The kid watches the pool before going for a swim.", "The kid swam while watching the pool for danger.", "The kid's watch falls into the pool during the swim.", "The kid swims in the pool when I watch him." ] }
    
    ## Example 2:
    { "words": ["fish", "man", "cook", "eat"], "sentences": [ "The man cooked and ate the fish.", "The fishing man eats the raw, without cooking it.", "The fish was cooked, but the man refuse to eat it.", "The man was cooking when he saw the fish eating." ] }
    Output the result in JSON format.
    """
    task = """# Your Task\n {"words": " """+ input_text + """ "}"""

    chat_record = [{"role": "system","content":instruction},{"role": "user", "content": task}]    
    return chat_record

    return chat_record

# You cant paraphrase those required these five words [eye.....]
def paraphrase_template(sample):
    sentence = " \n".join(sample['sentences'])
    input_text = sample['src'].replace(" ",", ")
    instruction = """# Instruction
    For each provided sentence, paraphrase it, ensuring that the original meaning is preserved and that all required keywords are included in the paraphrase. You could apply following methods to paraphrase. 
    1. Passive Voice: Convert sentences from active to passive voice, focusing on the recipient of the action.
    2. Change of Tense: Adjust the verb tense within the sentence. This could involve changing from present to past, past to future, or any other tense modifications appropriate to the context.
    3. Synonym Replacement: Replace words in the sentence with their synonyms except the provided keywords. Care must be taken to ensure that the synonyms fit naturally within the context of the sentence and maintain the original meaning.

    Do not explain the methods in the output. 
    Output JSON\n\n
    """
    example1 = """# Examples\n\n##Example 1\n{ "keywords": ["store", "front", "park", "vehicle"], "original_sentences": [ "I parked my vehicle in front of the store before going in to shop.", "A delivery truck parked its vehicle right in front of the store.", "People often park their vehicles in front of the store when doing grocery shopping.", "She parked the vehicle in front of the store and went inside to browse the new arrivals." ], "paraphrases": [ "My vehicle was parked in front of the store prior to my entrance for shopping.","Right in front of the store, a delivery truck had its vehicle parked.","Vehicles are frequently parked by people in front of the store while they do grocery shopping.", "In front of the store, the vehicle was parked by her before she went inside to look over the new arrivals." ] }\n\n"""
    example2 = """##Example 2\n{ "keywords": ["shake", "dance", "head", "music"], "original_sentences": [ "She loves to shake her head and dance to the music.", "The music was so good that he couldn't help but shake his head and dance.", "They shake their heads and dance wildly whenever the music starts playing.", "He started to dance and shake his head as the music got louder." ], "paraphrases": [ "Her head is shaken and she dances to the music, which she loves.","So good was the music that shaking his head and dancing were uncontrollable for him.","Whenever the music begins, their heads are shaken and they dance wildly.","As the music increased in volume, his head was shaken and he began to dance." ] }\n\n"""
    task = """# Your Task\n {"keywords": " """+ input_text + """ "}, "original_sentences": """ +sentence + "}\n"
    chat_record = [{"role": "system","content": instruction + example1 + example2},{"role": "user", "content": task}]    
    return chat_record

def passive_three_template(sample):
    sentence = " \n".join(sample['sentences'])
    input_text = sample['src'].replace(" ",", ")
    instruction = """# Instruction
    For each provided sentence, apply the following transformation methods to paraphrase, ensuring that the original meaning is preserved and that all required keywords should be included in the paraphrase. 
    1. Passive Voice: Convert sentences from active to passive voice, focusing on the recipient of the action. This method restructures the sentence, often changing its grammatical subject and sometimes adding auxiliary verbs.
    2. Change of Tense: Adjust the verb tense within the sentence. This could involve changing from present to past, past to future, or any other tense modifications appropriate to the context.
    3. Introduction of Modifiers: Enhance the sentence by adding adjectives, adverbs, or modifying phrases to provide more detail, description, or emphasis.
    Each paraphrase should clearly reflect some or all of the three methods. Do not explain the methods in the output. 
    Output JSON\n\n
    """
    example1 = """# Examples\n\n##Example 1\n{ "keywords": ["store", "front", "park", "vehicle"], "original_sentences": [ "I parked my vehicle in front of the store before going in to shop.", "A delivery truck parked its vehicle right in front of the store.", "People often park their vehicles in front of the store when doing grocery shopping.", "She parked the vehicle in front of the store and went inside to browse the new arrivals." ], "paraphrases": [ "In front of the store, the vehicle had been parked by me before I proceeded to shop.", "The vehicle of a delivery truck had been strategically positioned right in front of the store.", "Vehicles are frequently parked in front of the store by shoppers when they come to purchase groceries.", "The vehicle had been parked in front of the store by her, after which she advanced inside to peruse the new arrivals." ] }\n\n"""
    example2 = """##Example 2\n{ "keywords": ["shake", "dance", "head", "music"], "original_sentences": [ "She loves to shake her head and dance to the music.", "The music was so good that he couldn't help but shake his head and dance.", "They shake their heads and dance wildly whenever the music starts playing.", "He started to dance and shake his head as the music got louder." ], "paraphrases": [ "The music was danced to with love, as her head was being shaken.", "Such was the quality of the music that his head had been shaken and he had danced uncontrollably.", "Whenever the music begins to play, their heads are shaken and they dance with wild enthusiasm.", "As the music grew louder, the dancing and head shaking had been started by him." ] }\n\n"""
    task = """# Your Task\n {"keywords": " """+ input_text + """ "}, "original_sentences": """ +sentence + "}\n"
    chat_record = [{"role": "system","content": instruction + example1 + example2},{"role": "user", "content": task}]    
    return chat_record


def shuffle_template(sample):
    sentence = " \n".join(sample['sentences'])
    input_text = sample['src'].replace(" ",", ")
    instruction = """# Instruction: 
For each provided sentence, shuffle all the nouns' (including pronoun)  position in each sentence. 
Output the result in JSON format.

Examples:
Example 1:
{"original_sentences": [ "I had to wait for my wife to finish her shopping.", "Once the shopping was done, my wife and I didn't have to wait any longer.", "He decided to wait outside until his wife could finish her shopping.", "After my wife and I finish our shopping, we can wait at the cafe." ], "shuffle": ["Wife wait her I to my for to shopping finish had.", "Once the my done, and wife I shopping was didn't have wait any to longer.", "Wife wait decided outside his shopping until he finish her could to.", "After my wife I finish shopping, our we wait cafe the at can." ]

Example 2:
{"original_sentences": [ "I stand by the board and throw the knife at it.", "She threw the knife onto the cutting board and stood back.", "The chef stood near the board, ready to throw the knife.", "He stood still, knife in hand, and aimed at the board to throw."],  "shuffle": [ "It I by the and throw knife stand the at board.","She knife board the the knife and back stood threw cutting onto.", "The near knife the ready throw to the chef stood board.", "He knife still, hand, and the stood aimed at board in to the throw."]}\n\n"""
    task = """{"original_sentences": """ +sentence + "}\n"
    chat_record = [{"role": "system","content": instruction},{"role": "user", "content": task}]    
    return chat_record


def absurd_template(sample):
    input_text = sample['src'].replace(" ","', '")
    instruction = """# Instruction 
Given a set of specific concepts, write four sentences that are nonsensical and conflict with commonsense in daily life. Each sentence mush contain all the required words. 

Output the result in JSON format.

# Examples : 
Example 1 :  
{ "words": ['finish', 'wait', 'shopping', 'wife'], sentences": [ "Finish waiting for the shopping wife when the sun sings.", "The wife finished waiting after the shopping mall flew away.", "Shopping for time, the wife decided to wait by finishing a mountain.",  "Wait to finish things, said the shopping wife flying over the moon."] } 

Example 2 :  
{ "words": ['throw', 'knife', 'stand', 'board'], sentences": [ "She decided to throw the board so the knife could stand and dance.","He asked the knife to stand on the sun while throwing a board at the moon.","To make the board happy, she threw the stand and the knife into the sky.","The board told the knife to stand still while throwing a unicorn at the stand."] } \n\n"""
    task = """# Your Task\n {"words": [' """+ input_text + """ '] }\n"""
    chat_record = [{"role": "system","content": instruction},{"role": "user", "content": task}]    
    return chat_record

def absurd_semeval_template(sample):
    input_text = sample['src']
    instruction = """# Instruction
    Given a counterfactual statement, write three explanations for the statement that are nonsensical and conflict with commonsense in daily life.
    
    Output the result in JSON format. 
    
    # Examples:
    Example 1 :
    {"src": "We use book to know the time.", "sentences": ["Books are not typically used to tell time, they are used for eat.", "Books are primarily used for chatting, not for timekeeping.", "Books are not designed or intended for timekeeping purposes."]} 
    """
    task = """# Your Task\n {"src": [' """+ input_text + """ '] }\n"""
    chat_record = [{"role": "system","content": instruction},{"role": "user", "content": task}]    
    return chat_record
   
def batch_inputs(args, template, batch_inputs):
    tasks = []
    for i in range(len(batch_inputs)):
        chat_record = template(batch_inputs[i])
        task = {
        "custom_id": f"task-{i}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": args.model_name,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "response_format":{ "type": "json_object" },
            "messages": chat_record,
            }
        }
        tasks.append(task)
        
    return tasks
 
def retrieve_data(source_file, openai_file, output_file, if_rewrite=True):
    data = []
    with open(source_file, 'r', encoding='utf-8') as source_file, open(openai_file, 'r', encoding='utf-8') as sentence_file, open(output_file, 'w', encoding='utf-8') as output_file:
        for src, line in zip(source_file, sentence_file):
            src_obj = json.loads(src)
            inputs = src_obj['src']
            obj = json.loads(line)["response"]["body"]["choices"][0]["message"]["content"]
            
            generated_sentences = json.loads(obj)["sentences"]
            print(generated_sentences)

            data.append({"src": inputs, "sentences": generated_sentences})
        if if_rewrite:
            for obj in data:
                output_file.write(json.dumps(obj) + "\n")
    evaluate_diversity(data)
    return data    
        


def upload_to_openai(args, template, data_path, upload_path):
    data = read_data(data_path)
    batchs = batch_inputs(args, template, data)
    with open(upload_path, 'w', encoding='utf-8') as output_file:
        for batch in batchs:
            output_file.write(json.dumps(batch) + "\n")
    batch_id = upload_batch(upload_path)  
    return batch_id


def single_paraphrase(sample, template):
    chat_record = template(sample)
    paraphrase = None
    while paraphrase is None:
        paraphrase = ollama_generation(args, chat_record)
    print(sample['src'], paraphrase)
    return paraphrase

    

if __name__ == "__main__":
    args = parse_args()
    words = "You cannot hydrate yourself by drinking water."
    sentences = ["She squeezed her eyes shut and hung her head in shame.", "He hangs his coat, shuts his eyes, and squeezes the stress ball to calm his nerves.", "With her head hanging low, she squeezed her eyes shut to avoid seeing the mess.", "After squeezing the juice, he shut the fridge door and hung the towel over his head."]
    sample = {"src": words, "sentences": sentences}
    chat_record = absurd_semeval_template(sample)
    paraphrase = openai_generation(args, chat_record)
    result = []
    
    
    
    original_data = read_data("good_pairs_gpt/original.jsonl")
    data = read_data("para_2.jsonl")
    short_id = upload_to_openai(args, absurd_semeval_template,"temp/human.jsonl", "temp/nonsensical.jsonl")
    
    
    for obj,origin in tqdm(zip(data,original_data), total=len(data),desc="Generating"):
        if obj['sentences'] is None or len(obj['sentences']) != 4 or len(obj['sentences'][0]) < 4:
            sample = {"src": obj['src'], "sentences": origin['sentences']}
            paraphrase = single_paraphrase(sample, paraphrase_template)
            if isinstance(paraphrase, list) and len(paraphrase) == 4:
                obj['sentences'] = paraphrase
            else:
                obj['sentences'] = None
        result.append(obj)
    with open("para_2.jsonl", 'w', encoding='utf-8') as file:
        for obj in result:
            file.write(json.dumps(obj) + "\n")
    
    # Paraphrase
    # data = read_data("good_pairs_llama3/original.jsonl")
    # for obj in tqdm(data, total=len(data),desc="Generating"):
    #     temp_set = []
    #     for sentence in obj['sentences']:
    #         sample = {"src": obj['src'], "sentences": [sentence]}
    #         paraphrase = single_paraphrase(sample, passive_three_template)
    #         temp_set.append(paraphrase[0])
    #     obj['sentences'] = temp_set
    #     result.append(obj)
    # with open("para_2.jsonl", 'w', encoding='utf-8') as file:
    #     for obj in result:
    #         file.write(json.dumps(obj) + "\n")
        
        