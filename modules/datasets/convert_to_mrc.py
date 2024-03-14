import json
from glob import glob
from tqdm import tqdm
import re
import logging
import pandas as pd
from nltk import word_tokenize as lib_tokenizer
from modules.datasets.patterns import TRIGGER_GENERATE,SUBJECT_GENERATE,OBJECT_GENERATE,TIME_GENERATE,PLACE_GENERATE
import argparse
import nltk
nltk.download('punkt')

def convert_to_squad(file_path):
    raw_data = []
    with open(file_path,encoding='utf8') as f:
        for line in f:
            raw_data.append(json.loads(line))

    filtered_data = []
    for item in raw_data:
        data_format = {}
        data_format['id'] = item['id']
        data_format['text'] = item['text']
        for label in item['label']:
            if label[2] not in data_format:
                data_format[label[2]] = [label[0], item['text'][label[0]:label[1]]]
        filtered_data.append(data_format)
    df_json = pd.DataFrame(filtered_data).to_json(orient='records', force_ascii=False)  # force_ascii=False to keep non-ASCII characters
    json_array = json.loads(df_json)
    data = {}
    data["data"] = []
    for item in json_array:
        data_format = {"context": item["text"], "qas": []}
        trigger_question = TRIGGER_GENERATE()
        if item["trigger_1"]:
            data_format["qas"].append({"answers": [{"answer_start": item["trigger_1"][0], "text": item["trigger_1"][1]}], "question": trigger_question})
            arguments = ["subject", "object", "time", "place"]
            
            for argument_type in arguments:
                question_func = globals()[f"{argument_type.upper()}_GENERATE"]
                question = question_func(item["trigger_1"][1])
                answer_key = f"{argument_type}_1"
                
                if item[answer_key] is not None:
                    qas_format = {
                        "answers": [{"answer_start": item[answer_key][0], "text": item[answer_key][1]}],
                        "question": question
                    }
                    data_format["qas"].append(qas_format)
                else:
                    qas_format = {"answers": [], "question": question}
                    data_format["qas"].append(qas_format)
        else:
            data_format["qas"].append({"answers": [], "question": trigger_question})
        paragraphs_format = {"paragraphs":[data_format]}
        data["data"].append(paragraphs_format)
    return data

dict_map = dict({})
def word_tokenize(text):
    global dict_map
    words = text.split()
    words_norm = []
    for w in words:
        if dict_map.get(w, None) is None:
            dict_map[w] = ' '.join(lib_tokenizer(w)).replace('``', '"').replace("''", '"')
        words_norm.append(dict_map[w])
    return words_norm

def strip_answer_string(text):
    text = text.strip()
    while text[-1] in '.,/><;:\'"[]{}+=-_)(*&^!~`':
        if text[0] != '(' and text[-1] == ')' and '(' in text:
            break
        if text[-1] == '"' and text[0] != '"' and text.count('"') > 1:
            break
        text = text[:-1].strip()
    while text[0] in '.,/><;:\'"[]{}+=-_)(*&^!~`':
        if text[0] == '"' and text[-1] != '"' and text.count('"') > 1:
            break
        text = text[1:].strip()
    text = text.strip()
    return text


def strip_context(text):
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def handle_file(json_data):
    qa_data = json_data['data']
    norm_samples = []
    for item in tqdm(qa_data, total=len(qa_data), desc="Chunk data"):
        for par in item['paragraphs']:
            context_raw = par['context']
            for qa_sample in par['qas']:
                question = qa_sample['question']
                if len(qa_sample['answers']) > 0:
                    # if not qa_sample['is_impossible']:
                    answer_raw = qa_sample['answers'][0]['text']
                    answer_index_raw = qa_sample['answers'][0]['answer_start']
                    if context_raw[answer_index_raw: answer_index_raw + len(answer_raw)] == answer_raw:
                        context_prev = strip_context(context_raw[:answer_index_raw])
                        answer = strip_answer_string(answer_raw)
                        context_next = strip_context(context_raw[answer_index_raw + len(answer):])

                        context_prev = ' '.join(word_tokenize(context_prev))
                        context_next = ' '.join(word_tokenize(context_next))
                        answer = ' '.join(word_tokenize(answer))
                        question = ' '.join(word_tokenize(question))

                        context = "{} {} {}".format(context_prev, answer, context_next).strip()

                        norm_samples.append({
                            "context": context,
                            "question": question,
                            "answer_text": answer,
                            "answer_start_idx": len("{} {}".format(context_prev, answer).strip()) - len(answer)
                        })
                else:
                    context_raw = ' '.join(word_tokenize(context_raw))
                    question = ' '.join(word_tokenize(question))
                    norm_samples.append({
                        "context": context_raw,
                        "question": question,
                        "answer_text": '',
                        "answer_start_idx": 0
                    })
    return norm_samples


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Running..")
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_in",
                        default=r"data/raw_data/all.jsonl",
                        type=str,
                        help="")

    parser.add_argument("--file_out",
                        default=r"data/data_processed/squad_mrc.jsonl",
                        type=str,
                        help="")
    args = parser.parse_args()
    squad = handle_file(convert_to_squad(args.file_in))
    with open(args.file_out, 'w', encoding='utf-8') as file:
        for item in squad:
            file.write("{}\n".format(json.dumps(item, ensure_ascii=False)))

    logger.info("Total: {} samples".format(len(squad)))








