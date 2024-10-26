# -*- coding: utf-8 -*-
import logging
from transformers import AutoTokenizer
from modules.model_architeture.mrc_model import MRCQuestionAnswering
import torch
from nltk import word_tokenize
import re
import unicodedata

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detailed output
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class MRCSystem:
    dict_map = {
        "òa": "oà", "Òa": "Oà", "ÒA": "OÀ", "óa": "oá", "Óa": "Oá", "ÓA": "OÁ",
        "ỏa": "oả", "Ỏa": "Oả", "ỎA": "OẢ", "õa": "oã", "Õa": "Oã", "ÕA": "OÃ",
        "ọa": "oạ", "Ọa": "Oạ", "ỌA": "OẠ", "òe": "oè", "Òe": "Oè", "ÒE": "OÈ",
        "óe": "oé", "Óe": "Oé", "ÓE": "OÉ", "ỏe": "oẻ", "Ỏe": "Oẻ", "ỎE": "OẺ",
        "õe": "oẽ", "Õe": "Oẽ", "ÕE": "OẼ", "ọe": "oẹ", "Ọe": "Oẹ", "ỌE": "OẸ",
        "ùy": "uỳ", "Ùy": "Uỳ", "ÙY": "UỲ", "úy": "uý", "Úy": "Uý", "ÚY": "UÝ",
        "ủy": "uỷ", "Ủy": "Uỷ", "ỦY": "UỶ", "ũy": "uỹ", "Ũy": "Uỹ", "ŨY": "UỸ",
        "ụy": "uỵ", "Ụy": "Uỵ", "ỤY": "UỴ"
    }

    def __init__(self, model_checkpoint, threshold=0.75):
        logger.info("Initializing MRCSystem...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.model = MRCQuestionAnswering.from_pretrained(model_checkpoint)
        self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.threshold = threshold
        logger.info("MRCSystem initialized with model checkpoint: %s", model_checkpoint)

    def add_space_between_words(self, text):
        logger.debug("Adding space between words...")
        text = unicodedata.normalize('NFKC', text)
        pattern = r'(?<=\w)[.](?=\s*[A-Z])|(?<=[a-z])[.](?=\s*[A-Z])'
        modified_text = re.sub(pattern, '. ', text)
        modified_text = re.sub(r'\s+', ' ', modified_text)
        return modified_text

    def align_text(self, text):
        logger.debug("Aligning text...")
        for i, j in self.dict_map.items():
            text = text.replace(i, j)
        return self.add_space_between_words(text)

    def tokenize_function(self, example):
        question_word = word_tokenize(example["question"])
        context_word = word_tokenize(example["context"])

        question_sub_words_ids = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(w)) for w in question_word]
        context_sub_words_ids = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(w)) for w in context_word]
        valid = True
        if len([j for i in question_sub_words_ids + context_sub_words_ids for j in i]) > self.tokenizer.max_len_single_sentence - 1:
            valid = False

        question_sub_words_ids = [[self.tokenizer.bos_token_id]] + question_sub_words_ids + [[self.tokenizer.eos_token_id]]
        context_sub_words_ids = context_sub_words_ids + [[self.tokenizer.eos_token_id]]

        input_ids = [j for i in question_sub_words_ids + context_sub_words_ids for j in i]
        if len(input_ids) > self.tokenizer.max_len_single_sentence + 2:
            valid = False

        words_lengths = [len(item) for item in question_sub_words_ids + context_sub_words_ids]

        return {
            "input_ids": input_ids,
            "words_lengths": words_lengths,
            "valid": valid
        }

    def data_collator(self, samples):
        logger.info("Collating data samples...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if len(samples) == 0:
            return {}

        def collate_tokens(values, pad_idx):
            size = max(v.size(0) for v in values)
            res = values[0].new(len(values), size).fill_(pad_idx)
            for i, v in enumerate(values):
                res[i][:len(v)] = v
            return res

        input_ids = collate_tokens([torch.tensor(item['input_ids']).to(device) for item in samples], pad_idx=self.tokenizer.pad_token_id)
        attention_mask = torch.zeros_like(input_ids).to(device)
        for i in range(len(samples)):
            attention_mask[i][:len(samples[i]['input_ids'])] = 1
        words_lengths = collate_tokens([torch.tensor(item['words_lengths']).to(device) for item in samples], pad_idx=0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'words_lengths': words_lengths,
        }

    def extract_answer(self, inputs, outputs):
        plain_result = []
        for sample_input, start_logit, end_logit in zip(inputs, outputs.start_logits, outputs.end_logits):
            sample_words_length = sample_input['words_lengths']
            input_ids = sample_input['input_ids']

            answer_start = sum(sample_words_length[:torch.argmax(start_logit)])
            answer_end = sum(sample_words_length[:torch.argmax(end_logit) + 1])

            if answer_start <= answer_end:
                answer = self.tokenizer.convert_tokens_to_string(
                    self.tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
                answer = '' if answer == self.tokenizer.bos_token else answer
            else:
                answer = ''

            score_start = torch.max(torch.softmax(start_logit, dim=-1)).cpu().detach().numpy().tolist()
            score_end = torch.max(torch.softmax(end_logit, dim=-1)).cpu().detach().numpy().tolist()
            plain_result.append({
                "answer": answer,
                "score_start": score_start,
                "score_end": score_end
            })
        return plain_result

    def qa_mrc(self, questions, context):
        inputs = [self.tokenize_function({'question': question, 'context': self.align_text(context)}) for question in questions]
        inputs_ids = self.data_collator(inputs)
        inputs_ids = {key: value.to(self.model.device) for key, value in inputs_ids.items()}
        outputs = self.model(**inputs_ids)
        return self.extract_answer(inputs, outputs)

    def demo_sys(self, context):
        logger.info("Running demo system...")
        trigger_a = self.qa_mrc(["What is the main action in the text?"], context)
        trigger = trigger_a[0]["answer"]

        if not trigger:
            logger.warning("No trigger found.")
            return "Have no trigger"
        
        questions = [
            f"What was {trigger} affected ?", f"Who or what was involved in {trigger} ?",
            f"When did the {trigger} happen ?", f"Where did the {trigger} take place ?"
        ]
        answers = self.qa_mrc(questions, context)
        
        events = {
            "subject": answers[1]["answer"] if answers[1]["score_start"] >= self.threshold else None,
            "trigger": trigger,
            "object": answers[0]["answer"] if answers[0]["score_start"] >= self.threshold else None,
            "time": answers[2]["answer"] if answers[2]["score_start"] >= self.threshold else None,
            "place": answers[3]["answer"] if answers[3]["score_start"] >= self.threshold else None
        }
        logger.info("Demo system completed.")
        return events, context

if __name__ == "__main__":
    mrc_system = MRCSystem(model_checkpoint="jester6136/NewsScope")
    text = """Tàu đổ bộ Mỹ vĩnh viễn chìm vào giấc ngủ đông trên mặt trăng. Tàu đổ bộ không người lái của Mỹ, do công ty tư nhân Intuitive Machines vận hành và có tên Odysseus, đã trở thành tàu thương mại đầu tiên kết thúc số phận trên mặt trăng sau sự kiện đổ bộ lịch sử kể từ thời Apollo. 00:00 Previous Play Next 00:00 / 01:43 Mute Settings Fullscreen Copy video url Play / Pause Mute / Unmute Report a problem Language Share Vidverto Player Intuitive Machines hy vọng con tàu có thể "thức giấc" nếu tiếp nhận được ánh sáng mặt trời như trường hợp của tàu đổ bộ SLIM Nhật Bản hồi tháng 1."""
    items, context = mrc_system.demo_sys(context=text)
    print(items)
