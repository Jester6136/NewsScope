# -*- coding: utf-8 -*-
from modules.model_architeture.mrc_model import MRCQuestionAnswering
from transformers import AutoTokenizer
import torch
from nltk import word_tokenize
from utils import *

THRESH = 0.7
tokenizer_path = "model/checkpoint-1248-v3"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

def tokenize_function(example):
    global tokenizer  # Use the global tokenizer
    question_word = word_tokenize(example["question"])
    context_word = word_tokenize(example["context"])

    question_sub_words_ids = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(w)) for w in question_word]
    context_sub_words_ids = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(w)) for w in context_word]
    valid = True
    if len([j for i in question_sub_words_ids + context_sub_words_ids for j in
            i]) > tokenizer.max_len_single_sentence - 1:
        valid = False

    question_sub_words_ids = [[tokenizer.bos_token_id]] + question_sub_words_ids + [[tokenizer.eos_token_id]]
    context_sub_words_ids = context_sub_words_ids + [[tokenizer.eos_token_id]]

    input_ids = [j for i in question_sub_words_ids + context_sub_words_ids for j in i]  
    if len(input_ids) > tokenizer.max_len_single_sentence + 2:
        valid = False

    words_lengths = [len(item) for item in question_sub_words_ids + context_sub_words_ids]

    return {
        "input_ids": input_ids,
        "words_lengths": words_lengths,
        "valid": valid
    }

def data_collator(samples):
    global tokenizer  # Use the global tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if len(samples) == 0:
        return {}

    def collate_tokens(values, pad_idx, eos_idx=None, left_pad=False, move_eos_to_beginning=False):
        """Convert a list of 1d tensors into a padded 2d tensor."""
        size = max(v.size(0) for v in values)
        res = values[0].new(len(values), size).fill_(pad_idx)

        def copy_tensor(src, dst):
            assert dst.numel() == src.numel()
            if move_eos_to_beginning:
                assert src[-1] == eos_idx
                dst[0] = eos_idx
                dst[1:] = src[:-1]
            else:
                dst.copy_(src)

        for i, v in enumerate(values):
            copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
        return res

    input_ids = collate_tokens([torch.tensor(item['input_ids']).to(device) for item in samples], pad_idx=tokenizer.pad_token_id)
    attention_mask = torch.zeros_like(input_ids).to(device)
    for i in range(len(samples)):
        attention_mask[i][:len(samples[i]['input_ids'])] = 1
    words_lengths = collate_tokens([torch.tensor(item['words_lengths']).to(device) for item in samples], pad_idx=0)

    batch_samples = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'words_lengths': words_lengths,
    }

    return batch_samples

def extract_answer(inputs, outputs):
    global tokenizer  # Use the global tokenizer
    plain_result = []
    for sample_input, start_logit, end_logit in zip(inputs, outputs.start_logits, outputs.end_logits):
        sample_words_length = sample_input['words_lengths']
        input_ids = sample_input['input_ids']
        # Get the most likely beginning of answer with the argmax of the score
        answer_start = sum(sample_words_length[:torch.argmax(start_logit)])
        # Get the most likely end of answer with the argmax of the score
        answer_end = sum(sample_words_length[:torch.argmax(end_logit) + 1])

        if answer_start <= answer_end:
            answer = tokenizer.convert_tokens_to_string(
                tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
            if answer == tokenizer.bos_token:
                answer = ''
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


def qa_mrc(questions,context):
    global tokenizer  # Use the global tokenizer
    inputs = []
    for question in questions:
        QA_input = {
            'question': question,
            'context': align_text(context)
        }
        inputs.append(tokenize_function(QA_input))
    inputs_ids = data_collator(inputs)
    inputs_ids = {key: value.to(device) for key, value in inputs_ids.items()}
    outputs = model(**inputs_ids)
    answer = extract_answer(inputs, outputs)
    return (answer)

def demo_sys(context):
    global tokenizer  # Use the global tokenizer
    trigger_o = None
    if not trigger_o:
        trigger_a = qa_mrc(["What is the main action in the text?"], context)
        trigger = trigger_a[0]["answer"]
    else:
        trigger = trigger_o
    if not trigger:
        return "Have no trigger"
    else:
        Object_q = f"What was {trigger} affected ?"
        Subject_q = f"Who or what was involved in {trigger} ?"
        Time_q = f"When did the {trigger} happen ?"
        Location_q = f"Where did the {trigger} take place ?"

        
        out_model = qa_mrc([Object_q,Subject_q,Time_q,Location_q], context)
        q,w,e,r = out_model[0],out_model[1],out_model[2],out_model[3]
        
        if q['score_start'] < THRESH:
            Model_infered_Object = "Not clear!"
        else:
            Model_infered_Object = q["answer"] if q["answer"]!="" else "Not found!"
        if w['score_start'] < THRESH:
            Model_infered_Subject = "Not clear!"
        else:
            Model_infered_Subject = w["answer"] if w["answer"]!="" else "Not found!"
        if e['score_start'] < THRESH:
            Model_infered_Time = "Not clear!"
        else:
            Model_infered_Time = e["answer"] if e["answer"]!="" else "Not found!"
        if r['score_start'] < THRESH:
            Model_infered_Location = "Not clear!"
        else:
            Model_infered_Location = r["answer"] if r["answer"]!="" else "Not found!"

        Subject_r = '- Chủ thể: ' + Model_infered_Subject
        Trigger_r = '- Action: ' + trigger
        Object_r = '- Khách thể: ' + Model_infered_Object
        Time_r = '- Thời gian: ' + Model_infered_Time
        Location_r = '- Địa điểm: ' + Model_infered_Location
        return "\n".join([Subject_r, Trigger_r, Object_r, Time_r, Location_r])
        
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_checkpoint = "model/checkpoint-1248-v3"
    model = MRCQuestionAnswering.from_pretrained(model_checkpoint)
    model.to(device)
    text = """Canada công bố kế hoạch loại bỏ trợ cấp cho nhiên liệu hóa thạch. Khí thải phát ra từ một nhà máy lọc dầu ở Fort McMurray, Canada. ( Ảnh: AFP/TTXVN) Ngày 24/7, Canada đã công bố kế hoạch loại bỏ các khoản trợ cấp cho nhiên liệu hóa thạch và trở thành quốc gia đầu tiên trong Nhóm các nền kinh tế phát triển và mới nổi hàng đầu thế giới (G20) thực hiện cam kết năm 2009 nhằm hợp lý hóa và loại bỏ trợ cấp đối với khu vực này. Theo Bộ trưởng Mội trường và Biến đổi khí hậu Steven Guilbeault, các khoản trợ cấp này khuyến khích tiêu dùng lãng phí, làm giảm an ninh năng lượng, cản trở đầu tư vào năng lượng sạch và làm giảm nỗ lực đối phó với biến đổi khí hậu. Ông này khẳng định Canada đang loại bỏ các khoản trợ cấp để sản xuất nhiên liệu hóa thạch ở nước này, ngoại trừ những khoản trợ cấp đó là nhằm giảm lượng khí thải carbon của chính khu vực này. [Công nghệ thu giữ CO2 không thể là "đèn xanh" cho nhiên liệu hóa thạch] Kế hoạch này sẽ áp dụng bằng các biện pháp thuế và phi thuế hiện nay, nhưng Chính phủ chưa hủy bỏ các thỏa thuận trợ cấp nhiều năm đang được thực hiện."""
    items = demo_sys(context = text)
    print(items)
