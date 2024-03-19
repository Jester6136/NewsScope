# -*- coding: utf-8 -*-
import gradio as gr
from modules.model_architeture.mrc_model import MRCQuestionAnswering
from transformers import AutoTokenizer, pipeline, RobertaForQuestionAnswering
import torch
from nltk import word_tokenize
from transformers.models.auto import MODEL_FOR_QUESTION_ANSWERING_MAPPING
from utils import *

tokenizer_path = "/mnt/wsl/PHYSICALDRIVE0p1/bagsnlp/NewsScope/NewsScope/model/checkpoint-1680"
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


def qa_mrc(question,context):
    global tokenizer  # Use the global tokenizer
    QA_input = {
        'question': question,
        'context': align_text(context)
    }
    inputs = [tokenize_function(QA_input)]
    inputs_ids = data_collator(inputs)
    inputs_ids = {key: value.to(device) for key, value in inputs_ids.items()}
    outputs = model(**inputs_ids)
    answer = extract_answer(inputs, outputs)[0]
    print(answer)
    # answer be like:answer['answer'],answer['score_start'],answer['score_end'])
    return (answer)

def demo_sys(context):
    global tokenizer  # Use the global tokenizer
    trigger_o = None
    if not trigger_o:
        trigger_a = qa_mrc("What is the main action in the text?", context)
        trigger = trigger_a["answer"]
    else:
        trigger = trigger_o
    if not trigger:
        return "Have no trigger"
    else:
        Object_q = f"What was {trigger} affected ?"
        Subject_q = f"Who or what was involved in {trigger} ?"
        Time_q = f"When did the {trigger} happen ?"
        Location_q = f"Where did the {trigger} take place ?"

        q = qa_mrc(Object_q, context)
        w = qa_mrc(Subject_q, context)
        e = qa_mrc(Time_q, context)
        r = qa_mrc(Location_q, context)
        
        if q['score_start'] < 0.7:
            Model_infered_Object = "Not clear!"
        else:
            Model_infered_Object = q["answer"] if q["answer"]!="" else "Not found!"
        if w['score_start'] < 0.7:
            Model_infered_Subject = "Not clear!"
        else:
            Model_infered_Subject = w["answer"] if w["answer"]!="" else "Not found!"
        if e['score_start'] < 0.7:
            Model_infered_Time = "Not clear!"
        else:
            Model_infered_Time = e["answer"] if e["answer"]!="" else "Not found!"
        if r['score_start'] < 0.7:
            Model_infered_Location = "Not clear!"
        else:
            Model_infered_Location = r["answer"] if r["answer"]!="" else "Not found!"

        Subject_r = '- Chủ thể: ' + Model_infered_Subject
        Trigger_r = '- Action: ' + trigger
        Object_r = '- Khách thể: ' + Model_infered_Object
        Time_r = '- Thời gian: ' + Model_infered_Time
        Location_r = '- Địa điểm: ' + Model_infered_Location
        return "\n".join([Subject_r, Trigger_r, Object_r, Time_r, Location_r, "##  Loại sự kiện: sắp có  ##"])
        
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_checkpoint = "/mnt/wsl/PHYSICALDRIVE0p1/bagsnlp/NewsScope/NewsScope/model/checkpoint-1680"
    model = MRCQuestionAnswering.from_pretrained(model_checkpoint)
    model.to(device)
    
    demo = gr.Interface(
        fn=demo_sys,
        inputs=gr.inputs.Textbox(lines=5, label="Đầu vào báo: bao gồm title và 5 câu sau, nếu title không dấu chấm thì thêm vào!"),
        outputs=gr.outputs.Textbox(label="Câu trả lời"),
        title="NewsScope Dev 1 tuần. Nếu câu trả lời ngố, hãy bấm vào Flag",
        examples=["""Nữ vận động viên Việt Nam ghi dấu ấn ở giải đua khắc nghiệt nhất thế giới. (Dân trí) - VĐV Thanh Vũ là đại diện duy nhất của Việt Nam có mặt ở Thụy Điển thi đấu giải đua khắc nghiệt Montane Lapland Arctic Ultra 2024 trên tuyết và về đích sau 9 ngày tranh tài liên tục. Giải đua Montane Lapland Arctic Ultra (MLAU) 2024 bắt đầu khởi tranh từ ngày 3/3 và kết thúc vào ngày 13/3. Các vận động viên (VĐV) đăng ký theo các cự ly 185km hoặc 500km với điểm xuất phát tại Overkalix (Thụy Điển). Thanh Vũ (tên đầy đủ là Vũ Phương Thanh) là đại diện duy nhất của Việt Nam nằm trong số 19 VĐV tham dự cự ly 500km. Đây là cuộc đua trail (địa hình), người tham dự được chọn một trong 3 hình thức gồm di chuyển bộ, dắt xe đạp vượt tuyết và hoặc ván trượt tuyết. Ngoài việc phải vượt qua quãng đường rất dài, các VĐV còn phải chịu đựng thử thách khắc nghiệt với thời tiết 1 độ C vào ban ngày và -13 độ C vào ban đêm.""","""Xấp xỉ 32.000 tỷ đồng đổ vào thị trường chứng khoán. (Dân trí) - Mặc dù áp lực chốt lời khiến VN-Index điều chỉnh nhưng dòng tiền hỗ trợ rất mạnh, thanh khoản toàn thị trường được đẩy lên xấp xỉ 32.000 tỷ đồng. Tình trạng chốt lời trong phiên hôm nay (14/3) đã khiến chỉ số chính VN-Index điều chỉnh 6,25 điểm tương ứng 0,49% về còn 1.264,26 điểm. Sàn HoSE có 292 mã giảm giá so với 193 mã tăng. Trong đó, riêng rổ VN30 có 22 mã giảm và chỉ có 4 mã tăng giá. Chỉ số VN30-Index giảm 11,96 điểm tương ứng đánh rơi 0,94%, thiệt hại lớn hơn so với VN-Index. Nhà đầu tư đang tìm cơ hội tại những mã cổ phiếu nhỏ. Bằng chứng là trong khi các mã lớn giảm thì chỉ số VNSML-Index đại diện cho cổ phiếu penny vẫn tăng 13,86 điểm tương ứng 0,93%. Trên sàn Hà Nội, HNX-Index tăng 1,48 điểm tương ứng 0,62% và UPCoM-Index nhích nhẹ 0,09 điểm tương ứng 0,1%. Mặc dù độ rộng sàn HoSE nghiêng về phía các mã giảm giá nhưng thị trường đang được hỗ trợ bởi dòng tiền mạnh. Không một mã nào trên sàn HoSE rơi vào trạng thái giảm sàn phiên hôm nay. Chỉ cần giá cổ phiếu điều chỉnh lập tức đã thu hút tiền chực chờ đổ xô vào mua."""]
    )
    
    demo.launch(server_port=5556)
