from modules.model_architeture.mrc_model import MRCQuestionAnswering
from transformers import AutoTokenizer, pipeline, RobertaForQuestionAnswering
import torch
from nltk import word_tokenize
from transformers.models.auto import MODEL_FOR_QUESTION_ANSWERING_MAPPING


def tokenize_function(example, tokenizer):
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


def data_collator(samples, tokenizer):
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

    input_ids = collate_tokens([torch.tensor(item['input_ids']) for item in samples], pad_idx=tokenizer.pad_token_id)
    attention_mask = torch.zeros_like(input_ids)
    for i in range(len(samples)):
        attention_mask[i][:len(samples[i]['input_ids'])] = 1
    words_lengths = collate_tokens([torch.tensor(item['words_lengths']) for item in samples], pad_idx=0)

    batch_samples = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'words_lengths': words_lengths,
    }

    return batch_samples


def extract_answer(inputs, outputs, tokenizer):
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


if __name__ == "__main__":
    model_checkpoint = "/data2/cmdir/home/ioit104/aiavn/NewsScope/cache/v1/checkpoint-1680"
    tokenizer_path = "/data2/cmdir/home/ioit104/aiavn/NewsScope/cache/mrc_model"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MRCQuestionAnswering.from_pretrained(model_checkpoint)

    print(model)

    QA_input = {
        'question': "The action ?",
        'context': "Điện Kremlin: Việc mở rộng BRICS sẽ giúp nhóm lớn mạnh hơn. Người phát ngôn Điện Kremlin Dmitry Peskov. (Ảnh: AFP\/TTXVN) Theo hãng tin TASS, ngày 3\/8, người phát ngôn Điện Kremlin Dmitry Peskov cho biết Nga đánh giá việc mở rộng Nhóm các nền kinh tế mới nổi (BRICS) sẽ giúp nhóm lớn mạnh hơn, song khẳng định Nga không đưa ra quan điểm về việc kết nạp một số quốc gia mới trước khi tất cả các nước thành viên thảo luận vấn đề này. Trả lời câu hỏi của báo giới liên quan khả năng Argentina cùng Saudi Arabia và Các Tiểu vương quốc Arab thống nhất (UAE) gia nhập BRICS, ông Peskov nêu rõ Nga tin tưởng rằng dưới bất kỳ hình thức nào, việc mở rộng BRICS sẽ góp phần vào sự phát triển và lớn mạnh hơn nữa của khối. Người phát ngôn Điện Kremlin cho biết thêm Nga có các mối quan hệ mang tính xây dựng với ba quốc gia còn lại trong nhóm, song vẫn còn quá sớm để đề cập các quốc gia ứng cử viên cụ thể trước khi chủ đề này được thảo luận tại Hội nghị Thượng đỉnh BRICS ở Nam Phi vào ngày 22-24\/8 tới. [Hội nghị thượng đỉnh BRICS ưu tiên vấn đề kết nạp thêm thành viên] Trước đó, Đại sứ lưu động của Nam Phi về châu Á và BRICS Anil Sooklal cho biết hiện có khoảng 30 quốc gia quan tâm đến việc gia nhập BRICS."
    }
    inputs = [tokenize_function(QA_input,tokenizer)]
    inputs_ids = data_collator(inputs,tokenizer)
    outputs = model(**inputs_ids)
    answer = extract_answer(inputs, outputs, tokenizer)[0]
    print("answer: {}. Score start: {}, Score end: {}".format(answer['answer'],
                                                                answer['score_start'],
                                                                answer['score_end']))