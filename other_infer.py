# -*- coding: utf-8 -*-
from transformers import pipeline

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
        return "\n".join([Subject_r, Trigger_r, Object_r, Time_r, Location_r])
        
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_checkpoint = "model/checkpoint-1680"
    tokenizer_path = "model/checkpoint-1680"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MRCQuestionAnswering.from_pretrained(model_checkpoint)
    model.to(device)
    text = """Canada công bố kế hoạch loại bỏ trợ cấp cho nhiên liệu hóa thạch. Khí thải phát ra từ một nhà máy lọc dầu ở Fort McMurray, Canada. ( Ảnh: AFP/TTXVN) Ngày 24/7, Canada đã công bố kế hoạch loại bỏ các khoản trợ cấp cho nhiên liệu hóa thạch và trở thành quốc gia đầu tiên trong Nhóm các nền kinh tế phát triển và mới nổi hàng đầu thế giới (G20) thực hiện cam kết năm 2009 nhằm hợp lý hóa và loại bỏ trợ cấp đối với khu vực này. Theo Bộ trưởng Mội trường và Biến đổi khí hậu Steven Guilbeault, các khoản trợ cấp này khuyến khích tiêu dùng lãng phí, làm giảm an ninh năng lượng, cản trở đầu tư vào năng lượng sạch và làm giảm nỗ lực đối phó với biến đổi khí hậu. Ông này khẳng định Canada đang loại bỏ các khoản trợ cấp để sản xuất nhiên liệu hóa thạch ở nước này, ngoại trừ những khoản trợ cấp đó là nhằm giảm lượng khí thải carbon của chính khu vực này. [Công nghệ thu giữ CO2 không thể là "đèn xanh" cho nhiên liệu hóa thạch] Kế hoạch này sẽ áp dụng bằng các biện pháp thuế và phi thuế hiện nay, nhưng Chính phủ chưa hủy bỏ các thỏa thuận trợ cấp nhiều năm đang được thực hiện."""
    text = align_text(text)
    items = demo_sys(context = text)
    print(items)
