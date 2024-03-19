# -*- coding: utf-8 -*-
import re
import unicodedata
from nltk.tokenize import sent_tokenize

def add_space_between_words(text):
    text = unicodedata.normalize('NFKC', text)
    pattern = r'(?<=\w)[.](?=\s*[A-Z])|(?<=[a-z])[.](?=\s*[A-Z])'
    modified_text = re.sub(pattern, '. ', text)
    modified_text = re.sub(r'\s+', ' ', modified_text)
    return modified_text

dict_map = {
    "òa": "oà",
    "Òa": "Oà",
    "ÒA": "OÀ",
    "óa": "oá",
    "Óa": "Oá",
    "ÓA": "OÁ",
    "ỏa": "oả",
    "Ỏa": "Oả",
    "ỎA": "OẢ",
    "õa": "oã",
    "Õa": "Oã",
    "ÕA": "OÃ",
    "ọa": "oạ",
    "Ọa": "Oạ",
    "ỌA": "OẠ",
    "òe": "oè",
    "Òe": "Oè",
    "ÒE": "OÈ",
    "óe": "oé",
    "Óe": "Oé",
    "ÓE": "OÉ",
    "ỏe": "oẻ",
    "Ỏe": "Oẻ",
    "ỎE": "OẺ",
    "õe": "oẽ",
    "Õe": "Oẽ",
    "ÕE": "OẼ",
    "ọe": "oẹ",
    "Ọe": "Oẹ",
    "ỌE": "OẸ",
    "ùy": "uỳ",
    "Ùy": "Uỳ",
    "ÙY": "UỲ",
    "úy": "uý",
    "Úy": "Uý",
    "ÚY": "UÝ",
    "ủy": "uỷ",
    "Ủy": "Uỷ",
    "ỦY": "UỶ",
    "ũy": "uỹ",
    "Ũy": "Uỹ",
    "ŨY": "UỸ",
    "ụy": "uỵ",
    "Ụy": "Uỵ",
    "ỤY": "UỴ",
    }

def align_text(text):
    for i, j in dict_map.items():
        text = text.replace(i, j)
    return add_space_between_words(text)

def find_containing_sentence(text, begin_index, end_index):
    sentences = sent_tokenize(text)
    start_index = 0
    space_index = 1
    for sentence in sentences:
        end_index_sentence = start_index + len(sentence) + space_index
        if start_index <= begin_index and end_index_sentence >= end_index:
            return sentence
        start_index = end_index_sentence + 1
    return None

if __name__ == "__main__":
    sample_text = """449 of 511 Yêu cầu một doanh nghiệp trả nợ tiền thuê nhà ở Đà Lạt. (NLĐO) - Số tiền mà Công ty Cổ phần thực phẩm Lâm Đồng còn nợ liên quan đến hợp đồng thuê nhà số 01 Trần Quý Cáp, TP Đà Lạt hơn 700 triệu đồng.Sở Tài chính tỉnh Lâm Đồng có đề nghị lần 3 gửi Công ty cổ phần thực phẩm Lâm Đồng (Công ty CPTP Lâm Đồng) khẩn trương thực hiện nghĩa vụ tài chính khi chấm dứt hợp đồng thuê nhà tại số 01 Trần Quý Cáp, phường 10, TP Đà Lạt. Số tiền mà doanh nghiệp này phải khẩn trương nộp là hơn 704 triệu đồng trước ngày 10-8.9 Trong đó, nộp tiền thuê nhà còn nợ hơn 116 triệu đồng; tiền chậm nộp hơn 264 triệu đồng và tiền giá trị còn lại của căn nhà là gần 324 triệu đồng.Trước đó, Sở Tài chính đã có các thông báo số 511 ngày 8-9-2021, văn bản số 1731 ngày 11-7-2023 gửi đến Công ty CPTP Lâm Đồng đề nghị thanh toán tiền thuê nhà còn nợ, hoàn trả giá trị nhà tháo dỡ nhưng đến nay doanh nghiệp này vẫn chưa thực hiện.Sau thời hạn trên, nếu doanh nghiệp này chưa nộp tiền, Sở Tài chính sẽ báo cáo UBND tỉnh phương án xử lý theo quy định.Liên quan đến việc thuê nhà tại số 01 Trần Quý Cáp, Báo Người Lao Động đã có bài viết"Trắc trở thu hồi đất vàng" về việc UBND tỉnh Lâm Đồng quyết định thu hồi khu đất này của Công ty CPTP Lâm Đồng vì doanh nghiệp vi phạm Luật Đất đai năm 2013 và Thanh tra tỉnh Lâm Đồng kết luận vi phạm tại Kết luận số 121/2016.Tuy nhiên, 2 năm sau ngày tỉnh Lâm Đồng ban hành quyết định thu hồi, UBND TP Đà Lạt chỉ mới tiếp nhận hiện trạng khu đất trên hồ sơ do các sở, ngành bàn giao, còn trên thực địa thì vẫn do doanh nghiệp và một số hộ kinh doanh sử dụng.Lý do được chỉ ra đó là các tổ chức, cá nhân tại số 01 Trần Quý Cáp không hợp tác. Cụ thể như đóng cửa, không nhận giấy mời làm việc hoặc thông báo đi nước ngoài trong thời gian TP Đà Lạt mời làm việc, khởi kiện tại tòa án hoặc cử người không có thẩm quyền tham dự buổi làm việc.Đến nay, khu nhà tại số 01 Trần Quý Cáp đã gần hoàn tất di dời các hộ dân và hộ kinh doanh để giao trả khu đất vàng này về cho UBND TP Đà Lạt quản lý."""
    cleaned_text = align_text(sample_text)
    print(cleaned_text)
