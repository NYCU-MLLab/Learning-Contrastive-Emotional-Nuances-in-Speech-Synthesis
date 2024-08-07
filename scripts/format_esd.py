import os
import csv
import re
from pypinyin import pinyin, Style

def hanzi_to_pinyin(hanzi_text):
    pinyin_result = pinyin(hanzi_text, style=Style.TONE3, heteronym=True)
    pinyin_list = []

    for item in pinyin_result:
        # Check if the character is a punctuation mark or number
        if not re.match('[\u4e00-\u9fa5a-zA-Z]', item[0]):
            pinyin_list.append(item[0])  # Keep the punctuation or number as it is
        elif not item[0].endswith(('1', '2', '3', '4', '5')):
            # Append a default tone (5) for characters without a specific tone
            pinyin_list.append(item[0] + '5')
        else:
            pinyin_list.append(item[0])

    pinyin_with_spaces = ' '.join(pinyin_list)

    # Remove punctuation
    chinese_punctuation_pattern = re.compile('[^\u4e00-\u9fa5a-zA-Z0-9]')
    pinyin_without_punctuation = chinese_punctuation_pattern.sub(' ', pinyin_with_spaces)

    # Clean up consecutive spaces
    pinyin_cleaned = ' '.join(pinyin_without_punctuation.split())

    return pinyin_cleaned

def process_csv(csv_path):
    output_rows = []
    
    with open(csv_path, 'r', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file, delimiter='|')
        
        for row in reader:
            if len(row) == 2:
                identifier, hanzi_text = row
                pinyin_text = hanzi_to_pinyin(hanzi_text)
                output_rows.append([identifier, pinyin_text])
                
    with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file, delimiter='|')
        writer.writerows(output_rows)

def process_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file == 'metadata.csv':
                csv_path = os.path.join(root, file)
                process_csv(csv_path)

if __name__ == "__main__":
    folder_path = "/home/ubisoft-laforge-daft-exprt/scripts/ESD/mandarin"  # Replace with the actual path to your 'mandarin' folder
    process_folder(folder_path)
