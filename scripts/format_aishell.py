import os
import csv
import re
from pypinyin import pinyin, Style

def remove_pinyin(text):
    # Remove Pinyin characters (e.g., guang3)
    return re.sub(r'\s*\w+\d\s*', '', text)

def remove_hanzi(text):
    # Remove Hanzi characters
    text_without_hanzi = re.sub(r'[\u4e00-\u9fff]', '', text)
    
    # Replace multiple spaces with a single space
    text_with_single_space = re.sub(r'\s+', ' ', text_without_hanzi)
    
    # Remove leading and trailing spaces
    final_result = text_with_single_space.strip()
    
    return final_result

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

# Define the path to the train folder
train_folder = "/home/ubisoft-laforge-daft-exprt/scripts/train/wav"
# train_folder = "/home/ubisoft-laforge-daft-exprt/data_test"
# Define the path to the folder where metadata will be saved
metadata_save_folder ="/home/ubisoft-laforge-daft-exprt/data_dir"
# metadata_save_folder ="/home/ubisoft-laforge-daft-exprt/data_output"

# Iterate through the folders in the train directory
for speaker_folder in os.listdir(train_folder):
    speaker_folder_path = os.path.join(train_folder, speaker_folder)

    # Check if the item in the train folder is a directory
    if os.path.isdir(speaker_folder_path):
        # Extract the speaker number from the folder name
        speaker_number = re.search(r'\d+', speaker_folder).group()
        
        # Create the metadata save folder for the current speaker
        save_folder = os.path.join(metadata_save_folder, speaker_number)
        os.makedirs(save_folder, exist_ok=True)

        metadata_file_path = os.path.join(save_folder, "metadata.csv")

        # Open the metadata CSV file for writing
        with open(metadata_file_path, mode="w", encoding="utf-8", newline="") as metadata_file:
            metadata_writer = csv.writer(metadata_file, delimiter="|")

            # Read the content.txt file for the current speaker
            content_file_path = os.path.join(speaker_folder_path, "content.txt")
            with open("/home/ubisoft-laforge-daft-exprt/scripts/train/content.txt", mode="r", encoding="utf-8") as content_file:
            # with open("/home/ubisoft-laforge-daft-exprt/scripts/train/content_test.txt", mode="r", encoding="utf-8") as content_file:
                for line in content_file:
                    # Split the line into audio file name and transcript content
                    parts = line.strip().split("\t")
                    if len(parts) == 2:
                        audio_file_name, transcript_content = parts
                        # Remove Pinyin characters from the transcript content
                        # cleaned_content = remove_hanzi(transcript_content)
                        cleaned_content = remove_pinyin(transcript_content)
                        cleaned_content = hanzi_to_pinyin(cleaned_content)
                        # Check if the audio file belongs to the current speaker
                        if audio_file_name.startswith(speaker_folder):
                            metadata_writer.writerow([audio_file_name.split('.')[0], cleaned_content])

print("Metadata files created successfully.")

