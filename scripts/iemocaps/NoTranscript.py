import csv
import os
import wave
    
base_path = '/home/ubisoft-laforge-daft-exprt/datasets/english/22050Hz/'
output = '/home/ubisoft-laforge-daft-exprt/trainings/ENGLISH_FIRST_ALL/ignore.txt'
csv_file_paths = []

for i in range(1, 11):
    file_number = f'{i:04d}'  # This formats the number with leading zeros to make it 4 digits
    file_path = f'{base_path}{file_number}/metadata.csv'
    csv_file_paths.append(file_path)
    
print(csv_file_paths)

# Iterate through each CSV file
count = 0
output_file = open(output, "w")
for csv_file_path in csv_file_paths:
    print(f'\nProcessing file: {csv_file_path}')
    
    # Read the CSV file
    with open(csv_file_path, 'r') as file:
        reader = csv.reader(file, delimiter='|')

        # Iterate through each row in the CSV
        for row in reader:
            # Extract WAV ID and Transcription
            wav_id, transcription = row[0], row[1].strip()

            # Check if transcription is empty
            if not transcription:
                count+=1
                output_file.write(wav_id +'\n')

def get_wav_duration(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        # Get the duration in seconds
        duration = wav_file.getnframes() / float(wav_file.getframerate())
    return duration

def list_long_wav_files(folder_path, duration_threshold=12):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.wav'):
                file_path = os.path.join(root, file)
                duration = get_wav_duration(file_path)
                if duration > duration_threshold:
                    output_file.write(file[:-4]+'\n')
                    # long_wav_files.append(file[:-4])
    

# Specify the path to the root folder containing the nested WAV files
folder_path = '/home/ubisoft-laforge-daft-exprt/data_dir'

# List WAV files with duration greater than 12 seconds
list_long_wav_files(folder_path, duration_threshold=12)

# Print the results
# for wav_file in long_wav_files:
#     print(wav_file)
# print(len(long_wav_files))
# print(long_wav_files[0])
# output_file.close()
# print(count)