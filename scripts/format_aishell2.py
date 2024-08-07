import os
import shutil

# Set the paths to your old and new folders
old_folder = "/home/ubisoft-laforge-daft-exprt/scripts/train/wav"
new_folder = "/home/ubisoft-laforge-daft-exprt/data_dir"

# Loop through each speaker folder in the old folder
for speaker_folder in os.listdir(old_folder):
    speaker_folder_path = os.path.join(old_folder, speaker_folder)

    if os.path.isdir(speaker_folder_path):
        # Extract the speaker ID from the folder name (e.g., SSB0005)
        speaker_id = speaker_folder.lstrip("SSB")

        # Create the corresponding new speaker folder
        new_speaker_folder = os.path.join(new_folder, speaker_id, 'wavs')
        os.makedirs(new_speaker_folder, exist_ok=True)

        # Copy .wav files from old to new folder
        for wav_file in os.listdir(speaker_folder_path):
            if wav_file.endswith(".wav"):
                src_path = os.path.join(speaker_folder_path, wav_file)
                dst_path = os.path.join(new_speaker_folder, wav_file)
                shutil.copy(src_path, dst_path)

print("Files copied successfully.")