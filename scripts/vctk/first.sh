#!/bin/bash

# Set the path to your main folder
main_folder="/home/ubisoft-laforge-daft-exprt/scripts/VCTK-Corpus"
target_folder="/home/ubisoft-laforge-daft-exprt/data_dir_vctk"

# Create a list of all speaker IDs
speaker_ids=$(find "$main_folder" -type d -name 'p*' -exec basename {} \; | sort -u)

# Loop through each speaker ID
for speaker_id in $speaker_ids; do
    # Create a new folder with the speaker ID
    new_folder="${target_folder}/${speaker_id}"
    mkdir -p "${new_folder}/wavs"

    # Copy all speaker's wav files to the new wavs folder
    cp "${main_folder}/wav48/${speaker_id}"/* "${new_folder}/wavs/"

    # Generate metadata.csv file
    metadata_file="${new_folder}/metadata.csv"
    find "${new_folder}/wavs" -type f -name "${speaker_id}_*.wav" | sort | while read -r wav_file; do
        wav_id=$(basename "$wav_file" .wav | cut -d'_' -f2)
        txt_file="${main_folder}/txt/${speaker_id}/${speaker_id}_${wav_id}.txt"
        transcription=$(cat "$txt_file")
        echo "${speaker_id}_${wav_id}|${transcription}" >> "$metadata_file"
    done

    # Rename the speaker folder by replacing 'p' with '0'
    new_speaker_id=$(echo "$speaker_id" | sed 's/p/0/')
    mv "$new_folder" "${target_folder}/${new_speaker_id}"
done
