#!/bin/bash

# Set the base directory
base_dir="/home/ubisoft-laforge-daft-exprt/scripts/IEMOCAP_full_release"

# Set the target directory
target_dir="/home/ubisoft-laforge-daft-exprt/data_dir_iemocaps"

# Loop through sessions
for session in {1..5}; do
    session_dir="Session${session}"

    # Loop through folders 0001 to 0010
    # for folder in {1..5}; do
    folder_dir1="${target_dir}/000$((session*2-1))"
    folder_dir2="${target_dir}/000$((session*2))"

    # Create necessary directories
    mkdir -p "${folder_dir1}/wavs"
    mkdir -p "${folder_dir2}/wavs"

    # Copy Male files
    find "${base_dir}/${session_dir}/sentences/wav" -name "*_M*.wav" -exec cp {} "${folder_dir1}/wavs" \;

    # Copy Female files
    find "${base_dir}/${session_dir}/sentences/wav" -name "*_F*.wav" -exec cp {} "${folder_dir2}/wavs" \;
    # done
done
