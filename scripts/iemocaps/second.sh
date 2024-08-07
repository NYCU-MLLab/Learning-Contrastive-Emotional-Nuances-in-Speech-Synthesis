#!/bin/bash

# Set the base directory
base_dir="/home/ubisoft-laforge-daft-exprt/scripts/IEMOCAP_full_release"

# Set the target directory
target_dir="/home/ubisoft-laforge-daft-exprt/data_dir_iemocaps"

for session in {1..5}; do
    session_dir="/home/ubisoft-laforge-daft-exprt/scripts/IEMOCAP_full_release/Session${session}/dialog/transcriptions"

    find "$session_dir" -type f -name "*.txt" | while read -r file; do
        awk -F'[][]' '{print $1 "|" $3}' "$file" > "${file%.txt}_modified.txt"
    done

    find "$session_dir" -type f -name "*_modified.txt" -exec grep '_M' {} + > "/home/ubisoft-laforge-daft-exprt/data_dir_iemocaps/000$((session*2-1))/metadata.csv"
    find "$session_dir" -type f -name "*_modified.txt" -exec grep '_F' {} + > "/home/ubisoft-laforge-daft-exprt/data_dir_iemocaps/000$((session*2))/metadata.csv"

    # grep '_F' "${session_dir}/*_modified.txt" > /home/ubisoft-laforge-daft-exprt/data_dir_iemocaps/000$((session*2))/metadata.csv

    find "$session_dir" -type f -name "*_modified.txt" -delete

    temp_file=$(mktemp)
    awk -F '|' '{split($1, a, ":"); print a[2] "|" $2}' /home/ubisoft-laforge-daft-exprt/data_dir_iemocaps/000$((session*2-1))/metadata.csv > "$temp_file"
    mv "$temp_file" /home/ubisoft-laforge-daft-exprt/data_dir_iemocaps/000$((session*2-1))/metadata.csv

    temp_file=$(mktemp)
    awk -F '|' '{split($1, a, ":"); print a[2] "|" $2}' /home/ubisoft-laforge-daft-exprt/data_dir_iemocaps/000$((session*2))/metadata.csv > "$temp_file"
    mv "$temp_file" /home/ubisoft-laforge-daft-exprt/data_dir_iemocaps/000$((session*2))/metadata.csv

    sed -i 's/ |: /|/g' /home/ubisoft-laforge-daft-exprt/data_dir_iemocaps/000$((session*2-1))/metadata.csv
    sed -i 's/ |: /|/g' /home/ubisoft-laforge-daft-exprt/data_dir_iemocaps/000$((session*2))/metadata.csv
done