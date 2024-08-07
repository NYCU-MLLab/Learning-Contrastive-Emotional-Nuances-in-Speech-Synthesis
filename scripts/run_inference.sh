#!/bin/bash

# Specify the path to the main folder
main_folder="/home/ubisoft-laforge-daft-exprt/data_dir"

# Specify the output file
output_file="output.txt"

# Find all subdirectories with a "wavs" folder and count the number of .wav files
find "$main_folder" -type d -name "wavs" -exec bash -c 'echo -n "{} "; find "{}" -type f -name "*.wav" | wc -l' \; |
  while read -r folder count; do
    echo "$folder: $count .wav files" >> "$output_file"
  done

echo "Listing completed. Results saved to $output_file."
