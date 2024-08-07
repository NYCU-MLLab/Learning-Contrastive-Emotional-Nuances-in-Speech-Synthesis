#!/bin/bash

# Specify the path to the main folder
main_folder="/home/ubisoft-laforge-daft-exprt/data_dir"

# Find all subdirectories with a "wavs" folder and count the number of .wav files
find "$main_folder" -type d -name "wavs" -exec bash -c 'echo -n "{} "; find "{}" -maxdepth 1 -type f -name "*.wav" | wc -l' \; |
  while read -r folder count; do
    if [ "$count" -lt 10 ] && [ -n "$folder" ] && [ "$folder" != "." ]; then
      echo "Deleting $folder and its contents (less than 10 .wav files)"
      rm -r "$folder"
    # else
    #   echo "$folder has $count .wav files, keeping it."
    fi
  done
