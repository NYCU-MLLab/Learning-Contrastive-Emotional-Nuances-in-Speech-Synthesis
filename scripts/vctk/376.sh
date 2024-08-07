#!/bin/bash

input_file="/home/ubisoft-laforge-daft-exprt/data_dir_vctk/0376/metadata.csv"  # Replace with the actual path to your file

# Check if the file exists
if [ -f "$input_file" ]; then
    # Use sed to remove double quotes and trim unnecessary spaces
    sed 's/"//g; s/ *| */|/g' "$input_file" > "${input_file}.new"
    echo "Processing complete. Output saved to ${input_file}.new"
else
    echo "Error: File not found."
fi
