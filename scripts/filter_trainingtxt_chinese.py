input_file_path = "/home/ubisoft-laforge-daft-exprt/trainings/CHINESE_FIRST/validation_mandarin_full.txt"
output_file_path = "/home/ubisoft-laforge-daft-exprt/trainings/CHINESE_FIRST/validation_mandarin.txt"

with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
    for line in input_file:
        if "SSB" in line.split("|")[1]:
            output_file.write(line)
