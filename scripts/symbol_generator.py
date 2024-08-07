unique_phonemes = set()

# with open("/root/Documents/MFA/pretrained_models/mandarin_pinyin.dict", "r") as file:
with open("/root/Documents/MFA/pretrained_models/dictionary/mandarin_mfa.dict", "r") as file:
    for line in file:
        # Split the line into words and phonemes
        parts = line.split()
        phonemes = parts[1:]

        # Add each phoneme to the set
        unique_phonemes.update(phonemes)

# Convert the set to a sorted list
unique_phonemes_list = sorted(list(unique_phonemes))

print(unique_phonemes_list)
