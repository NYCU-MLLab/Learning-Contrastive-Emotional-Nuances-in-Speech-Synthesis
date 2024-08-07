import numpy as np
import shutil

wav = '0005_001100'
speaker = '2005'
# Replace these file paths with the actual paths to your data files
energy_file = '/home/ubisoft-laforge-daft-exprt/datasets/mandarin/22050Hz/{}/{}.frames_nrg'.format(speaker, wav)
pitch_file = '/home/ubisoft-laforge-daft-exprt/datasets/mandarin/22050Hz/{}/{}.frames_f0'.format(speaker, wav)
mel_spec_file = '/home/ubisoft-laforge-daft-exprt/datasets/mandarin/22050Hz/{}/{}.npy'.format(speaker, wav)

source_file = '/home/ubisoft-laforge-daft-exprt/data_dir/{}/wavs/{}.wav'.format(speaker, wav)
destination_file = '/home/ubisoft-laforge-daft-exprt/scripts/style_bank/mandarin/{}.wav'.format(wav)
shutil.copy(source_file, destination_file)

# Load data from files
energy_data = np.loadtxt(energy_file)
pitch_data = np.loadtxt(pitch_file)
mel_spec_data = np.load(mel_spec_file)

# Create a .npz file with specified keys
npz_file_path = '/home/ubisoft-laforge-daft-exprt/scripts/style_bank/mandarin/{}.npz'.format(wav)
np.savez(npz_file_path, energy=energy_data, pitch=pitch_data, mel_spec=mel_spec_data)

print(f"Data saved to {npz_file_path}")

# Load the .npz file
data = np.load(npz_file_path)

# Print the keys (names) of the arrays stored in the .npz file
print("Keys in the NPZ file:", list(data.keys()))

# Access and print the contents of each array
for key in data.keys():
    print(f"Content of '{key}':")
    print(data[key])