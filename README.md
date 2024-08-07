## Installation

### Local Environment
Requirements:
- Ubuntu >= 20.04
- Python >= 3.8
- [NVIDIA Driver](https://www.nvidia.com/Download/index.aspx?lang=en-us) >= 450.80.02
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) >= 11.1
- [CuDNN](https://developer.nvidia.com/rdp/cudnn-archive) >= v8.0.5

We recommend using conda for python environment management, for example download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).  
Create your python environment and install dependencies using the Makefile:
1. `conda create -n daft_exprt python=3.8 -y`
2. `conda activate daft_exprt`
3. `cd environment`
4. `make`

All Linux/Conda/Pip dependencies will be installed by the Makefile, and the repository will be installed as a pip package in editable mode.

### Docker Image
Requirements:
- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
- [NVIDIA Driver](https://www.nvidia.com/Download/index.aspx?lang=en-us) >= 450.80.02

Build the Docker image using the associated Dockerfile:  
1. `docker build -f environment/Dockerfile -t daft_exprt .`

## Quick Start Example

### Introduction
This quick start guide will illustrate how to use the different scripts of this repository to:
1. Format datasets
2. Pre-process these datasets
3. Train Daft-Exprt on the pre-processed data
4. Generate a dataset for vocoder fine-tuning
5. Use Daft-Exprt for TTS synthesis

All scripts are located in [scripts](scripts) directory.  
Daft-Exprt source code is located in [daft_exprt](src/daft_exprt) directory.  
Config parameters used in the scripts are all instanciated in [hparams.py](src/daft_exprt/hparams.py).  

As a quick start example, we consider using the 22kHz [LJ speech dataset](https://keithito.com/LJ-Speech-Dataset/) and the 16kHz emotional speech dataset (ESD) from [Zhou et al](https://github.com/HLTSingapore/Emotional-Speech-Data).  
This combines a total of 11 speakers. All speaker datasets must be in the same root directory. For example:
```
/data_dir
    LJ_Speech
    ESD
        spk_1
        ...
        spk_N
```

In this example, we use the docker image built in the previous section:
 ```
docker run -it --gpus all -v /path/to/data_dir:/workdir/data_dir -v path/to/repo_dir:/workdir/repo_dir IMAGE_ID
```


### Dataset Formatting
The source code expects the specific tree structure for each speaker data set:
```
/speaker_dir
    metadata.csv
    /wavs
        wav_file_name_1.wav
        ...
        wav_file_name_N.wav
```

metadata.csv must be formatted as follows:
```
wav_file_name_1|text_1
...
wav_file_name_N|text_N
```

Given each dataset has its own nomenclature, this project does not provide a ready-made universal script.  
However, the script [format_dataset.py](scripts/format_dataset.py) already proposes the code to format LJ and ESD:
```
python format_dataset.py \
    --data_set_dir /workdir/data_dir/LJ_Speech \
    LJ

python format_dataset.py \
    --data_set_dir /workdir/data_dir/ESD \
    ESD \
    --language english
```

### Data Pre-Processing
In this section, the code will:
1. Align data using MFA
2. Extract features for training
3. Create train and validation sets
4. Extract features stats on the train set for speaker standardization

To pre-process all available formatted data (i.e. LJ and ESD in this example):
```
python training.py \
    --experiment_name EXPERIMENT_NAME \
    --data_set_dir /workdir/data_dir \
    pre_process
```

This will pre-process data using the default hyper-parameters that are set for 22kHz audios.  
All outputs related to the experiment will be stored in `/workdir/repo_dir/trainings/EXPERIMENT_NAME`.  

The pre-process function takes several arguments:
- `--features_dir`: absolute path where pre-processed data will be stored. Default to `/workdir/repo_dir/datasets`
- `--proportion_validation`: Proportion of examples that will be in the validation set. Default to `0.1`% per speaker.
- `--nb_jobs`: number of cores to use for python multi-processing. If set to `max`, all CPU cores are used. Default to `6`.

Note that if it is the first time that you pre-process the data, this step will take several hours.  
You can decrease computing time by increasing the `--nb_jobs` parameter.  
    
### Training
Once pre-processing is finished, launch training. To train on all pre-processed data:
```
python training.py \
    --experiment_name EXPERIMENT_NAME \
    --data_set_dir /workdir/data_dir \
    train
```

All outputs related to the experiment will be stored in `/workdir/repo_dir/trainings/EXPERIMENT_NAME`.  

The train function takes several arguments:
- `--checkpoint`: absolute path of a Daft-Exprt checkpoint. Default to `""`
- `--no_multiprocessing_distributed`: disable PyTorch multi-processing distributed training. Default to `False`
- `--world_size`: number of nodes for distributed training. Default to `1`.
- `--rank`: node rank for distributed training. Default to `0`.
- `--master`: url used to set up distributed training. Default to `tcp://localhost:54321`.

These default values will launch a new training starting at iteration 0, using all available GPUs on the machine.  
The code supposes that only 1 GPU is available on the machine.  
Default [batch size](src/daft_exprt/hparams.py#L66) and [gradient accumulation](src/daft_exprt/hparams.py#L67) hyper-parameters are set to values to reproduce the batch size of 48 from the paper.

The code also supports tensorboard logging. To display logging outputs:  
`tensorboard --logdir_spec=EXPERIMENT_NAME:/workdir/repo_dir/trainings/EXPERIMENT_NAME/logs`

### Vocoder Fine-Tuning
Once training is finished, you can create a dataset for vocoder fine-tuning:
```
python training.py \
    --experiment_name EXPERIMENT_NAME \
    --data_set_dir /workdir/data_dir \
    fine_tune \
    --checkpoint CHECKPOINT_PATH
```

Fine-tuning dataset will be stored in `/workdir/repo_dir/trainings/EXPERIMENT_NAME/fine_tuning_dataset`.  

### TTS Synthesis
For an example on how to use Daft-Exprt for TTS synthesis, run the script [synthesize.py](scripts/synthesize.py).  
```
python synthesize.py \
    --output_dir OUTPUT_DIR \
    --checkpoint CHECKPOINT
```

### Pre-trained models
Pre-trained models can be found at https://drive.google.com/drive/folders/1T87vfRtRtkdsEIRCdqyPnkWTnRkuZ4CM?usp=sharing

### Chinese
For chinese, please convert the transcription into pinyin, for example:
兒子趁機玩兒爸爸的手機 -> er2 zi5 chen4 ji1 wan2 er2 ba4 ba4 de5 shou3 ji1
You can use pypinyin package
```python
from pypinyin import pinyin, Style
pinyin_result = pinyin(hanzi_text, style=Style.TONE3, heteronym=True)
```