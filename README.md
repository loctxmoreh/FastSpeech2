# [Moreh] Running on HAC VM - Moreh AI Framework

## Prepare

### Data
Download and extract the [LJSpeech dataset](https://keithito.com/LJ-Speech-Dataset/).
The dataset directory should be named `LJSpeech-1.1`.

### Environment
First, create conda environment:
```bash
conda create -n fastspeech python=3.8
conda activate fastspeech
```
#### On HAC VM
Install `torch` first:
```bash
conda install -y torchvision torchaudio numpy protobuf==3.13.0 pytorch==1.7.1 cpuonly -c pytorch
```
Then force update Moreh with latest version (`22.9.0` at the moment this document is written):
```bash
update-moreh --force --target 22.9.0
```
Comment out `torch` in `requirements.txt`, and then install the rest of the dependencies:
```bash
pip install -r requirements.txt
```
#### On A100 VM
Comment out `torch` in `requirements.txt`, and then install the dependencies:
```bash
pip install -r requirements.txt
```
Then, install `torch` with correct CUDA:
```bash
# torch==1.7.1
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# or torch==1.12.1
conda install pytorch torchvision torchaudio cudatoolkit=1.13 -c pytorch
```
### Code
Clone the repo:
```bash
git clone https://github.com/ming024/FastSpeech2
cd FastSpeech2
```

## Run

Edit `./config/LJSpeech/preprocess.yaml` by changing `path.corpus_path` to point
to the actual location of `LJSpeech-1.1` dataset on the machine.

### Preprocessing
First, run:
```bash
python3 prepare_align.py config/LJSpeech/preprocess.yaml
```
for some preparations.
Then, download the alignment for LJSpeech dataset [here](https://drive.google.com/drive/folders/1DBRkALpPd6FL9gjHMmMEdHODmkgNIIK4?usp=sharing).
The file is named `LJSpeech.zip`. Put it in `./preprocessed_data/LJSpeech/` and
then unzip it. It will be extracted to a directory named `TextGrid` within
`./preprocessed_data/LJSpeech/`.
After that, run the preprocessing script:
```bash
python3 preprocess.py config/LJSpeech/preprocess.yaml
```

### Train
First, edit `./config/LJSpeech/train.yaml` by adjusting `step.total_step`, `step.log_step`
and `step.save_step` to appropriate values. The recommended values are `total_step`
from `900000` to `9000`, `log_step` from `100` to `1000` and `save_step` from
`100000` to `1000`.

Then, extract the HifiGAN model as a vocoder:
```bash
cd hifigan/
unzip ./generator_LJSpeech.pth.tar.zip
cd ../
```

Then, train the model with:
```bash
python3 train.py -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml
```

**NOTE**: when train on A100 VM, you may encounter this error:
```
Traceback (most recent call last):
  File "train.py", line 8, in <module>
    from torch.utils.tensorboard import SummaryWriter
  File "/data/work/anaconda3/envs/fastspeech2-torch/lib/python3.8/site-packages/torch/utils/tensorboard/__init__.py"
, line 8, in <module>
    from .writer import FileWriter, SummaryWriter  # noqa F401
  File "/data/work/anaconda3/envs/fastspeech2-torch/lib/python3.8/site-packages/torch/utils/tensorboard/writer.py",
line 10, in <module>
    from tensorboard.compat.proto.event_pb2 import SessionLog
  File "/data/work/anaconda3/envs/fastspeech2-torch/lib/python3.8/site-packages/tensorboard/compat/proto/event_pb2.p
y", line 17, in <module>
    from tensorboard.compat.proto import summary_pb2 as tensorboard_dot_compat_dot_proto_dot_summary__pb2
  File "/data/work/anaconda3/envs/fastspeech2-torch/lib/python3.8/site-packages/tensorboard/compat/proto/summary_pb$
.py", line 17, in <module>
    from tensorboard.compat.proto import tensor_pb2 as tensorboard_dot_compat_dot_proto_dot_tensor__pb2
  File "/data/work/anaconda3/envs/fastspeech2-torch/lib/python3.8/site-packages/tensorboard/compat/proto/tensor_pb2$
py", line 16, in <module>
    from tensorboard.compat.proto import resource_handle_pb2 as tensorboard_dot_compat_dot_proto_dot_resource__hand$
e__pb2
  File "/data/work/anaconda3/envs/fastspeech2-torch/lib/python3.8/site-packages/tensorboard/compat/proto/resource_h$
ndle_pb2.py", line 16, in <module>
    from tensorboard.compat.proto import tensor_shape_pb2 as tensorboard_dot_compat_dot_proto_dot_tensor__shape__pb$
  File "/data/work/anaconda3/envs/fastspeech2-torch/lib/python3.8/site-packages/tensorboard/compat/proto/tensor_sha$
e_pb2.py", line 36, in <module>
    _descriptor.FieldDescriptor(
  File "/data/work/anaconda3/envs/fastspeech2-torch/lib/python3.8/site-packages/google/protobuf/descriptor.py", lin$
 560, in __new__
    _message.Message._CheckCalledFromGeneratedFile()
TypeError: Descriptors cannot not be created directly.
If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.$
9.0.
If you cannot immediately regenerate your protos, some other possible workarounds are:
 1. Downgrade the protobuf package to 3.20.x or lower.
 2. Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python (but this will use pure-Python parsing and will be much slowe$
).

More information: https://developers.google.com/protocol-buffers/docs/news/2022-05-06#python-updates
```
To resolve this issue by setting the `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` to `python`
```bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

# Original README content:
---
# FastSpeech 2 - PyTorch Implementation

This is a PyTorch implementation of Microsoft's text-to-speech system [**FastSpeech 2: Fast and High-Quality End-to-End Text to Speech**](https://arxiv.org/abs/2006.04558v1).
This project is based on [xcmyz's implementation](https://github.com/xcmyz/FastSpeech) of FastSpeech. Feel free to use/modify the code.

There are several versions of FastSpeech 2.
This implementation is more similar to [version 1](https://arxiv.org/abs/2006.04558v1), which uses F0 values as the pitch features.
On the other hand, pitch spectrograms extracted by continuous wavelet transform are used as the pitch features in the [later versions](https://arxiv.org/abs/2006.04558).

![](./img/model.png)

# Updates
- 2021/7/8: Release the checkpoint and audio samples of a multi-speaker English TTS model trained on LibriTTS
- 2021/2/26: Support English and Mandarin TTS
- 2021/2/26: Support multi-speaker TTS (AISHELL-3 and LibriTTS)
- 2021/2/26: Support MelGAN and HiFi-GAN vocoder

# Audio Samples
Audio samples generated by this implementation can be found [here](https://ming024.github.io/FastSpeech2/).

# Quickstart

## Dependencies
You can install the Python dependencies with
```
pip3 install -r requirements.txt
```

## Inference

You have to download the [pretrained models](https://drive.google.com/drive/folders/1DOhZGlTLMbbAAFZmZGDdc77kz1PloS7F?usp=sharing) and put them in ``output/ckpt/LJSpeech/``,  ``output/ckpt/AISHELL3``, or ``output/ckpt/LibriTTS/``.

For English single-speaker TTS, run
```
python3 synthesize.py --text "YOUR_DESIRED_TEXT" --restore_step 900000 --mode single -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml
```

For Mandarin multi-speaker TTS, try
```
python3 synthesize.py --text "大家好" --speaker_id SPEAKER_ID --restore_step 600000 --mode single -p config/AISHELL3/preprocess.yaml -m config/AISHELL3/model.yaml -t config/AISHELL3/train.yaml
```

For English multi-speaker TTS, run
```
python3 synthesize.py --text "YOUR_DESIRED_TEXT"  --speaker_id SPEAKER_ID --restore_step 800000 --mode single -p config/LibriTTS/preprocess.yaml -m config/LibriTTS/model.yaml -t config/LibriTTS/train.yaml
```

The generated utterances will be put in ``output/result/``.

Here is an example of synthesized mel-spectrogram of the sentence "Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition", with the English single-speaker TTS model.
![](./img/synthesized_melspectrogram.png)

## Batch Inference
Batch inference is also supported, try

```
python3 synthesize.py --source preprocessed_data/LJSpeech/val.txt --restore_step 900000 --mode batch -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml
```
to synthesize all utterances in ``preprocessed_data/LJSpeech/val.txt``

## Controllability
The pitch/volume/speaking rate of the synthesized utterances can be controlled by specifying the desired pitch/energy/duration ratios.
For example, one can increase the speaking rate by 20 % and decrease the volume by 20 % by

```
python3 synthesize.py --text "YOUR_DESIRED_TEXT" --restore_step 900000 --mode single -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml --duration_control 0.8 --energy_control 0.8
```

# Training

## Datasets

The supported datasets are

- [LJSpeech](https://keithito.com/LJ-Speech-Dataset/): a single-speaker English dataset consists of 13100 short audio clips of a female speaker reading passages from 7 non-fiction books, approximately 24 hours in total.
- [AISHELL-3](http://www.aishelltech.com/aishell_3): a Mandarin TTS dataset with 218 male and female speakers, roughly 85 hours in total.
- [LibriTTS](https://research.google/tools/datasets/libri-tts/): a multi-speaker English dataset containing 585 hours of speech by 2456 speakers.

We take LJSpeech as an example hereafter.

## Preprocessing

First, run
```
python3 prepare_align.py config/LJSpeech/preprocess.yaml
```
for some preparations.

As described in the paper, [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/) (MFA) is used to obtain the alignments between the utterances and the phoneme sequences.
Alignments of the supported datasets are provided [here](https://drive.google.com/drive/folders/1DBRkALpPd6FL9gjHMmMEdHODmkgNIIK4?usp=sharing).
You have to unzip the files in ``preprocessed_data/LJSpeech/TextGrid/``.

After that, run the preprocessing script by
```
python3 preprocess.py config/LJSpeech/preprocess.yaml
```

Alternately, you can align the corpus by yourself.
Download the official MFA package and run
```
./montreal-forced-aligner/bin/mfa_align raw_data/LJSpeech/ lexicon/librispeech-lexicon.txt english preprocessed_data/LJSpeech
```
or
```
./montreal-forced-aligner/bin/mfa_train_and_align raw_data/LJSpeech/ lexicon/librispeech-lexicon.txt preprocessed_data/LJSpeech
```

to align the corpus and then run the preprocessing script.
```
python3 preprocess.py config/LJSpeech/preprocess.yaml
```

## Training

Train your model with
```
python3 train.py -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml
```

The model takes less than 10k steps (less than 1 hour on my GTX1080Ti GPU) of training to generate audio samples with acceptable quality, which is much more efficient than the autoregressive models such as Tacotron2.

# TensorBoard

Use
```
tensorboard --logdir output/log/LJSpeech
```

to serve TensorBoard on your localhost.
The loss curves, synthesized mel-spectrograms, and audios are shown.

![](./img/tensorboard_loss.png)
![](./img/tensorboard_spec.png)
![](./img/tensorboard_audio.png)

# Implementation Issues

- Following [xcmyz's implementation](https://github.com/xcmyz/FastSpeech), I use an additional Tacotron-2-styled Post-Net after the decoder, which is not used in the original FastSpeech 2.
- Gradient clipping is used in the training.
- In my experience, using phoneme-level pitch and energy prediction instead of frame-level prediction results in much better prosody, and normalizing the pitch and energy features also helps. Please refer to ``config/README.md`` for more details.

Please inform me if you find any mistakes in this repo, or any useful tips to train the FastSpeech 2 model.

# References
- [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558), Y. Ren, *et al*.
- [xcmyz's FastSpeech implementation](https://github.com/xcmyz/FastSpeech)
- [TensorSpeech's FastSpeech 2 implementation](https://github.com/TensorSpeech/TensorflowTTS)
- [rishikksh20's FastSpeech 2 implementation](https://github.com/rishikksh20/FastSpeech2)

# Citation
```
@INPROCEEDINGS{chien2021investigating,
  author={Chien, Chung-Ming and Lin, Jheng-Hao and Huang, Chien-yu and Hsu, Po-chun and Lee, Hung-yi},
  booktitle={ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  title={Investigating on Incorporating Pretrained and Learnable Speaker Representations for Multi-Speaker Multi-Style Text-to-Speech},
  year={2021},
  volume={},
  number={},
  pages={8588-8592},
  doi={10.1109/ICASSP39728.2021.9413880}}
```
