## Brief Introduction

JOSIEv4-o build on NExt-GPT is built on top of existing pre-trained LLM, multimodal encoder and SoTA diffusion models, with sufficient end-to-end instruction tuning.

<p align="center" width="100%">
<a target="_blank"><img src="figures/framework.png" alt="Video-LLaMA" style="width: 90%; min-width: 200px; display: block; margin: auto;"></a>
</p>

- **Multimodal Encoding Stage.** Leveraging established encoders to encode inputs in various modalities, where these representations are projected into language-like representations comprehensible to the LLM through a projection layer.
- **LLM Understanding and Reasoning Stage.** Harnessing an existing open-sourced LLM as the core to process input information for semantic understanding and reasoning. The LLM not only directly generates text tokens but also produces unique “modality signal” tokens that serve as instructions to dictate the decoding layers whether & what modal content to output correspondingly.
- **Multimodal Generation Stage.** Receiving the multimodal signals with specific instructions from LLM (if any), the Transformer-based output projection layers map the signal token representations into the ones that are understandable to following multimodal decoders.

For more technical details, kindly refer to the [paper](https://arxiv.org/pdf/2309.05519.pdf).

---

<span id='Usage'/>

## Getting Started

<span id='all_catelogue'/>

### Table of Contents:

- <a href='#Code Structure'>1. Code Structure</a>
- <a href='#Environment Preparation'>2. Environment Preparation </a>
- <a href='#Training on Your Own'>3. Training/Adapting NExt-GPT on Your Own</a>
  - <a href='#Prepare Pre-trained Checkpoint'>3.1. Preparing Pre-trained Checkpoint</a>
  - <a href='#Prepare Dataset'>3.2. Preparing Dataset </a>
  - <a href='#Precompute Embeddings'>3.3. Precomputing Embeddings</a>
  - <a href='#Train NExT-GPT'>3.4. Training NExT-GPT</a>
- <a href='#Run NExT-GPT System'>4. Running NExT-GPT System</a>
  - <a href='#Prepare checkpoints'>4.1. Preparing checkpoints</a>
  - <a href='#Deploy Demo System'>4.2. Deploying Demo System</a>

---

<span id='Code Structure'/>

### 1. Code Structure

```
├── figures
├── data
│   ├── T-X_pair_data
│   │   ├── audiocap                      # text-autio pairs data
│   │   │   ├── audios                    # audio files
│   │   │   └── audiocap.json             # the audio captions
│   │   ├── cc3m                          # text-image paris data
│   │   │   ├── images                    # image files
│   │   │   └── cc3m.json                 # the image captions
│   │   └── webvid                        # text-video pairs data
│   │   │   ├── videos                    # video files
│   │   │   └── webvid.json               # the video captions
│   ├── IT_data                           # instruction data
│   │   ├── T+X-T_data                    # text+[image/audio/video] to text instruction data
│   │   │   ├── alpaca                    # textual instruction data
│   │   │   ├── llava                     # visual instruction data
│   │   ├── T-T+X                         # synthesized text to text+[image/audio/video] instruction data
│   │   └── MosIT                         # Modality-switching Instruction Tuning instruction data
├── code
│   ├── config
│   │   ├── base.yaml                     # the model configuration
│   │   ├── stage_1.yaml                  # enc-side alignment training configuration
│   │   ├── stage_2.yaml                  # dec-side alignment training configuration
│   │   └── stage_3.yaml                  # instruction-tuning configuration
│   ├── dsconfig
│   │   ├── stage_1.json                  # deepspeed configuration for enc-side alignment training
│   │   ├── stage_2.json                  # deepspeed configuration for dec-side alignment training
│   │   └── stage_3.json                  # deepspeed configuration for instruction-tuning training
│   ├── datast
│   │   ├── base_dataset.py
│   │   ├── catalog.py                    # the catalog information of the dataset
│   │   ├── cc3m_datast.py                # process and load text-image pair dataset
│   │   ├── audiocap_datast.py            # process and load text-audio pair dataset
│   │   ├── webvid_dataset.py             # process and load text-video pair dataset
│   │   ├── T+X-T_instruction_dataset.py  # process and load text+x-to-text instruction dataset
│   │   ├── T-T+X_instruction_dataset.py  # process and load text-to-text+x instruction dataset
│   │   └── concat_dataset.py             # process and load multiple dataset
│   ├── model
│   │   ├── ImageBind                     # the code from ImageBind Model
│   │   ├── common
│   │   ├── anyToImageVideoAudio.py       # the main model file
│   │   ├── agent.py
│   │   ├── modeling_llama.py
│   │   ├── custom_ad.py                  # the audio diffusion
│   │   ├── custom_sd.py                  # the image diffusion
│   │   ├── custom_vd.py                  # the video diffusion
│   │   ├── layers.py                     # the output projection layers
│   │   └── ...
│   ├── scripts
│   │   ├── train.sh                      # training NExT-GPT script
│   │   └── app.sh                        # deploying demo script
│   ├── header.py
│   ├── process_embeddings.py             # precompute the captions embeddings
│   ├── train.py                          # training
│   ├── inference.py                      # inference
│   ├── demo_app.py                       # deploy Gradio demonstration
│   └── ...
├── ckpt
│   ├── delta_ckpt                        # tunable NExT-GPT params
│   │   ├── nextgpt
│   │   │   ├── 7b_tiva_v0                # the directory to save the log file
│   │   │   │   ├── log                   # the logs
│   └── ...
│   ├── pretrained_ckpt                   # frozen params of pretrained modules
│   │   ├── imagebind_ckpt
│   │   │   ├──huge                       # version
│   │   │   │   └──imagebind_huge.pth
│   │   ├── vicuna_ckpt
│   │   │   ├── 7b_v0                     # version
│   │   │   │   ├── config.json
│   │   │   │   ├── pytorch_model-00001-of-00002.bin
│   │   │   │   ├── tokenizer.model
│   │   │   │   └── ...
├── LICENCE.md
├── README.md
└── requirements.txt
```

<span id='Environment Preparation'/>

### 2. Environment Preparation <a href='#all_catelogue'>[Back to Top]</a>

Please first clone the repo and install the required environment, which can be done by running the following commands:

```
conda env create -n nextgpt python=3.8

conda activate nextgpt

# CUDA 11.6
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia

git clone https://github.com/NExT-GPT/NExT-GPT.git
cd NExT-GPT

pip install -r requirements.txt
```

<span id='Training on Your Own'/>

### 3. Training/Adapting NExt-GPT on Your Own

####

<span id='Prepare Pre-trained Checkpoint'/>

#### 3.1. Preparing Pre-trained Checkpoint <a href='#all_catelogue'>[Back to Top]</a>

NExT-GPT is trained based on following excellent existing models.
Please follow the instructions to prepare the checkpoints.

- `ImageBind`
  is the unified image/video/audio encoder. The pre-trained checkpoint can be downloaded from [here](https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth) with version `huge`. Afterward, put the `imagebind_huge.pth` file at [[./ckpt/pretrained_ckpt/imagebind_ckpt/huge]](ckpt/pretrained_ckpt/imagebind_ckpt/).
- `Vicuna`:
  first prepare the LLaMA by following the instructions [[here]](ckpt/pretrained_ckpt/prepare_vicuna.md). Then put the pre-trained model at [[./ckpt/pretrained_ckpt/vicuna_ckpt/]](ckpt/pretrained_ckpt/vicuna_ckpt/).
- `Image Diffusion`
  is used to generate images. NExT-GPT uses [Stable Diffusion](https://huggingface.co/runwayml/stable-diffusion-v1-5) with version `
v1-5`. (_will be automatically downloaded_)
- `Audio Diffusion`
  for producing audio content. NExT-GPT employs [AudioLDM](https://github.com/haoheliu/AudioLDM) with version `l-full`. (_will be automatically downloaded_)
- `Video Diffusion`
  for the video generation. We employ [ZeroScope](https://huggingface.co/cerspense/zeroscope_v2_576w) with version `v2_576w`. (_will be automatically downloaded_)

<span id='Prepare Dataset'/>

#### 3.2. Preparing Dataset <a href='#all_catelogue'>[Back to Top]</a>

Please download the following datasets used for model training:

A) T-X pairs data

- `CC3M` of **_text-image_** pairs, please follow this instruction [[here]](./data/T-X_pair_data/cc3m/prepare.md). Then put the data at [[./data/T-X_pair_data/cc3m]](./data/T-X_pair_data/cc3m).
- `WebVid` of **_text-video_** pairs, see the [[instruction]](./data/T-X_pair_data/webvid/prepare.md). The file should be saved at [[./data/T-X_pair_data/webvid]](./data/T-X_pair_data/webvid).
- `AudioCap` of **_text-audio_** pairs, see the [[instruction]](./data/T-X_pair_data/audiocap/prepare.md). Save the data in [[./data/T-X_pair_data/audiocap]](./data/T-X_pair_data/audiocap).

B) Instruction data

- T+X-T

  - `LLaVA` of the **_visual instruction data_**, download it from [here](https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md), and then put it at [[./data/IT_data/T+X-T_data/llava]](./data/IT_data/T+X-T_data/llava/).
  - `Alpaca` of the **_textual instruction data_**, download it from [here](https://github.com/tatsu-lab/stanford_alpaca), and then put it at [[./data/IT_data/T+X-T_data/alpaca/]](data/IT_data/T+X-T_data/alpaca/).
  - `VideoChat`, download the **_video instruction data_** [here](https://github.com/OpenGVLab/InternVideo/tree/main/Data/instruction_data), and then put it at [[./data/IT_data/T+X-T_data/videochat/]](data/IT_data/T+X-T_data/videochat/).

  Side note：After downloading dataset, please run `preprocess_dataset.py` to preprocess the dataset into a unified format.

- T-X+T (T2M)

  - The `T-X+T` instruction datasets (T2M) are saved at [[./data/IT_data/T-T+X_data]](./data/IT_data/T-T+X_data).

- MosIT
  - Download the file from [here](), put them in [[./data/IT_data/MosIT_data/]](./data/IT*data/MosIT_data/). (\_We are in the process of finalizing the data and handling the copyright issue. Will release later.*)

<span id='Precompute Embeddings'/>

#### 3.3. Precomputing Embeddings <a href='#all_catelogue'>[Back to Top]</a>

In decoding-side alignment training, we minimize the distance between the representation of signal tokens and captions.
To save costs of time and memory, we precompute the text embeddings for image, audio and video captions using the text encoder within the respective diffusion models.

Please run this command before the following training of NExT-GPT, where the produced `embedding` file will be saved at [[./data/embed]](./data/embed).

```angular2html
cd ./code/
python process_embeddings.py ../data/T-X_pair_data/cc3m/cc3m.json image ../data/embed/ runwayml/stable-diffusion-v1-5
```

Note of arguments:

- args[1]: path of caption file;
- args[2]: modality, which can be `image`, `video`, and `audio`;
- args[3]: saving path of embedding file;
- args[4]: corresponding pre-trained diffusion model name.

<span id='Train NExT-GPT'/>

#### 3.4. Training NExT-GPT <a href='#all_catelogue'>[Back to Top]</a>

First of all, please refer to the base configuration file [[./code/config/base.yaml]](./code/config/base.yaml) for the basic system setting of overall modules.

Then, the training of NExT-GPT starts with this script:

```angular2html
cd ./code
bash scripts/train.sh
```

Specifying the command:

```angular2html
deepspeed --include localhost:0 --master_addr 127.0.0.1 --master_port 28459 train.py \
    --model nextgpt \
    --stage 1\
    --save_path  ../ckpt/delta_ckpt/nextgpt/7b_tiva_v0/\
    --log_path ../ckpt/delta_ckpt/nextgpt/7b_tiva_v0/log/
```

where the key arguments are:

- `--include`: `localhost:0` indicating the GPT cuda number `0` of deepspeed.
- `--stage`: training stage.
- `--save_path`: the directory which saves the trained delta weights. This directory will be automatically created.
- `--log_path`: the directory which saves the log file.

The whole NExT-GPT training involves 3 steps:

- **Step-1**: Encoding-side LLM-centric Multimodal Alignment. This stage trains the **_input projection layer_** while freezing the ImageBind, LLM, output projection layer.

  Just run the above `train.sh` script by setting: `--stage 1`

  Also refer to the running config file [[./code/config/stage_1.yaml]](./code/config/stage_1.yaml) and deepspeed config file [[./code/dsconfig/stage_1.yaml]](./code/dsconfig/stage_1.yaml) for more step-wise configurations.

  Note that the dataset used for training in this step is included `dataset_name_list` and the dataset name must precisely match the definition in [[./code/dataset/catalog.py]](./code/dataset/catalog.py)

- **Step-2**: Decoding-side Instruction-following Alignment. This stage trains the **_output projection layers_** while freezing the ImageBind, LLM, input projection layers.

  Just run the above `train.sh` script by setting: `--stage 2`

  Also refer to the running config file [[./code/config/stage_2.yaml]](./code/config/stage_2.yaml) and deepspeed config file [[./code/dsconfig/stage_2.yaml]](./code/dsconfig/stage_2.yaml) for more step-wise configurations.

- **Step-3**: Instruction Tuning. This stage instruction-tune 1) the **_LLM_** via LoRA, 2) **_input projection layer_** and 3) **_output projection layer_** on the instruction dataset.

  Just run the above `train.sh` script by setting: `--stage 3`

  Also refer to the running config file [[./code/config/stage_3.yaml]](./code/config/stage_3.yaml) and deepspeed config file [[./code/dsconfig/stage_3.yaml]](./code/dsconfig/stage_3.yaml) for more step-wise configurations.

<span id='Run NExT-GPT System'/>

## 4. Running NExT-GPT System <a href='#all_catelogue'>[Back to Top]</a>

<span id='Prepare checkpoints'/>

#### 4.1. Preparing Checkpoints

First, loading the pre-trained NExT-GPT system.

- **Step-1**: load `Frozen parameters`. Please refer to <a href='#Prepare Pre-trained Checkpoint'>3.1 Preparing Pre-trained Checkpoint</a>.

- **Step-2**: load `Tunable parameters`. Please put the NExT-GPT system at [[./ckpt/delta_ckpt/nextgpt/7b_tiva_v0]](./ckpt/delta_ckpt/nextgpt/7b_tiva_v0). You may either 1) use the params trained yourselves, or 2) download our checkpoints from [Huggingface](https://huggingface.co/ChocoWu/nextgpt_7b_tiva_v0).

<span id='Deploy Demo System'/>

#### 4.2. Deploying Gradio Demo

Upon completion of the checkpoint loading, you can run the demo locally via:

```angular2html
cd ./code
bash scripts/app.sh
```

Specifying the key arguments as:

- `--nextgpt_ckpt_path`: the path of pre-trained NExT-GPT params.

---

## Contact

For any questions or feedback, feel free to contact [Shengqiong Wu](mailto:swu@u.nus.edu) and [Hao Fei](mailto:haofei37@nus.edu.sg).

## Citation

If you find NextGPT useful in your research or applications, please kindly cite:

```
@articles{wu2023nextgpt,
  title={NExT-GPT: Any-to-Any Multimodal LLM},
  author={Shengqiong Wu and Hao Fei and Leigang Qu and Wei Ji and Tat-Seng Chua},
  journal = {CoRR},
  volume = {abs/2309.05519},
  year={2023}
}
```

## Acknowledgements

You may refer to related work that serves as foundations for our framework and code repository,
[Vicuna](https://github.com/lm-sys/FastChat),
[ImageBind](https://github.com/facebookresearch/ImageBind),
[Stable Diffusion](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img),
[AudioLDM](https://github.com/haoheliu/AudioLDM), and
[Zeroscope](https://huggingface.co/cerspense/zeroscope_v2_576w).
We also partially draw inspirations from
[PandaGPT](https://github.com/yxuansu/PandaGPT),
[VPGTrans](https://vpgtrans.github.io/),
[GILL](https://github.com/kohjingyu/gill/),
[CoDi](https://codi-gen.github.io/),
[Video-LLaMA](https://github.com/DAMO-NLP-SG/Video-LLaMA),
and [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4).
Thanks for their wonderful works.

## License Notices

This repository is under [BSD 3-Clause License](LICENSE.txt).
NExT-GPT is a research project intended for non-commercial use only.
One must NOT use the code of NExT-GPT for any illegal, harmful, violent, racist, or sexual purposes.
One is strictly prohibited from engaging in any activity that will potentially violate these guidelines.
Any potential commercial use of this code should be approved by the authors.
