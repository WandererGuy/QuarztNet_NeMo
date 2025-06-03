# intro
this repo shows basic training of QuarztNet from NVIDIA to train for ASR for single language from scratch

# create env samples
conda create -p /home/manh264/code_linux/NeMo/env python==3.10 -y
conda install anaconda::ipykernel
pip install wget
sudo apt-get install sox libsndfile1 ffmpeg
pip install text-unidecode
pip install matplotlib>=3.3.2

# install torch for nemo 
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# install nemo 
sudo apt-get update && sudo apt-get install -y libsndfile1 ffmpeg
pip install Cython packaging
pip install nemo_toolkit['asr']
pip install numpy==1.25.0

# tutorials:
- Vietnamese Automatic Speech Recognition Using NVIDIA – QuartzNet Model
https://www.neurond.com/blog/vietnamese-automatic-speech-recognition-vietasr
- Based on 
https://github.com/NVIDIA/NeMo/blob/main/tutorials/asr/ASR_CTC_Language_Finetuning.ipynb

- Nvidia asr 
https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html#quartznet
- Vietnam asr 
https://github.com/dangvansam/viet-asr
- ASR
https://github.com/NVIDIA/NeMo/blob/main/tutorials/asr/ASR_with_NeMo.ipynb


# usage 
- read https://www.neurond.com/blog/vietnamese-automatic-speech-recognition-vietasr
- use the script main.py 