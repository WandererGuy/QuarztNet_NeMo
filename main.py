import os
import glob
import json
import subprocess
import librosa
import tarfile
from tqdm.auto import tqdm
import wget
import copy
from omegaconf import OmegaConf, open_dict
import nemo
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.utils import logging, exp_manager

data_dir = '.'
LANGUAGE = 'vi'
# Function to build a manifest
def build_manifest(transcripts_path, manifest_path, wav_path):
    with open(transcripts_path, 'r',encoding='utf8') as fin:
        with open(manifest_path, 'w', encoding='utf8') as fout:
            for line in fin:

                transcript = line[line.find(' ') : -1].lower()
                transcript = transcript.strip()

                file_id = line[: line.find(' ')]

                audio_path = os.path.join(
                    data_dir, wav_path,
                    file_id[file_id.find('V') : file_id.rfind('_')],
                    file_id + '.wav')

                duration = librosa.core.get_duration(filename=audio_path)

                # Write the metadata to the manifest
                metadata = {
                    "audio_filepath": audio_path,
                    "duration": duration,
                    "text": transcript
                    }
                json.dump(metadata, fout, ensure_ascii=False)
                fout.write('\n')


def read_manifest(path):
    manifest = []
    with open(path, 'r', encoding = 'utf8') as f:
        for line in tqdm(f, desc="Reading manifest data"):
            line = line.replace("\n", "")
            data = json.loads(line)
            manifest.append(data)
    return manifest


print ("############################## Building Manifests ##############################")
print("******")
train_transcripts = data_dir + '/vivos/train/prompts.txt'
train_manifest = data_dir + '/vivos/train_manifest.json'
if not os.path.isfile(train_manifest):
    build_manifest(train_transcripts, train_manifest, 'vivos/train/waves')
    print("Train manifest created.")

test_transcripts = data_dir + '/vivos/test/prompts.txt'
test_manifest = data_dir + '/vivos/test_manifest.json'
if not os.path.isfile(test_manifest):
    build_manifest(test_transcripts, test_manifest, 'vivos/test/waves')
    print("Test manifest created.")
print("***Done***")

train_manifest_data = read_manifest(train_manifest)
test_manifest_data = read_manifest(test_manifest)

train_text = [data['text'] for data in train_manifest_data]
test_text = [data['text'] for data in test_manifest_data]

print ("############################## Building Character Set ##############################")
from collections import defaultdict

def get_charset(manifest_data):
    charset = defaultdict(int)
    for row in tqdm(manifest_data, desc="Computing character set"):
        text = row['text']
        for character in text:
            charset[character] += 1
    return charset

train_charset = get_charset(train_manifest_data)
test_charset = get_charset(test_manifest_data)

train_set = set(train_charset.keys())
test_set = set(test_charset.keys())

print(f"Number of tokens in train set : {len(train_set)}")
print(f"Number of tokens in test set : {len(test_set)}")

print ("############################## prepare Character Encoding CTC model ##############################")
char_model = nemo_asr.models.ASRModel.from_pretrained("stt_en_quartznet15x5", map_location='cpu')
char_model.change_vocabulary(new_vocabulary=list(train_set))


print ("############################## UNFRẺEZE ENCODER  ##############################")
#@title Freeze Encoder { display-mode: "form" }
freeze_encoder = False #@param ["False", "True"] {type:"raw"}
freeze_encoder = bool(freeze_encoder)

import torch
import torch.nn as nn

def enable_bn_se(m):
    if type(m) == nn.BatchNorm1d:
        m.train()
        for param in m.parameters():
            param.requires_grad_(True)

    if 'SqueezeExcite' in type(m).__name__:
        m.train()
        for param in m.parameters():
            param.requires_grad_(True)

if freeze_encoder:
  char_model.encoder.freeze()
  char_model.encoder.apply(enable_bn_se)
  logging.info("Model encoder has been frozen, and batch normalization has been unfrozen")
else:
  char_model.encoder.unfreeze()
  logging.info("Model encoder has been un-frozen")

print ("############################## UPDATE CONFIG ##############################")
#### update character set of model
char_model.cfg.labels = list(train_set)
cfg = copy.deepcopy(char_model.cfg)

print ("############################## SET UP DATALOADER ##############################")
# Setup train, validation, test configs
with open_dict(cfg):
  # Train dataset  (Concatenate train manifest cleaned and dev manifest cleaned)
  cfg.train_ds.manifest_filepath = f"{train_manifest}"
  cfg.train_ds.labels = list(train_set)
  cfg.train_ds.normalize_transcripts = False
  cfg.train_ds.batch_size = 32
  cfg.train_ds.num_workers = 8
  cfg.train_ds.pin_memory = True
  cfg.train_ds.trim_silence = True

  # Validation dataset  (Use test dataset as validation, since we train using train + dev)
  cfg.validation_ds.manifest_filepath = test_manifest
  cfg.validation_ds.labels = list(train_set)
  cfg.validation_ds.normalize_transcripts = False
  cfg.validation_ds.batch_size = 8
  cfg.validation_ds.num_workers = 8
  cfg.validation_ds.pin_memory = True
  cfg.validation_ds.trim_silence = True

# setup data loaders with new configs
char_model.setup_training_data(cfg.train_ds)
char_model.setup_multiple_validation_data(cfg.validation_ds)

# Original optimizer + scheduler
print(OmegaConf.to_yaml(char_model.cfg.optim))

with open_dict(char_model.cfg.optim):
  char_model.cfg.optim.lr = 5e-5
  char_model.cfg.optim.betas = [0.95, 0.5]  # from paper
  char_model.cfg.optim.weight_decay = 0.001  # Original weight decay
  char_model.cfg.optim.sched.warmup_steps = None  # Remove default number of steps of warmup
  char_model.cfg.optim.sched.warmup_ratio = None
  char_model.cfg.optim.sched.min_lr = 0.0

print(OmegaConf.to_yaml(char_model.cfg.spec_augment))

char_model.spec_augmentation = char_model.from_config_dict(char_model.cfg.spec_augment)

print ("############################## SET UP METRIC ##############################")
#@title Metric
use_cer = True #@param ["False", "True"] {type:"raw"}
log_prediction = True #@param ["False", "True"] {type:"raw"}

char_model.wer.use_cer = use_cer
char_model.wer.log_prediction = log_prediction


print ("############################## SET UP TRAINER ##############################")
import torch
import lightning.pytorch as ptl

if torch.cuda.is_available():
  accelerator = 'gpu'
else:
  accelerator = 'cpu'

EPOCHS = 1000  # 100 epochs would provide better results, but would take an hour to train

trainer = ptl.Trainer(devices=1,
                      accelerator=accelerator,
                      max_epochs=EPOCHS,
                      accumulate_grad_batches=1,
                      enable_checkpointing=False,
                      logger=False,
                      log_every_n_steps=5,
                      check_val_every_n_epoch=10)

# Setup model with the trainer
char_model.set_trainer(trainer)

# Finally, update the model's internal config
char_model.cfg = char_model._cfg


# Environment variable generally used for multi-node multi-gpu training.
# In notebook environments, this flag is unnecessary and can cause logs of multiple training runs to overwrite each other.
# os.environ.pop('NEMO_EXPM_VERSION', None)

config = exp_manager.ExpManagerConfig(
    exp_dir=f'experiments/lang-{LANGUAGE}/',
    name=f"ASR-Char-Model-Language-{LANGUAGE}",
    checkpoint_callback_params=exp_manager.CallbackParams(
        monitor="val_wer",
        mode="min",
        always_save_nemo=True,
        save_best_model=True,
    ),
)

config = OmegaConf.structured(config)

logdir = exp_manager.exp_manager(trainer, config)

trainer.fit(char_model)