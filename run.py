import logging

numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)

import librosa

import matplotlib.pyplot as plt
import IPython.display as ipd

import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text.cleaners import japanese_phrase_cleaners
from text import cleaned_text_to_sequence

def get_text(text, hps):
    text_norm = cleaned_text_to_sequence(text)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm
# hps_ms = utils.get_hparams_from_file("./configs/vctk_base.json")
hps = utils.get_hparams_from_file("./configs/nan.json")
# net_g_ms = SynthesizerTrn(
#     len(symbols),
#     hps_ms.data.filter_length // 2 + 1,
#     hps_ms.train.segment_size // hps.data.hop_length,
#     n_speakers=hps_ms.data.n_speakers,
#     **hps_ms.model)

net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model)
_ = net_g.eval()

def tts(text):
  stn_tst = get_text(text, hps)
  with torch.no_grad():
    x_tst = stn_tst.unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
    audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.float().numpy()
  ipd.display(ipd.Audio(audio, rate=hps.data.sampling_rate))
def jtts(text):
  stn_tst = get_text(japanese_phrase_cleaners(text), hps)
  with torch.no_grad():
    x_tst = stn_tst.unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
    audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.float().numpy()
  ipd.display(ipd.Audio(audio, rate=hps.data.sampling_rate))
