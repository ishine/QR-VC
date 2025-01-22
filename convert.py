import sys, os

import argparse
import torch
import librosa
import time
from scipy.io.wavfile import write
from tqdm import tqdm
import json

# import torch_hpss
import numpy as np

import utils.utils as utils

# from models.models_v9_wavlm12_40000 import SynthesizerTrn
from models.models_v9_concat_5_40000 import SynthesizerTrn
# from models.models_v8_VQ8192 import SynthesizerTrn
# from models.models_v9_concat import SynthesizerTrn




from utils.mel_processing import mel_spectrogram_torch, spectrogram_torch, spec_to_mel_torch
from wavlm import WavLM, WavLMConfig
import shutil

from evaluation.WER_EER_FakeScore import get_scores

os.environ["CUDA_VISIBLE_DEVICES"]="1"


import logging
logging.getLogger('numba').setLevel(logging.WARNING)

def get_path(*args):
        return os.path.join('', *args)
    
if __name__ == "__main__":
    
    model_name = 'V9_VQ256_concat_5_Libri'
    meta_data = 'LibriTTS_TEST_unseen'
    ckpt_num = 700
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=f"./config/V9_VQ256_concat_5_40000.json", help="path to json config file")
    parser.add_argument("--ptfile", type=str, default=f"./logs/{model_name}/G_{ckpt_num}000.pth", help="path to pth file")
    parser.add_argument("--src_path", type=str, default=f"/home/yjsim/VoiceConversion/ICASSP2025/conversion_metas/{meta_data}_pairs(1000).txt", help="path to txt file")
    parser.add_argument("--tgt_path", type=str, default=f"/home/yjsim/VoiceConversion/ICASSP2025/conversion_metas/{meta_data}_pairs(1000).txt", help="path to txt file")
    parser.add_argument("--outdir", type=str, default=f"./convert_result", help="path to output dir")
    
    parser.add_argument("--use_timestamp", default=False, action="store_true")
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    hps = utils.get_hparams_from_file(args.hpfile)

    print("Loading model...")
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda()
    _ = net_g.eval()
    print("Loading checkpoint...")
    _ = utils.load_checkpoint(args.ptfile, net_g, None, True)

    print("Loading WavLM for content...")
    cmodel = utils.get_cmodel(0)
    
    print("Processing text...")

    print(args.txtpath)
    print(args.outdir)
    print("Synthesizing...")
    with torch.no_grad():

        # src
        wav_src, _ = librosa.load(hps.src_path, sr=hps.data.sampling_rate)
        wav_src = torch.from_numpy(wav_src).unsqueeze(0).cuda()
        src_c = utils.get_content(cmodel, wav_src, layer=6)
        
        wav_tgt, _ = librosa.load(hps.tgt_path, sr=hps.data.sampling_rate)
        wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)
        wav_tgt = torch.from_numpy(wav_tgt).unsqueeze(0).cuda()
        tgt_c = utils.get_content(cmodel, wav_tgt, layer=6)
        

        audio = net_g.convert(src_c, tgt_c)
        audio = audio[0][0].data.cpu().float().numpy()
        
            
        title = 'src;' + hps.src_path.split('/')[-1][:-4] + '&tgt;' + hps.tgt_path.split('/')[-1][:-4]
        save_dir = os.path.join(args.outdir, f"{title}")
        os.makedirs(save_dir, exist_ok=True)
        
        write(os.path.join(save_dir, f"C!{title}.wav"), hps.data.sampling_rate, audio)
        
        shutil.copy2(src, f"{save_dir}/S!{hps.src_path.split('/')[-1]}")
        shutil.copy2(tgt, f"{save_dir}/T!{hps.tgt_path.split('/')[-1]}")
