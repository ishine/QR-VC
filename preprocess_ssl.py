import sys
# sys.path.append('/home/yjsim/VoiceConversion/ICASSP2025')
# sys.path.append('/home/yjsim/VoiceConversion/ICASSP2025/utils')

import os
import argparse
import torch
import librosa
from glob import glob
from tqdm import tqdm

# import utils.utils as utils
import utils

from wavlm import WavLM, WavLMConfig
os.environ["CUDA_VISIBLE_DEVICES"]="0"


def process(filename):
    basename = os.path.basename(filename)
    speaker = basename[:4]
    save_dir = os.path.join(args.out_dir, speaker)
    os.makedirs(save_dir, exist_ok=True)
    wav, _ = librosa.load(filename, sr=args.sr)
    wav = torch.from_numpy(wav).unsqueeze(0).cuda()
    # c = utils.get_content(cmodel, wav)
    c = utils.get_content(cmodel, wav, layer=6)
    save_name = os.path.join(save_dir, basename.replace(".wav", ".pt"))
    torch.save(c.cpu(), save_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr", type=int, default=16000, help="sampling rate")
    # parser.add_argument("--in_dir", type=str, default="/shared/racoon_fast/sim/VCTK/preprocessed/vctk-16k_no_trim", help="path to input dir")
    # parser.add_argument("--out_dir", type=str, default="/shared/racoon_fast/sim/VCTK/preprocessed/wavlm-6L_no_trim", help="path to output dir")
    parser.add_argument("--in_dir", type=str, default="/shared/NAS_HDD/VC/Dataset/LibriTTS/preprocessed/LibriTTS-360-16k_train_no_trim", help="path to input dir")
    parser.add_argument("--out_dir", type=str, default="/shared/NAS_HDD/VC/Dataset/LibriTTS/preprocessed/wavlm-360-6L_train_no_trim", help="path to output dir")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading WavLM for content...")
    checkpoint = torch.load('/home/yjsim/VoiceConversion/ICASSP2025/wavlm/WavLM-Large.pt')
    cfg = WavLMConfig(checkpoint['cfg'])
    cmodel = WavLM(cfg).cuda()
    cmodel.load_state_dict(checkpoint['model'])
    cmodel.eval()
    print("Loaded WavLM.")
    
    filenames = glob(f'{args.in_dir}/*/*.wav', recursive=True)
    
    for filename in tqdm(filenames):
        process(filename)
    