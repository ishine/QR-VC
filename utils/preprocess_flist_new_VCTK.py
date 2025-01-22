import os
import argparse
from tqdm import tqdm
from random import shuffle


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_list", type=str, default="./filelists_un/train.txt", help="path to train list")
    parser.add_argument("--val_list", type=str, default="./filelists_un/val.txt", help="path to val list")
    parser.add_argument("--test_list", type=str, default="./filelists_un/test.txt", help="path to test list")
    parser.add_argument("--unseen_list", type=str, default="./filelists_un/unseen.txt", help="path to test list")
    
    parser.add_argument("--source_dir", type=str, default="/shared/NAS_HDD/VC/Dataset/VCTK/vctk-16k_no_trim", help="path to source dir")
    args = parser.parse_args()
    
    train = []
    val = []
    test = []
    unseen = []
    idx = 0
    
    unseen_spk = ['p226', 'p256', 'p266', 'p297', 'p323', 'p376']
    for speaker in tqdm(os.listdir(args.source_dir)):
        wavs = os.listdir(os.path.join(args.source_dir, speaker))
        wavs = [file for file in wavs if '.wav' in file]
        shuffle(wavs)
        
        # if speaker in unseen_spk:
        #     unseen += wavs
        train += wavs[2:-10]
        val += wavs[:2]
        test += wavs[-10:]
        
    shuffle(train)
    shuffle(val)
    shuffle(test)
    shuffle(unseen)
    
    print("Writing", args.train_list)
    with open(args.train_list, "w") as f:
        for fname in tqdm(train):
            speaker = fname[:4]
            wavpath = os.path.join("DUMMY", speaker, fname)
            f.write(wavpath + "\n")
        
    print("Writing", args.val_list)
    with open(args.val_list, "w") as f:
        for fname in tqdm(val):
            speaker = fname[:4]
            wavpath = os.path.join("DUMMY", speaker, fname)
            f.write(wavpath + "\n")
            
    print("Writing", args.test_list)
    with open(args.test_list, "w") as f:
        for fname in tqdm(test):
            speaker = fname[:4]
            wavpath = os.path.join("DUMMY", speaker, fname)
            f.write(wavpath + "\n")
            
    print("Writing", args.unseen_list)
    with open(args.unseen_list, "w") as f:
        for fname in tqdm(unseen):
            speaker = fname[:4]
            wavpath = os.path.join("DUMMY", speaker, fname)
            f.write(wavpath + "\n")
            