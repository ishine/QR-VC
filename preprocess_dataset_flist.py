import os
import argparse
from tqdm import tqdm
from random import shuffle


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_list", type=str, default="../filelists_LibriTTS/train.txt", help="path to train list")
    parser.add_argument("--val_list", type=str, default="../filelists_LibriTTS/val.txt", help="path to val list")
    parser.add_argument("--test_list", type=str, default="../filelists_LibriTTS/test.txt", help="path to test list")
    # parser.add_argument("--unseen_list", type=str, default="./filelists_un/unseen.txt", help="path to test list")
    
    parser.add_argument("--source_dir_train", type=str, default="/shared/NAS_HDD/VC/Dataset/LibriTTS/preprocessed/LibriTTS-360-16k_train_no_trim", help="path to source dir")
    parser.add_argument("--source_dir_test", type=str, default="/shared/NAS_HDD/VC/Dataset/LibriTTS/preprocessed/LibriTTS-16k_test_no_trim", help="path to source dir")
    args = parser.parse_args()
    
    train = []
    val = []
    test = []
    # unseen = []
    idx = 0
    
    file_paths =[]
    for speaker in tqdm(os.listdir(args.source_dir_train)):
        wavs = os.listdir(os.path.join(args.source_dir_train, speaker))
        wavs = [file for file in wavs if '.wav' in file]
        shuffle(wavs)
        file_paths += wavs
        # if speaker in unseen_spk:
        #     unseen += wavs
    shuffle(file_paths)
    train_index = int(len(file_paths) * 0.95)
    val_index = int(len(file_paths) * 0.05)

    train += file_paths[:train_index]
    val += file_paths[train_index:train_index + val_index]
    
    
    file_paths =[]
    for speaker in tqdm(os.listdir(args.source_dir_test)):
        wavs = os.listdir(os.path.join(args.source_dir_test, speaker))
        wavs = [file for file in wavs if '.wav' in file]
        shuffle(wavs)
        file_paths += wavs

    shuffle(file_paths) 
    test = file_paths
        
    shuffle(train)
    shuffle(val)
    shuffle(test)
    # shuffle(unseen)
    
    print("Writing", args.train_list)
    with open(args.train_list, "w") as f:
        for fname in tqdm(train):
            speaker = fname.split('_')[0]
            wavpath = os.path.join(args.source_dir_train, speaker, fname)
            f.write(wavpath + "\n")
        
    print("Writing", args.val_list)
    with open(args.val_list, "w") as f:
        for fname in tqdm(val):
            speaker = fname.split('_')[0]
            wavpath = os.path.join(args.source_dir_train, speaker, fname)
            f.write(wavpath + "\n")
            
    print("Writing", args.test_list)
    with open(args.test_list, "w") as f:
        for fname in tqdm(test):
            speaker = fname.split('_')[0]
            wavpath = os.path.join(args.source_dir_test, speaker, fname)
            f.write(wavpath + "\n")
            
    # print("Writing", args.unseen_list)
    # with open(args.unseen_list, "w") as f:
    #     for fname in tqdm(unseen):
    #         speaker = fname[:4]
    #         wavpath = os.path.join("DUMMY", speaker, fname)
    #         f.write(wavpath + "\n")
            