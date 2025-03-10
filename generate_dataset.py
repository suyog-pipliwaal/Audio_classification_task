import h5py
import librosa
import os
import numpy as np
from sklearn import preprocessing
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import joblib

def get_log_mel_spectrogram(file_path, target_duration=3, n_mels=128, hop_length=512, fmax=8000, padding_mode='constant'):
    y, sr = librosa.load(file_path,sr=None)
    target_frames = int(np.ceil(target_duration * sr / hop_length))
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, fmax=fmax)
    num_frames = mel_spec.shape[1]

    if num_frames > target_frames:
        mel_spec = mel_spec[:, :target_frames]

    # Pad if too short
    elif num_frames < target_frames:
        pad_left = (target_frames - num_frames) // 2
        pad_right = target_frames - num_frames - pad_left
        mel_spec = np.pad(mel_spec, ((0, 0), (pad_left, pad_right)), mode=padding_mode)

    # Convert to Log-Mel Spectrogram (dB scale)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    return log_mel_spec
if __name__ == '__main__':
    train_csv                   = './Dataset/Train.csv'
    train_dataset               = pd.read_csv(train_csv)
    label_encoder               = preprocessing.LabelEncoder()
    train_dataset['label']      = label_encoder.fit_transform(train_dataset['Classification'])
    h5_file_path                = "./train.h5"
    log_melspec                 = []
    labels                      = []

    for row in tqdm(train_dataset.itertuples(index=False), total=len(train_dataset)):
        file_path = Path(f'./Dataset/Audio_Files/{row.New}')    
        label = row.label
        log_melspec.append(get_log_mel_spectrogram(file_path))
        labels.append(label)

    log_melspec = np.array(log_melspec, dtype=np.float32)  # (N, Freq_bins, Time_frames)
    labels = np.array(labels, dtype=np.int64)
    with h5py.File(h5_file_path, 'w') as hf:
        hf.create_dataset('log_melspecs', data=log_melspec)
        hf.create_dataset('labels', data=labels)
    joblib.dump(label_encoder, "./label_encoder.pkl")

