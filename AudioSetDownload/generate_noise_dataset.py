# -*- coding: utf-8 -*-

import datetime
import os
import numpy as np
import pandas as pd
import soundfile as sf
import subprocess as subp


df_train = pd.read_csv('noiseCSV.csv', header=0)
directory = r'./noiseDataset'
directory_16K = r'./noiseDataset16K'

SR = 16000
counter, maxLength, minLength = 0, 0, 1000000


# initial data dir
def init_dir(path=directory, path_out=directory_16K):
    if not os.path.exists(path):
        os.mkdir(path)

    if not os.path.exists(path_out):
        os.mkdir(path_out)


def download_audios(directory, df_train):
    """
    Using the CSV file provided, use youtube-dl to download only the audios.
    """
    for index, row in df_train.iterrows():
        command = 'cd {};'.format(directory)
        link = "https://www.youtube.com/watch?v=" + row[0]
        start_time = float(row[1])
        end_time = float(row[2])
        start_time = datetime.timedelta(seconds=start_time)
        end_time = datetime.timedelta(seconds=end_time)

        command += 'youtube-dl -f bestaudio --audio-quality 0 --audio-format wav -o "' + str(index) + '.%(ext)s" ' + link + ';'
        command += f'ffmpeg -i {index}.* -ss {start_time} -to {end_time} {index}.wav;'

        command += 'find . -type f -not -name \'*.wav\' -delete'
        os.system(command)


def convert_2_16Khz(directory_in, directory_16K, SR):
    """
    Use FFMPEG 4.3.1 to convert Sampling rate to 16Khz
    """
    for item in (os.listdir(directory_in)):
        input_path = directory + r'/' + item
        wav_path = directory_16K + r'/' + item[:-4] + '.wav'
        cmd = f"ffmpeg -i '{input_path}' -acodec pcm_s16le -ac 1 -ar {SR} {wav_path}"

        subp.run(cmd, shell=True)


def gen_dataset(directory_16K, counter, maxLength, minLength):
    """
    Generate a numpy array with the length of the longest audio.
    Includes zero padding.
    """
    for item in sorted(os.listdir(directory_16K)):
        wav_path = directory + r'/' + item
        data, samplerate = sf.read(wav_path)
        if(len(data) > maxLength):
            print("maxL went from "+str(maxLength)+" to "+str(len(data)))
            print("at index ", item)
            maxLength = len(data)

        if(len(data) < minLength):
            print("minL went from "+str(minLength)+" to "+str(len(data)))
            minLength = len(data)

        counter = counter + 1

    print("Number of Files: "+str(counter))
    print("Max Length of signal is "+str(maxLength))
    print("Min Length of signal is "+str(minLength))

    # DECLARE AND INITIALIZE MATRIX 'X'
    X = np.zeros([counter, maxLength])
    i = 0
    for item in sorted(os.listdir(directory)):
        wav_path = directory + r'/' + item
        raw_data, samplerate = sf.read(wav_path)
        print("{} item {} , shape {}".format(i, item, str(raw_data.shape)))
        if len(raw_data.shape) == 2:
            data = raw_data[:, 0].reshape((1, len(raw_data)))
        else:
            data = raw_data.reshape((1, len(raw_data)))
        X[i][0:data.shape[1]] = data
        if (np.count_nonzero(X[i]) == 0):
            print("All zeros. {} item {}, shape {}".format(i, item, str(data.shape)))
        i = i + 1

    # SAVE GT
    np.save('../Noises_all.npy', X)


if __name__ == '__main__':
    init_dir(path=directory, path_out=directory_16K)
    download_audios(directory, df_train)
    convert_2_16Khz(directory, directory_16K, SR)
    gen_dataset(directory_16K, counter, maxLength, minLength)
