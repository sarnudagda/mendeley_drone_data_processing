"""
This script is designed to preprocess the data from the Drone RF database available on Mendeley,
http://dx.doi.org/10.17632/f4c2b4n755.1, and discussed in the paper:

Al-Sa'd, Mohammad; Allahham, Mhd Saria; Mohamed, Amr; Al-Ali, Abdulla; Khattab, Tamer; Erbad, Aiman (2019),
“DroneRF dataset: A dataset of drones for RF-based detection, classification, and identification”,
Mendeley Data, v1. http://dx.doi.org/10.17632/f4c2b4n755.1

"""
from glob import glob
import os

import numpy as np
import pandas as pd
from tqdm import tqdm


m = 2048
l = 1e5
q = 10

root_folder_path = "" # set to the location of the different data types

data_types = ["AR drone", "Background RF activites", "Bepop drone", "Phantom drone"]
for data_type in tqdm(data_types):
    if data_type == "Background RF activites":
        hdata_folders = glob(f'{root_folder_path}/{data_type}/*_H*')
        ldata_folders = glob(f'{root_folder_path}/{data_type}/*_L*')
    else:
        hdata_folders = glob(f'{root_folder_path}/{data_type}/*_H')
        ldata_folders = glob(f'{root_folder_path}/{data_type}/*_L')

    hdata_folders.sort()
    ldata_folders.sort()

    for hdata_folder, ldata_folder in tqdm(zip(hdata_folders, ldata_folders)):
        hdata_files = glob(f'{hdata_folder}/*.csv')
        ldata_files = glob(f'{ldata_folder}/*.csv')

        hdata_files.sort()
        ldata_files.sort()

        for hdatafile, ldatafile in tqdm(zip(hdata_files, ldata_files)):
            data = []
            hdata = np.loadtxt(hdatafile, delimiter=',')
            ldata = np.loadtxt(ldatafile, delimiter=',')
            for ii in range(int(len(hdata)/l)):
                st = int(ii*l)
                fi = int((ii+1)*l)
                xsegment = ldata[st:fi] - np.mean(ldata[st:fi])
                ysegment = hdata[st:fi] - np.mean(hdata[st:fi])
                xf = abs(np.fft.fftshift(np.fft.fft(xsegment, m)))[m//2:]
                yf = abs(np.fft.fftshift(np.fft.fft(ysegment, m)))[m//2:]

                xf_mean = np.mean(xf[-q:])
                yf_mean = np.mean(yf[-q:])
                data.append(np.concatenate([xf, yf * xf_mean/yf_mean]))

            temp = np.square(np.array(data))
            temp = pd.DataFrame(temp)
            if not os.path.exists(f'{root_folder_path}/signals/'):
                os.mkdir(f'{root_folder_path}/signals/')
            if data_type == "Background RF activites":
                temp.to_csv(f"signals/background_fbins{m}_{ldatafile.split("/")[-1].replace("L", "")}")
            elif data_type == "AR drone":
                temp.to_csv(f"signals/ar_fbins{m}_{ldatafile.split("/")[-1].replace("L", "")}")
            elif data_type == "Phantom Drone":
                temp.to_csv(f"signals/phantom_fbins{m}_{ldatafile.split("/")[-1].replace("L", "")}")
            else:
                temp.to_csv(f"signals/bepop_fbins{m}_{ldatafile.split("/")[-1].replace("L", "")}")