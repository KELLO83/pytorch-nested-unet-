import torch
import numpy as np
import glob
import natsort
import os
import cv2
import itertools
from tqdm import tqdm
import math

def compute(file_path):
    reds = []
    blues = []
    greens = []
    
    for i in tqdm(file_path):
        image = cv2.imread(i,cv2.IMREAD_COLOR)
        if image is None:
            raise Exception('Image File not found')
        image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
        
        r ,g ,b = cv2.split(image)
        reds.append(r.flatten())
        blues.append(b.flatten())
        greens.append(g.flatten())
    
    # flatten_reds = [item for sublist in reds for item in sublist]
    # flatten_blues = [item for sublist in blues for item in sublist]
    
    flatten_reds = list(itertools.chain.from_iterable(reds))
    flatten_blues = list(itertools.chain.from_iterable(blues))
    flatten_greens = list(itertools.chain.from_iterable(greens))
    
    r_mean = round(np.mean(flatten_reds) / 255.0 , 3)
    b_mean = round(np.mean(flatten_blues) /255.0 , 3)
    g_mean = round(np.mean(flatten_greens) / 255.0 ,3)
    
    r_std = round(np.std(flatten_reds) /255.0 , 3)
    b_std = round(np.std(flatten_blues) /255.0 , 3)
    g_std = round(np.std(flatten_greens) /255.0 , 3)

    print(f"Mean r :{r_mean} b : {b_mean} g : {g_mean}")
    print(f"StandardDivation : r : {r_std} b: {b_std} g: {g_std} ")

    with open("record.txt" , 'w') as f:
        f.write(f"{r_mean} {b_mean} {g_mean}\n")
        f.write(f"{r_std} {b_std} {g_std}\n")

        
        
if __name__ == '__main__' :
    file_list = natsort.natsorted(glob.glob('inputs/CRACKTREE200_INPUT/images/*.jpg'))
    compute(file_list)
    