"""Simple script for extracting all the archives. 
"""
import os
import os.path as osp
from zipfile import ZipFile
from tqdm import tqdm 

data_folder = osp.join("./data/zenodo/")
for filename in tqdm(sorted(os.listdir(data_folder))):
    fpath = osp.join(data_folder, filename)
    if osp.isdir(fpath):
        continue
    
    basename, ext = osp.splitext(filename)
    tgt_folder = osp.join(data_folder, basename)
    # The file has already been extracted
    if osp.exists(tgt_folder):
        continue
    
    file_size = osp.getsize(fpath)
    # print(f"{filename:.<80}{file_size/1024:.>10.0f}KB")
    os.makedirs(tgt_folder, exist_ok=True) 
    fpath = osp.join(data_folder, filename)
    with ZipFile(fpath) as myzip:
        myzip.extractall(tgt_folder)
        print(fpath)