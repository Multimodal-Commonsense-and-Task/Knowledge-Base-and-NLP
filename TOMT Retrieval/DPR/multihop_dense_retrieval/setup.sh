#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.

conda install --file requirements.txt
# pip install pytrec_eval
# conda install pytorch cudatoolkit=10.2 -c pytorch
#conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install -c conda-forge faiss-gpu

git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir ./

#python setup.py develop