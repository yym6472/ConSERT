#!/bin/bash

data_path=./chinese

# Downloads atec_ccks, bq, lcqmc
git clone https://github.com/IceFlameWorm/NLP_Datasets.git

# Downloads pawsx
wget https://storage.googleapis.com/paws/pawsx/x-final.tar.gz
tar xvzf x-final.tar.gz

# Downloads chinese stsb
wget https://6a75-junzeng-uxxxm-1300734931.tcb.qcloud.la/STS-B.rar
unrar x STS-B.rar

# Pre-processing
python prepare_chinese_data.py $data_path

# Clean useless files
/bin/rm -rf ./NLP_Datasets
/bin/rm -rf ./STS-B
/bin/rm -rf ./x-final
/bin/rm STS-B.rar
/bin/rm x-final.tar.gz