#!/bin/bash

dataset_name='MELD'
hidden_dim=600
gnn_layers=5
modality='tav'

gpu=0

result_file=${dataset_name}_${modality}_result.txt
rm ${result_file}

for ckpt_index in 1 2 3 4 5
do
    CUDA_VISIBLE_DEVICES=${gpu} python evaluation.py --dataset_name=${dataset_name} --hidden_dim=${hidden_dim} --gnn_layers=${gnn_layers} --modality=${modality} --ckpt_index=${ckpt_index}
done

python average.py --file=${result_file}

