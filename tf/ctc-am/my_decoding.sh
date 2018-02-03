#!/usr/bin/env bash
pkill python
python test.py --data_dir=/home/haoxiang/Desktop/eesen-tf_clean/data/am_data_test --results_dir=/home/haoxiang/Desktop/eesen-tf_clean/am_results --train_config=train-full/model/config.pkl --trained_weights=train-full/model