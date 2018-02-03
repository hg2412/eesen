import argparse
import os
import json
import math
import pickle
from worker import Worker
from reader.reader_queue import run_reader_queue
from parameter_server import ParameterServer
import sys
import eesen

import constants
import tensorflow as tf

import shutil
from utils.fileutils import debug
from reader.feats_reader import feats_reader_factory
from reader.labels_reader import labels_reader_factory
from reader.sat_reader import sat_reader_factory

from train import *


def start_servers(cluster_spec1, gpu_config, args):
    workers = []
    pss = []
    for i in range(len(cluster_spec1["worker"])):
        workers.append(Worker(cluster_spec1, args, index=i, gpu_memory_fraction=gpu_config[cluster_spec1["worker"][i]][1], device_list=gpu_config[cluster_spec1["worker"][i]][0]))
        # workers.append(Worker(cluster_spec1, args, index=i))

    for i in range(len(cluster_spec1["ps"])):
        pss.append(ParameterServer(cluster_spec1, args, index=i, gpu_memory_fraction=gpu_config[cluster_spec1["ps"][i]][1], device_list=gpu_config[cluster_spec1["ps"][i]][0]))
        # pss.append(ParameterServer(cluster_spec1, args, index=i))

    for i in range(len(workers)):
        workers[i].run()

    for i in range(len(pss)):
        pss[i].run()

    for i in range(len(pss)):
        pss[i].process.join()

    for i in range(len(workers)):
        workers[i].process.join()
    return


def load_configs(config_path, cluster_spec_path, gpu_config_path):
    with open(config_path) as f:
        config = json.load(f)
    with open(cluster_spec_path) as f:
        cluster_spec = json.load(f)
    with open(gpu_config_path) as f:
        gpu_config = json.load(f)
    return config, cluster_spec, gpu_config

config, cluster_spec, gpu_config = load_configs("config.json", "cluster_spec.json", "gpu_alloc_config.json")

# load args
with open("args.pickle", "rb") as f:
    args = pickle.load(f)
    #args.data_dir = "/home/haoxiang/Desktop/eesen-tf_clean/data/am_data_small"
    args.data_dir = config["data_dir"]
    args.train_dir = config["train_dir"]
    print("Train dir")
    print( args.train_dir)
    args.start_epoch = 3
    args.half_after = 4

print(args)

start_servers(cluster_spec, gpu_config, args)
