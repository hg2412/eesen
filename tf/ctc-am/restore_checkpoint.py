import tensorflow as tf
import pickle
from multiprocessing import Process, Queue
import tensorflow as tf
import constants
import random
from train import *
from models.model_factory import create_model
import constants
from worker import *




# load args
# with open("args.pickle", "rb") as f:
#     args = pickle.load(f)
#     #args.data_dir = "/home/haoxiang/Desktop/eesen-tf_clean/data/am_data_small"
#     args.data_dir = "/home/haoxiang/Desktop/eesen-tf_clean/data/am_data_small"
#     args.train_dir = "test_restore"
#
# _, config = prepare(args)
# create_model(config)

sv = tf.train.Supervisor(logdir="model_copy")
with sv.managed_session() as sess:
    pass