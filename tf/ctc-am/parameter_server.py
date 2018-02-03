from multiprocessing import Process
import tensorflow as tf


class ParameterServer(object):
    def __init__(self, cluster_spec, config, index=0, task="train", gpu_memory_fraction=0.3, device_list="0"):
        self.cluster_spec = dict(cluster_spec)
        self.config = None
        self.index = index
        self.task = task
        self.gpu_memory_fraction = gpu_memory_fraction
        self.process = None
        self.device_list = device_list

    def run(self):
        p = Process(target=self.start)
        p.start()
        self.process = p
        print("ParameterServer %d started!\npid:%d" % (self.index, p.pid))

    def start(self):
        cluster = tf.train.ClusterSpec(self.cluster_spec)
        # config to limit GPU memory allocation when on single machine
        sessConfig = tf.ConfigProto(log_device_placement=False, device_count={'GPU':1})
        sessConfig.gpu_options.visible_device_list = self.device_list
        sessConfig.gpu_options.per_process_gpu_memory_fraction = self.gpu_memory_fraction

        server = tf.train.Server(cluster,
                                 job_name="ps",
                                 task_index=self.index,
                                 config=sessConfig)
        server.join()
        return



