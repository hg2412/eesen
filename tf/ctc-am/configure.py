import os
import json
import math
from tensorflow.python.client import device_lib

def get_input():
    data_dir = raw_input("Data dir?\n")
    train_dir = raw_input("Train dir (dir to store checkpoints and logs)?\n")
    config = {}
    config["data_dir"] = data_dir
    config["train_dir"] = train_dir

    print("Configure cluster:")
    num_ps = int(raw_input("How many parameter servers:\n"))
    parameter_servers = []
    for i in range(num_ps):
        parameter_servers.append(raw_input("Enter the server address:\n"))

    num_wk = int(raw_input("How many workers:"))
    workers = []
    for i in range(num_wk):
        workers.append(raw_input("Enter the server address:\n"))

    cluster_spec = {"ps": parameter_servers, "worker": workers}
    return config, cluster_spec


def parse_device_name(s):
    return s.split(":")[-1]


def get_GPU_assignment(cluster_spec):
    devices = device_lib.list_local_devices()
    gpus = [x.name for x in devices if x.device_type == 'GPU']
    procs = list(cluster_spec["ps"])
    procs.extend(cluster_spec["worker"])
    # allocate to gpu processes
    processes_per_gpu = int(math.ceil(len(procs) * 1.0 / len(gpus)))
    mem_frac = 0.8 / processes_per_gpu
    res = {}
    idx = 0
    count = 0
    for p in procs:
        res[p] = [parse_device_name(gpus[idx]), mem_frac]
        count += 1
        if count >= processes_per_gpu:
            count = 0
            idx += 1
    return res


def main():
    config, cluster_spec = get_input()
    # copy labels
    os.system("mkdir " + str(config["data_dir"]))
    # generate json configs
    print("Config saved!")
    with open('config.json', 'w') as fp:
        json.dump(config, fp)

    print("Cluster spec saved!")
    with open('cluster_spec.json', 'w') as fp:
        json.dump(cluster_spec, fp)

    # allocate GPU to processes
    print("Calculate GPU allocation:\n")
    print(device_lib.list_local_devices())
    gpu_alloc_config = get_GPU_assignment(cluster_spec)
    print("GPU allocation config saved!")
    with open('gpu_alloc_config.json', 'w') as fp:
        json.dump(gpu_alloc_config, fp)

    return


main()
