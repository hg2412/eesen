from multiprocessing import Process, Queue
import tensorflow as tf
import constants
import random
from train import *
from models.model_factory import create_model
import constants
import tensorflow as tf
import time

from reader.feats_reader import feats_reader_factory
from reader.labels_reader import labels_reader_factory
from reader.sat_reader import sat_reader_factory

import sys


# prepare configs and data of worker from args
def prepare(args):
    if (args.import_config):
        config = create_global_config(args)
        config.update(import_config(args))
    else:
        config = create_global_config(args)

    print(80 * "-")
    print("reading training set")
    print(80 * "-")
    print(80 * "-")
    print("tr_x:")
    print(80 * "-")
    # load training feats
    if config[constants.CONF_TAGS.MODEL] == constants.MODEL_NAME.ARCNET_VIDEO:
        tr_x = feats_reader_factory.create_reader('train', 'video', config)

    else:
        tr_x = feats_reader_factory.create_reader('train', 'kaldi', config)


    print("# of batches!!!!!")
    print(tr_x.get_num_batches())

    print(80 * "-")
    print("tr_y:")
    print(80 * "-")
    # load training targets
    if args.import_config:
        print("creating tr_y according imported config...")
        tr_y = labels_reader_factory.create_reader('train', 'txt', config, tr_x.get_batches_id(),
                                                   config[constants.CONF_TAGS.LANGUAGE_SCHEME])
    else:
        print("creating tr_y from scratch...")
        tr_y = labels_reader_factory.create_reader('train', 'txt', config, tr_x.get_batches_id())

    print(80 * "-")
    print("cv_x:")
    print(80 * "-")
    # create lm_reader for labels
    if config[constants.CONF_TAGS.MODEL] == constants.MODEL_NAME.ARCNET_VIDEO:
        cv_x = feats_reader_factory.create_reader('cv', 'video', config)
    else:
        cv_x = feats_reader_factory.create_reader('cv', 'kaldi', config)

    print(80 * "-")
    print("cv_y:")
    print(80 * "-")
    # create lm_reader for labels
    if args.import_config:
        print("creating cv_y according imported config...")
        cv_y = labels_reader_factory.create_reader('cv', 'txt', config, cv_x.get_batches_id(),
                                                   config[constants.CONF_TAGS.LANGUAGE_SCHEME])
    else:
        print("creating cv_y from scratch...")
        cv_y = labels_reader_factory.create_reader('cv', 'txt', config, cv_x.get_batches_id())

    # set config (targets could change)
    config[constants.CONF_TAGS.INPUT_FEATS_DIM] = cv_x.get_num_dim()
    config[constants.CONF_TAGS.LANGUAGE_SCHEME] = cv_y.get_language_scheme()

    if config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_TYPE] != constants.SAT_TYPE.UNADAPTED:

        print(80 * "-")
        print(80 * "-")
        print("reading speaker adaptation set:")
        print(80 * "-")
        print(80 * "-")

        print("tr_sat:")
        print(80 * "-")
        tr_sat = sat_reader_factory.create_reader('kaldi', config, tr_x.get_batches_id())
        print(80 * "-")

        print("cv_sat:")
        print(80 * "-")
        cv_sat = sat_reader_factory.create_reader('kaldi', config, cv_x.get_batches_id())
        print(80 * "-")
        print(80 * "-")

        config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_FEAT_DIM] = int(tr_sat.get_num_dim())
        config[constants.CONF_TAGS.MODEL_DIR] = os.path.join(config[constants.CONF_TAGS.TRAIN_DIR],
                                                             constants.DEFAULT_NAMES.MODEL_DIR_NAME,
                                                             constants.DEFAULT_NAMES.SAT_DIR_NAME + "_" +
                                                             config[constants.CONF_TAGS.SAT_CONF][
                                                                 constants.CONF_TAGS.SAT_TYPE] + "_" +
                                                             config[constants.CONF_TAGS.SAT_CONF][
                                                                 constants.CONF_TAGS.SAT_SATGE] + "_" +
                                                             str(config[constants.CONF_TAGS.SAT_CONF][
                                                                     constants.CONF_TAGS.NUM_SAT_LAYERS]))

        # checking that all sets are consistent
        set_checkers.check_sets_training(cv_x, cv_y, tr_x, tr_y, tr_sat, cv_sat)

        data = (cv_x, tr_x, cv_y, tr_y, cv_sat, tr_sat)

        print("adaptation data with a dimensionality of "
              + str(config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_FEAT_DIM]) +
              " prepared...\n")

    else:
        data = (cv_x, tr_x, cv_y, tr_y)
        config[constants.CONF_TAGS.MODEL_DIR] = os.path.join(config[constants.CONF_TAGS.TRAIN_DIR],
                                                             constants.DEFAULT_NAMES.MODEL_DIR_NAME)
        # checking that all sets are consistent
        set_checkers.check_sets_training(cv_x, cv_y, tr_x, tr_y)
    return data, config


def __print_counts_debug(epoch, batch_counter, total_number_batches, batch_cost, batch_size, batch_ters,
                         data_queue):
    print(
        "epoch={} batch={}/{} size={} batch_cost={}".format(epoch, batch_counter, total_number_batches, batch_size,
                                                            batch_cost))
    print("batch ", batch_counter, " of ", total_number_batches, "size ", batch_size,
          "queue ", data_queue.empty(), data_queue.full(), data_queue.qsize())

    print("ters: ")
    print(batch_ters)



def run_reader_queue_forever(queue, reader_x, reader_y, do_shuf, is_debug, reader_sat=None):
    idx_shuf = list(range(reader_x.get_num_batches()))
    cur = 0
    while True:
        print("Feed data")
        while not queue.full():
            if cur == 0:
                random.shuffle(idx_shuf)
            x = reader_x.read(cur)
            y = reader_y.read(cur)
            queue.put((x, y))
            cur = (cur + 1) % reader_x.get_num_batches()
        print("Queue is full")
        time.sleep(5)
    return


class Worker(object):
    def __init__(self, cluster_spec, args, index=0, task="train", gpu_memory_fraction=0.3, device_list="0"):
        self.cluster_spec = dict(cluster_spec)
        self.data, self.config = prepare(args)
        self.__config = self.config
        self.index = index
        self.is_chief = (index == 0)
        self.task = task
        self.start_epoch = args.start_epoch
        self.gpu_memory_fraction = gpu_memory_fraction
        self.__model = None
        self.train_data_queue_process = None
        self.max_targets_layers = 0
        self.device_list = device_list
        self.__ter_buffer = [float('inf'), float('inf')]

        if self.is_chief:
            #create folder for storing experiment
            if not os.path.exists(self.config[constants.CONF_TAGS.MODEL_DIR]):
                os.makedirs(self.config[constants.CONF_TAGS.MODEL_DIR])
            pickle.dump(self.config, open(os.path.join(self.config[constants.CONF_TAGS.MODEL_DIR], "config.pkl"), "wb"))
        return

    def run(self):
        if self.task == "train":
            self.__prepare_data_queue()
            p = Process(target=self.train)
            p.start()
            self.process = p
            print("Worker%d started!\npid:%d" % (self.index, p.pid))

        else:
            p = Process(target=self.test)
            p.start()
            print(p.pid)
        return

    def build_graph(self):
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.__model = create_model(self.config)
        cost = self.__model.debug_costs[-1][-1]
        ter = self.__model.ters[-1][-1]
        self.increment_global_step_op = tf.assign(self.global_step, self.global_step + 1)
        print("Build graph!!!!")
        print(ter)
        print(cost)
        print( self.__model.ters)
        print( self.__model.costs)
        tf.summary.scalar('cost',cost)
        tf.summary.scalar('ter', ter)
        self.__summary_op = tf.summary.merge_all()
        return


    def train(self):
        cluster = tf.train.ClusterSpec(self.cluster_spec)
        # config to limit GPU memory allocation when on single machine
        sessConfig = tf.ConfigProto(log_device_placement=False)
        sessConfig.gpu_options.visible_device_list = self.device_list
        sessConfig.gpu_options.per_process_gpu_memory_fraction = self.gpu_memory_fraction

        server = tf.train.Server(cluster,
                                 job_name="worker",
                                 task_index=self.index,
                                 config=sessConfig)

        # tf.set_random_seed(self.__config[constants.CONF_TAGS.RANDOM_SEED])
        # random.seed(self.__config[constants.CONF_TAGS.RANDOM_SEED])

        # construct the __model acoring to __config
        if self.__config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_TYPE] != constants.SAT_TYPE.UNADAPTED:
            cv_x, tr_x, cv_y, tr_y, cv_sat, tr_sat = self.data
        else:
            cv_x, tr_x, cv_y, tr_y = self.data
            tr_sat = None
            cv_sat = None

        if not os.path.exists(self.__config[constants.CONF_TAGS.MODEL_DIR]):
            os.makedirs(self.__config[constants.CONF_TAGS.MODEL_DIR])

        # build graph
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % self.index,
                cluster=cluster)):
            self.build_graph()

        init_op = tf.global_variables_initializer()
        sv = tf.train.Supervisor(is_chief=self.is_chief, init_op=init_op, save_model_secs=60, global_step=self.global_step,
                                 logdir=self.config[constants.CONF_TAGS.MODEL_DIR])
        print("Model DIR!!!!!")
        print(self.config[constants.CONF_TAGS.MODEL_DIR])
        with sv.prepare_or_wait_for_session(server.target) as sess:
            # print(tf.get_default_graph().get_all_collection_keys())
            # all_vars = tf.get_collection("model_variables")
            # print(all_vars)
            # for v in all_vars:
            #     print(v)
            #     v_ = sess.run(v)
            #     print(v_)
            self.__writer = tf.summary.FileWriter(self.config[constants.CONF_TAGS.MODEL_DIR],
                                                  sess.graph)

            # initialize
            lr_rate = self.__config[constants.CONF_TAGS.LR_RATE]
            best_epoch = 0
            best_avg_ters = float('Inf')

            for epoch in range(self.start_epoch, self.config[constants.CONF_TAGS.NEPOCH]):
                # log start
                print(80 * "-")
                print("Epoch " + str(epoch) + " starting ... ( lr_rate: " + str(lr_rate) + ")")
                print(80 * "-")
                # start timer...
                tic = time.time()

                if self.is_chief:
                    sv.saver.save(sess, sv.save_path, global_step=self.global_step)
                # training...
                train_cost, train_ters, ntrain = self.__train_epoch(sess, epoch, lr_rate, tr_x, tr_y, tr_sat)

                print("Save the model")

                cv_cost, cv_ters, ncv = self.__eval_epoch(sess, cv_x, cv_y, cv_sat)

                print("Train Cost: \n" + str(train_cost))
                print("Train TERs: \n" + str(train_ters))
                if self.is_chief:
                    self.__generate_logs(cv_ters, cv_cost, ncv, train_ters, train_cost, ntrain, epoch, lr_rate, tic)
                # update lr_rate if needed
                lr_rate, best_avg_ters, best_epoch = self.__update_lr_rate(epoch, cv_ters, best_avg_ters,
                                                                           best_epoch, None, lr_rate)
                print("CV Cost: \n" + str(cv_cost))
                print("CV TERs: \n" + str(cv_ters))

                print("Epoch " + str(epoch) + " done.")
                print(80 * "-")


        sv.stop()
        return

    def __prepare_data_queue(self):
        if self.__config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_TYPE] != constants.SAT_TYPE.UNADAPTED:
            cv_x, tr_x, cv_y, tr_y, cv_sat, tr_sat = self.data
        else:
            cv_x, tr_x, cv_y, tr_y = self.data
            tr_sat = None
            cv_sat = None

        queue_size = self.__config[constants.CONF_TAGS.BATCH_SIZE] * 10
        self.__train_data_queue = Queue(queue_size)
        self.__cv_data_queue = Queue(queue_size)

        if self.__config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_TYPE] \
                != constants.SAT_TYPE.UNADAPTED:
            p = Process(target=run_reader_queue_forever, args=(
            self.__train_data_queue, tr_x, tr_y, self.__config["do_shuf"], False, tr_sat))
            # run_reader_queue(data_queue, tr_x, tr_y, self.__config["do_shuf"], False, tr_sat)
        else:
            # run_reader_queue(data_queue, tr_x, tr_y, self.__config["do_shuf"], False, tr_sat)
            p = Process(target=run_reader_queue_forever, args=(
            self.__train_data_queue, tr_x, tr_y, self.__config["do_shuf"], False))
        self.train_data_queue_process = p
        p.start()
        print("Train Data Queue Process: " + str(p.pid))

        if self.__config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_TYPE] \
                != constants.SAT_TYPE.UNADAPTED:
            q = Process(target=run_reader_queue_forever, args=(
                self.__cv_data_queue, cv_x, cv_y, self.__config["do_shuf"], False, cv_sat))
            # run_reader_queue(data_queue, tr_x, tr_y, self.__config["do_shuf"], False, tr_sat)
        else:
            # run_reader_queue(data_queue, tr_x, tr_y, self.__config["do_shuf"], False, tr_sat)
            q = Process(target=run_reader_queue_forever, args=(
                self.__cv_data_queue, cv_x, cv_y, self.__config["do_shuf"], False))
        self.cv_data_queue_process = q
        q.start()
        print("CV Data Queue Process: " + str(q.pid))
        return

    def __train_epoch(self, sess, epoch, lr_rate, tr_x, tr_y, tr_sat):
        # initializing samples, steps and cost counters
        batch_counter = 0

        # initializinzing dictionaries that will count
        train_ters, ntr_labels, ntrain, train_cost = {}, {}, {}, {}

        # TODO change all iteritems for iter for python 3.0
        # TODO try to do an lm_utils for this kind of functions
        for language_id, target_scheme in self.__config[constants.CONF_TAGS.LANGUAGE_SCHEME].items():

            ntr_labels[language_id] = {}
            train_ters[language_id] = {}
            train_cost[language_id] = {}

            ntrain[language_id] = 0

            for target_id, _ in target_scheme.items():
                ntr_labels[language_id][target_id] = 0
                train_ters[language_id][target_id] = 0
                train_cost[language_id][target_id] = 0

        #
        # # start queue ...


        #
        # training starting...
        for i in xrange(tr_x.get_num_batches()):
            print("training batch " + str(i))
            # pop from queue
            data = self.__train_data_queue.get()
            # finish if there no more batches

            feed, batch_size, index_correct_lan = self.__prepare_feed(data, lr_rate)
            batch_cost, batch_ters, _, summary, _, step = sess.run([self.__model.debug_costs[index_correct_lan],
                                                  self.__model.ters[index_correct_lan],
                                                  self.__model.opt[index_correct_lan],
                                                  self.__summary_op,
                                                  self.increment_global_step_op,
                                                  self.global_step,
                                                  ],
                                                 feed)
            print(batch_ters)
            print(batch_cost)
            print(self.__model.ters)
            print(self.__model.costs)

            if self.is_chief:
                self.__writer.add_summary(summary, step)
            # updating values...
            self.__update_counters(train_ters, train_cost, ntrain, ntr_labels, batch_ters, batch_cost, batch_size,
                                   data[1])

            # # print if in debug mode
            if self.__config[constants.CONF_TAGS.DEBUG]:
                self.__print_counts_debug(epoch, batch_counter, tr_x.get_num_batches(), batch_cost, batch_size,
                                          batch_ters, self.__train_data_queue)
                batch_counter += 1

        # averaging counters
        for language_id, target_scheme in self.__config[constants.CONF_TAGS.LANGUAGE_SCHEME].items():
            for target_id, _ in target_scheme.items():
                if ntrain[language_id] != 0:
                    train_cost[language_id][target_id] /= float(ntrain[language_id])

        for language_id, target_scheme in train_ters.items():
            for target_id, train_ter in target_scheme.items():
                if ntr_labels[language_id][target_id] != 0:
                    train_ters[language_id][target_id] = train_ter / float(ntr_labels[language_id][target_id])

        return train_cost, train_ters, ntrain

    def __eval_epoch(self, sess, cv_x, cv_y, cv_sat):
        # init data_queue
        data_queue = self.__cv_data_queue
        # initializing counters and dicts
        ncv_labels, cv_ters, cv_cost, ncv = {}, {}, {}, {}

        for language_id, target_scheme in self.__config[constants.CONF_TAGS.LANGUAGE_SCHEME].items():
            ncv_labels[language_id] = {}
            cv_ters[language_id] = {}
            cv_cost[language_id] = {}

            ncv[language_id] = 0

            for target_id, _ in target_scheme.items():
                ncv_labels[language_id][target_id] = 0
                cv_ters[language_id][target_id] = 0
                cv_cost[language_id][target_id] = 0

        count = 0
        #
        for i in xrange(cv_x.get_num_batches()):
            print("cv batch " + str(i))
            # get batch
            data = data_queue.get()
            count += 1
            # getting the feed..
            feed, batch_size, index_correct_lan = self.__prepare_feed(data)
            # processing a batch...
            batch_cost, batch_ters, = sess.run(
                [self.__model.debug_costs[index_correct_lan], self.__model.ters[index_correct_lan]], feed)
            # updating values...
            self.__update_counters(cv_ters, cv_cost, ncv, ncv_labels, batch_ters, batch_cost, batch_size, data[1])

        # averaging counters
        for language_id, target_scheme in self.__config[constants.CONF_TAGS.LANGUAGE_SCHEME].items():
            for target_id, _ in target_scheme.items():
                if ncv[language_id] != 0:
                    cv_cost[language_id][target_id] /= float(ncv[language_id])

        for language_id, target_scheme in cv_ters.items():
            for target_id, cv_ter in target_scheme.items():
                if (ncv_labels[language_id][target_id] != 0):
                    cv_ters[language_id][target_id] = cv_ter / float(ncv_labels[language_id][target_id])
        return cv_cost, cv_ters, ncv

    def __prepare_feed(self, data, lr_rate=None):

        if self.__config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_TYPE] \
                != constants.SAT_TYPE.UNADAPTED:
            x_batch, y_batch, sat_batch = data
        else:
            x_batch, y_batch = data

        if self.__config[constants.CONF_TAGS.DEBUG]:
            print("")
            print("the following batch_id is prepared to be processed...")
            print(x_batch[1])
            print("size batch x:")
            for element in x_batch[0]:
                print(element.shape)

            print("sizes batch y:")
            for language_id, target in y_batch[0].items():
                for target_id, content in target.items():
                    print(content[2])

            print("")

        # it contains the actual value of x
        x_batch = x_batch[0]

        batch_size = len(x_batch)

        current_lan_index = 0
        for language_id, language_scheme in self.__config[constants.CONF_TAGS.LANGUAGE_SCHEME].items():
            if language_id == y_batch[1]:
                index_correct_lan = current_lan_index
            current_lan_index += 1

        y_batch_list = []
        for _, value in y_batch[0].items():
            for _, value in value.items():
                y_batch_list.append(value)

        if len(y_batch_list) < self.max_targets_layers:
            for count in range(self.max_targets_layers - len(y_batch_list)):
                y_batch_list.append(y_batch_list[0])

        # eventhough self.__model.labels will be equal or grater we will use only until_batch_list
        feed = {i: y for i, y in zip(self.__model.labels, y_batch_list)}

        # TODO remove this prelimenary approaches
        feed[self.__model.feats] = x_batch

        # it is training
        if lr_rate:
            feed[self.__model.lr_rate] = lr_rate
            feed[self.__model.is_training_ph] = True
        else:
            feed[self.__model.is_training_ph] = False

        if self.__config[constants.CONF_TAGS.SAT_CONF][constants.CONF_TAGS.SAT_TYPE] \
                != constants.SAT_TYPE.UNADAPTED:
            feed[self.__model.sat] = sat_batch

        return feed, batch_size, index_correct_lan

    def __update_counters(self, m_acum_ters, m_acum_cost, m_acum_samples, m_acum_labels,
                          batch_ters, batch_cost, batch_size, ybatch):

        # https://stackoverflow.com/questions/835092/python-dictionary-are-keys-and-values-always-the-same-order
        # TODO although this should be changed for now is a workaround

        for idx_lan, (language_id, target_scheme) in enumerate(
                self.__config[constants.CONF_TAGS.LANGUAGE_SCHEME].items()):
            if (ybatch[1] == language_id):
                for idx_tar, (target_id, _) in enumerate(target_scheme.items()):

                    # note that ybatch[0] contains targets and ybathc[1] contains language_id
                    m_acum_ters[language_id][target_id] += batch_ters[idx_tar]
                    m_acum_labels[language_id][target_id] += self.__get_label_len(ybatch[0][language_id][target_id])
                    if batch_cost[idx_tar] != float('Inf'):
                        m_acum_cost[language_id][target_id] += batch_cost[idx_tar] * batch_size

        m_acum_samples[ybatch[1]] += batch_size

    def __update_lr_rate(self, epoch, cv_ters, best_avg_ters, best_epoch, saver, lr_rate):
        avg_ters = self.__compute_avg_ters(cv_ters)
        if avg_ters <= best_avg_ters:
            print("Improved ter by %.1f%% over previous minimum %.1f%% in epoch %d, not updating learning rate" % (
                100.0 * (best_avg_ters - avg_ters), 100.0 * self.__ter_buffer[1], best_epoch))
            update_lr = False
        else:
            print("ter worsened by %.1f%% from previous minimum %.1f%% in epoch %d, updating learning rate" % (
                100.0 * (avg_ters - best_avg_ters), 100.0 * self.__ter_buffer[1], best_epoch))
            update_lr = True

        if epoch < self.__config[constants.CONF_TAGS.HALF_AFTER]:
            print("should not halve before " + str(self.__config[constants.CONF_TAGS.HALF_AFTER]))
            update_lr = False

        if update_lr:
            print("updating learning rate...")
            print("from: " + str(lr_rate))
            lr_rate /= 2
            print("to: " + str(lr_rate))

        if best_avg_ters > avg_ters:
            best_avg_ters = avg_ters
            best_epoch = epoch

        self.__ter_buffer[0] = self.__ter_buffer[1]
        self.__ter_buffer[1] = avg_ters
        return lr_rate, best_avg_ters, best_epoch


    def __compute_avg_ters(self, ters):
        nters=0
        avg_ters = 0.0
        for language_id, target_scheme in ters.items():
            for target_id, ter in target_scheme.items():
                if(ter > 0):
                    avg_ters += ter
                    nters+=1
        avg_ters /= float(nters)

        return avg_ters

    def __generate_logs(self, cv_ters, cv_cost, ncv, train_ters, train_cost, ntrain, epoch, lr_rate, tic):
        self.__info(
            "Epoch %d finished in %.0f minutes, learning rate: %.4g" % (epoch, (time.time() - tic) / 60.0, lr_rate))
        with open("%s/epoch%02d.log" % (self.__config["model_dir"], epoch), 'w') as fp:
            fp.write("Time: %.0f minutes, lrate: %.4g\n" % ((time.time() - tic) / 60.0, lr_rate))

            for language_id, target_scheme in cv_ters.items():
                if len(cv_ters) > 1:
                    print("Language: " + language_id)
                    fp.write("Language: " + language_id)

                for target_id, cv_ter in target_scheme.items():
                    if len(target_scheme) > 1:
                        print("\tTarget: %s" % (target_id))
                        fp.write("\tTarget: %s" % (target_id))
                    print("\t\t Train    cost: %.1f, ter: %.1f%%, #example: %d" % (
                    train_cost[language_id][target_id], 100.0 * train_ters[language_id][target_id],
                    ntrain[language_id]))
                    print("\t\t" + constants.LOG_TAGS.VALIDATE + " cost: %.1f, ter: %.1f%%, #example: %d" % (
                    cv_cost[language_id][target_id], 100.0 * cv_ter, ncv[language_id]))
                    fp.write("\t\tTrain    cost: %.1f, ter: %.1f%%, #example: %d\n" % (
                    train_cost[language_id][target_id], 100.0 * train_ters[language_id][target_id],
                    ntrain[language_id]))
                    fp.write("\t\t" + constants.LOG_TAGS.VALIDATE + " cost: %.1f, ter: %.1f%%, #example: %d\n" % (
                    cv_cost[language_id][target_id], 100.0 * cv_ter, ncv[language_id]))

    def __info(self, s):
        s = "[" + time.strftime("%Y-%m-%d %H:%M:%S") + "] " + s
        print(s)

    def __get_label_len(self, label):
        idx, _, _ = label
        return len(idx)

    def test(self):
        return
