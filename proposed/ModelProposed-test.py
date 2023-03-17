import os
import time
import numpy as np
import pickle as pkl
import tensorflow as tf
from capsnet import CapsNet  # 残差模块
from config import Config  # NEWs
from res_block import inference
from tool.data_parse_ import Data_Inter
from tensorflow.contrib.rnn import LSTMCell
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def check_multi_path(path):
    assert isinstance(path, str) and len(path) > 0
    if '\\' in path:
        path.replace('\\', '/')
    childs = path.split('/')
    root = childs[0]
    for index, cur_child in enumerate(childs):
        if index > 0:
            root = os.path.join(root, cur_child)
        if not os.path.exists(root):
            os.mkdir(root)


class ProposedModel:
    def __init__(self, param_config=None,
                 model_save_path=None,
                 record_path=None,
                 label_path=None):
        self.config = Config()
        self.model_path = self.config.model_saved_path
        self.log_file_path = self.config.logging_file_saved_path
        self.record_path = record_path,
        self.label_path = label_path,
        self.update_embedding = self.config.update_embedding
        self.data_inter = Data_Inter(batch_size=self.config.batch_size,
                                     task_sentence_path=['../../datagenerate/data/processed/sentence_task',
                                                         '../../datagenerate/data/processed/sentence_task_dev'],
                                     # task2id_id2task=['../../datagenerate/data/processed/index_conceptual_explanation_mapping',
                                     task2id_id2task=['../../datagenerate/data/processed/index_conceptual_explanation_mapping_with_n_gram_train',
                                                      # '../../datagenerate/data/processed/index_conceptual_explanation_mapping_dev'],
                                                      '../../datagenerate/data/processed/index_conceptual_explanation_mapping_with_n_gram_dev'],
                                     vocb_path='../../datagenerate/data/processed/vocb')  # 迭代器。
        self.add_placeholders()
        # self.word2id = pkl.load(open('../Tool_news/vocab', mode='rb'))  # 获取本地存放的字典。
        self.word2id = pkl.load(open('../../datagenerate/data/processed/vocb', mode='rb'))  # 获取本地存放的字典。
        self.embeddings = np.float32(np.random.uniform(-0.25, 0.25, (len(self.word2id), 100)))  # 给所有的分词生成初始化向量。详见函数
        # self.embeddings = pkl.load(open('../../Source/w2c/w2c_embedding.pkl', mode='rb'))
        self.build_layer_op()
        self.loss_op()
        self.trainstep_op()

    def add_placeholders(self):
        self.word_ids = tf.placeholder(tf.int32, shape=[self.config.batch_size * 3, 35], name="word_ids1")
        self.task_targets_kw = tf.placeholder(tf.int64, [self.config.batch_size * 35], name='task_targets_kw')
        self.task_targets_ngram = tf.placeholder(tf.int64, [self.config.batch_size * 34, 2], name='task_targets_kw')
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[self.config.batch_size, 3], name="sequence_lengths1")
        self.lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

    def bi_lstm_layer(self, hidden_num, pre_output, fw_name, bw_name, sqw, auto_use=False):
        with tf.variable_scope("a", reuse=tf.AUTO_REUSE):
            cell_fw1 = LSTMCell(hidden_num, name=fw_name)
            cell_bw1 = LSTMCell(hidden_num, name=bw_name)
            (output_fw_seq1, output_bw_seq1), (encoder_fw_final_state1, encoder_bw_final_state1) = \
                tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=cell_fw1,
                    cell_bw=cell_bw1,
                    inputs=pre_output,
                    sequence_length=sqw,
                    dtype=tf.float32)
            output = (output_fw_seq1 + output_bw_seq1) / 2  # 每个时间步的输出
            encoder_final_state_h = tf.concat((encoder_fw_final_state1.h, encoder_bw_final_state1.h), axis=1)
            return output, encoder_final_state_h

    def bi_lstm_layer_onne_use(self, hidden_num, pre_output, fw_name, bw_name):
        with tf.variable_scope("a_none", reuse=tf.AUTO_REUSE):
            cell_fw1 = LSTMCell(hidden_num, name=fw_name)
            cell_bw1 = LSTMCell(hidden_num, name=bw_name)
            (output_fw_seq1, output_bw_seq1), (encoder_fw_final_state1, encoder_bw_final_state1) = \
                tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=cell_fw1,
                    cell_bw=cell_bw1,
                    inputs=pre_output,
                    dtype=tf.float32)
            output = (output_fw_seq1 + output_bw_seq1) / 2  # 每个时间步的输出
            encoder_final_state_h = (encoder_fw_final_state1.h + encoder_bw_final_state1.h) / 2
            return output, encoder_final_state_h

    def create_variables(self, name, shape, initializer=tf.contrib.layers.xavier_initializer(), is_fc_layer=False):

        regularizer = tf.contrib.layers.l2_regularizer(scale=0.0002)
        new_variables = tf.get_variable(name, shape=shape, initializer=initializer,
                                        regularizer=regularizer)
        return new_variables

    def build_layer_op(self):
        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(self.embeddings,
                                           dtype=tf.float32,
                                           trainable=self.update_embedding,
                                           name="_word_embeddings")
            word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings,
                                                     ids=self.word_ids,
                                                     name="word_embeddings")
            # self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout_pl)  # 只有当训练的时候droup才会起作用。
        self.word_embeddings = word_embeddings
        # ######
        #
        # model one:77~92
        # ##############
        # input_word_embeddings = tf.expand_dims(self.word_embeddings, axis=3)
        print('self.word_embeddings:', self.word_embeddings)
        tmp_1 = self.word_embeddings[: 32, :, :]
        tmp_2 = self.word_embeddings[32: 64, :, :]
        tmp_3 = self.word_embeddings[64:, :, :]  # 32, 35, 100
        self.word_embeddings = tf.stack((tmp_1, tmp_2, tmp_3), axis=3)
        print('self.word_embeddings::::::::::', self.word_embeddings)
        # self.word_embeddings = tf.reshape(self.word_embeddings, (self.config.batch_size, 35, 100, 3))
        feature_embedded_input = inference(self.word_embeddings, 3, reuse=False)
        print('feature_embedded_input:', feature_embedded_input)
        # # conv
        # filter = self.create_variables(name='conv', shape=[2, 2, 3, 3])
        # conv_layer = tf.nn.relu(tf.nn.conv2d(feature_embedded_input, filter, strides=[1, 1, 2, 1], padding='SAME'))
        # w, h, b = conv_layer.shape[1:]
        # print('conv_layer:', conv_layer)
        # finall_feature = tf.reshape(conv_layer, (self.config.batch_size * 3, b, w * h))
        print('finall_feature:', feature_embedded_input)
        cause = feature_embedded_input[:, :, :, 0]
        effect = feature_embedded_input[:, :, :, 1]
        explain = feature_embedded_input[:, :, :, 2]
        print('shape of all:cause:{}\tffect:{}\texplain:{}'.format(cause.shape, effect.shape, explain.shape))
        # #################################
        # ######
        #
        # model two:96~123
        # ##############
        # collection_bilstms = []
        # for cur_data in range(0, self.word_embeddings.shape[0], 1):  # 每个样本均进行学习
        #     output1, encoder_final_state_h1 = self.bi_lstm_layer_onne_use(hidden_num=self.word_embeddings.shape[-1],
        #                                                                   pre_output=self.word_embeddings[cur_data: cur_data + 1, :, :],
        #                                                                   fw_name='cell_fw11', bw_name='cell_bw11')
        #     collection_bilstms.append(encoder_final_state_h1)
        # print(collection_bilstms)
        # finall_feature1 = tf.concat(collection_bilstms, axis=0)
        finall_feature_cause, _ = self.bi_lstm_layer(hidden_num=64,
                                                     pre_output=self.word_embeddings[:, :, :, 0],
                                                     fw_name='cell_fw11', bw_name='cell_bw11',
                                                     sqw=self.sequence_lengths[:, 0])
        finall_feature_effect, _ = self.bi_lstm_layer(hidden_num=64,
                                                      pre_output=self.word_embeddings[:, :, :, 1],
                                                      fw_name='cell_fw11', bw_name='cell_bw11',
                                                      sqw=self.sequence_lengths[:, 1])
        finall_feature_explain, _ = self.bi_lstm_layer(hidden_num=64,
                                                       pre_output=self.word_embeddings[:, :, :, 2],
                                                       fw_name='cell_fw11', bw_name='cell_bw11',
                                                       sqw=self.sequence_lengths[:, 2])
        print('finall_feature1:', finall_feature_cause, finall_feature_effect, finall_feature_explain)
        feature_merged_cause = tf.concat([finall_feature_cause, cause], axis=2)
        feature_merged_effect = tf.concat([finall_feature_effect, effect], axis=2)
        feature_merged = tf.concat([feature_merged_cause, feature_merged_effect], axis=2)
        print('shape of all:feature_merged_cause:{}\tfeature_merged_effect:{}\tfeature_merged:{}'.format(
            feature_merged_cause.shape, feature_merged_effect.shape, feature_merged.shape))
        with tf.variable_scope("proj"):
            # w_task = tf.get_variable(name="W_task",
            #                          shape=[128, len(self.word2id.keys())],  # rumor classification
            #                          initializer=tf.contrib.layers.xavier_initializer(),
            #                          dtype=tf.float32)
            # b_task = tf.get_variable(name="b_task",
            #                          shape=[len(self.word2id.keys())],
            #                          initializer=tf.zeros_initializer(),
            #                          dtype=tf.float32)
            print('finall_feature1:', feature_merged)
            # ##
            w_task1 = tf.get_variable(name="W_task1",
                                      shape=[feature_merged.shape[2], len(self.word2id.keys())],  # rumor classification
                                      initializer=tf.contrib.layers.xavier_initializer(),
                                      dtype=tf.float32)
            b_task1 = tf.get_variable(name="b_task1",
                                      shape=[len(self.word2id.keys())],
                                      initializer=tf.zeros_initializer(),
                                      dtype=tf.float32)
            print('finall_feature1:', feature_merged)

            #

            # print('feature2:', finall_feature2)
            # feature_merged = tf.concat([finall_feature, finall_feature1, finall_feature2], axis=1)
            # feature_merged = tf.concat([finall_feature, finall_feature1], axis=1)
            # print('feature_merged:', feature_merged)
            # intent_logits_kw_feature = tf.add(tf.matmul(feature_merged, w_task1), b_task1)
            # intent_logits_kw = tf.add(tf.matmul(intent_logits_kw_feature, w_task), b_task)
            intent_logits_kw = tf.add(tf.matmul(feature_merged, w_task1), b_task1)

            print('距离损失:', finall_feature_explain)
            # ####
            # 损失
            self.softmax_score_kw = tf.nn.softmax(intent_logits_kw)
            self.task = tf.argmax(intent_logits_kw, axis=2)
            # #########################################3
            self.task_kw_flatten = tf.reshape(self.task, shape=(self.config.batch_size * 35,))  # classify
            self.acc = tf.reduce_mean(tf.cast(tf.equal(self.task_kw_flatten, self.task_targets_kw), dtype=tf.float32))
            # 损失1
            cross_entropy_kw = tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.one_hot(self.task_targets_kw, depth=len(self.word2id), dtype=tf.float32),
                logits=tf.reshape(intent_logits_kw, (32 * 35, -1)))
            # 损失2
            ngram_tuple = tf.concat((self.task[:, 1:], self.task[:, 0: 1] * 0), axis=1)
            ngram_tuple = tf.stack((self.task, ngram_tuple), axis=2)[:, : -1]
            ngram_tuple = tf.reshape(ngram_tuple, (-1, 2))
            # cross_entropy_kw = tf.nn.softmax_cross_entropy_with_logits(
            #     labels=tf.one_hot(self.task_targets_kw, depth=len(self.word2id), dtype=tf.float32),
            #     logits=intent_logits_kw)
            # # #
            # real_target = tf.ones(shape=(1088, 2))
            # cross_entropy_kw1 = tf.nn.softmax_cross_entropy_with_logits(
            #     # labels=tf.one_hot(ngram_tuple, depth=2, dtype=tf.float32),
            #     labels=ngram_tuple,
            #     logits=real_target)

            print('ngram_tuple, self.task_targets_ngram:', ngram_tuple, self.task_targets_ngram)
            ngram_loss = tf.cast(tf.equal(ngram_tuple, self.task_targets_ngram), dtype=tf.float32)
            ngram_loss = tf.reduce_mean(tf.cast(tf.equal(ngram_loss[:, 0], ngram_loss[:, 1]), dtype=tf.float32))
            self.loss_task_kw = tf.reduce_mean(cross_entropy_kw) + ngram_loss * tf.log(ngram_loss)
            self.all_loss = self.loss_task_kw

    def loss_op(self):
            self.loss = self.all_loss  # 任务识别的损失

    def trainstep_op(self):
        """
        训练节点.
        """
        with tf.variable_scope("train_step"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)  # 全局训批次的变量，不可训练。
            if self.config.optimizer == 'Adam':
                optim = tf.train.AdamOptimizer(learning_rate=.0001)
            elif self.config.optimizer == 'Adadelta':
                optim = tf.train.AdadeltaOptimizer(learning_rate=.0001)
            elif self.config.optimizer == 'Adagrad':
                optim = tf.train.AdagradOptimizer(learning_rate=.0001)
            elif self.config.optimizer == 'RMSProp':
                optim = tf.train.RMSPropOptimizer(learning_rate=.0001)
            elif self.config.optimizer == 'Momentum':
                optim = tf.train.MomentumOptimizer(learning_rate=.0001, momentum=0.9)
            elif self.config.optimizer == 'SGD':
                optim = tf.train.GradientDescentOptimizer(learning_rate=.0001)
            else:
                optim = tf.train.GradientDescentOptimizer(learning_rate=.0001)
            grads_and_vars = optim.compute_gradients(self.loss)
            grads_and_vars_clip = [[tf.clip_by_value(g, -self.config.get_clip, self.config.get_clip), v] for g, v in grads_and_vars]
            self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)

    def get_word_em_bf(self, index_input):
        # print('shape of index_input:', index_input.shape)
        new_data = np.zeros(shape=[32, 21, 80, 128])
        for cur_batch in range(32):
            for cur_repos in range(21):
                for index, cur_word in enumerate(index_input[cur_batch][cur_repos]):
                    new_data[cur_batch][cur_repos][index] = self.embeddings[cur_word]
        return new_data

    def pad_sequences(self, sequences, pad_mark=0, predict=False):
        """
        批量的embedding，其中rowtext embedding的长度要与slots embedding的长度一致，不然使用crf时会出错。
        :param sequences: 批量的文本格式[[], [], ......, []]，其中子项[]里面是一个完整句子的embedding（索引。）
        :param pad_mark:  长度不够时，使用何种方式进行padding
        :param predict:  是否是测试
        :return:
        """
        # print('sequences:', sequences)
        max_len = 35
        # tmp_length = max([len(x) for x in sequences])
        # print('tmp_length:', tmp_length)
        # max_len = tmp_length
        seq_list, seq_len_list = [], []
        for seq in sequences:
            seq = list(seq)
            if predict:
                seq = list(map(lambda x: self.word2id.get(x, 0), seq))
            # seq_ = seq[:len(seq)] + [pad_mark] * max(max_len - len(seq), 0)  # 求得最大的索引长度。
            seq_ = seq[: max_len] if len(seq) >= max_len else seq + [pad_mark] * (max_len - len(seq))  # 求得最大的索引长度。
            seq_list.append(seq_)
            seq_len_list.append(min(len(seq), max_len))
        return seq_list, seq_len_list

    def train(self, log_file=None):
        """
            数据由一个外部迭代器提供。
        """
        if log_file is None:
            log_file = open(self.config.logging_file_saved_path.__add__('proposed_mt.txt'), mode='w', encoding='utf-8')
            print('Not lazy......')
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            # sess.run(tf.global_variables_initializer())
            ckpt_file = tf.train.latest_checkpoint('../../model_save/'.__add__('/proposed/'))
            if ckpt_file is not None:
                print('ckpt_file:', ckpt_file)
                saver.restore(sess, ckpt_file)
            else:
                sess.run(tf.global_variables_initializer())
            batches_recording = 0
            for epoch_index in range(0, 300000, 1):
                start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                sentence_, tasks_all, task_ngram_train = self.data_inter.next()  #
                sentence_length_train = tasks_all[:, -3:]
                tasks_train = tasks_all[:, : -3]
                # print('shape of tasks_all:', tasks_all.shape, task_ngram_train.shape)
                #
                tasks_train = tasks_train.flatten()
                cause_train = sentence_[:, 0]
                effect_train = sentence_[:, 1]
                explain_train = sentence_[:, 2]
                cause_train_f, cause_train_f_length = self.pad_sequences(cause_train)
                effect_train_f, effect_train_length = self.pad_sequences(effect_train)
                explain_train_f, explain_train_length = self.pad_sequences(explain_train)
                sentence_train = np.concatenate((cause_train_f, effect_train_f, explain_train_f), axis=0)
                # print('shape of sentence_length_train:', sentence_length_train.shape)
                # print('sentence_length_train:', sentence_train.shape, tasks_train.shape, sentence_length_train)
                _, loss_train, acc_cur, step_num_, qq = sess.run([self.train_op, self.loss, self.acc,
                                                                  self.global_step, self.task], feed_dict={
                    self.lr_pl: 0.001,
                    self.word_ids: sentence_train,
                    self.task_targets_kw: tasks_train,
                    self.sequence_lengths: sentence_length_train,
                    self.task_targets_ngram: task_ngram_train
                })

                if epoch_index % 1 == 0:
                    sentence_test, task_test, task_ngram = self.data_inter.next_test()  # 迭代器，每次取出一个batch块.
                    sentence_length_test = task_test[:, -3:]
                    task_test = task_test[:, : -3]
                    task_test = task_test.flatten()

                    cause_test = sentence_test[:, 0]
                    effect_test = sentence_test[:, 1]
                    explain_test = sentence_test[:, 2]
                    cause_test_f, cause_test_f_length = self.pad_sequences(cause_test)
                    effect_test_f, effect_test_length = self.pad_sequences(effect_test)
                    explain_test_f, explain_test_length = self.pad_sequences(explain_test)
                    sentence_test = np.concatenate((cause_test_f, effect_test_f, explain_test_f), axis=0)
                    sentence_length_test = np.array(sentence_length_test)
                    print('sentence_:', sentence_length_test.shape)
                    # feed_dict, sqer = self.get_feed_dict(sentence_, [tasks_], self.config.learning_rate, self.config.keep_dropout)
                    # sentence_ = np.array(self.pad_sequences(sentence_))
                    # sentence_ = self.get_word_em_bf(sentence_)
                    # print(sentence_.shape)
                    # try:
                    loss_test, acc_kw_test, task_kw_for_score, step_num_, qq = sess.run([self.loss, self.acc, self.task_kw_flatten,
                                                                                         self.global_step, self.task], feed_dict={
                        self.lr_pl: 0.001,
                        self.word_ids: sentence_test,
                        self.task_targets_kw: task_test,
                        self.sequence_lengths: sentence_length_test,
                        self.task_targets_ngram: task_ngram
                    })
                    # 加入其它的准确率
                    # #####################################
                    f1 = []
                    pre = []
                    recall = []
                    k_batch = len(task_test) // 35
                    for j in range(0, k_batch, 1):
                        task_test_tmp = list(filter(lambda x: x > 0, task_test[j * 35: (j + 1) * 35]))
                        task_kw_for_score_tmp = list(filter(lambda x: x > 0, task_kw_for_score[j * 35: (j + 1) * 35]))
                        min_test = min(len(task_test_tmp), len(task_kw_for_score_tmp))
                        task_test_tmp = task_test_tmp[: min_test]
                        task_kw_for_score_tmp = task_kw_for_score_tmp[: min_test]

                        f1_tmp = f1_score(task_test_tmp, task_kw_for_score_tmp, average='weighted')
                        pre_tmp = precision_score(task_test_tmp, task_kw_for_score_tmp, average='weighted')
                        recall_tmp = recall_score(task_test_tmp, task_kw_for_score_tmp, average='weighted')
                        f1.append(f1_tmp)
                        pre.append(pre_tmp)
                        recall.append(recall_tmp)

                    # ####
                    f1 = sum(f1) / len(f1)
                    pre = sum(pre) / len(pre)
                    r = sum(recall) / len(recall)
                    # ####################################
                    if log_file is not None:
                        log_file.write('time:'.__add__(start_time).__add__('\tepoch: ').
                                       __add__(str(epoch_index + 1)).__add__('\tstep:').
                                       __add__(str(batches_recording + epoch_index)).
                                       __add__('\tloss:').__add__(str(loss_test)).
                                       __add__('\tacc_kw:').__add__(str(acc_kw_test)).
                                       __add__('\tf1:').__add__(str(f1)).
                                       __add__('\tpre:').__add__(str(pre)).
                                       __add__('\trecall:').__add__(str(r)).__add__('\n'))
                        log_file.flush()
                    print('test time {} epoch {}, step {}, loss: {:.4}, acc_kw: {:.4}, f1-score_kw: {:.4}, '
                          'precision_kw: {:.4}, recall_kw: {:.4}'.
                          format(start_time, epoch_index + 1, batches_recording + epoch_index *
                                 epoch_index, loss_test, acc_kw_test, f1, pre, r))

                if epoch_index % 2000 == 0:
                    check_multi_path(self.model_path)
                    saver.save(sess, self.model_path, global_step=epoch_index)
                # except Exception as ex:
                #     print('batch error......')
        if log_file is not None:
            log_file.close()

    def get_feed_dict(self, seqs, labels=None, lr=None, dropout=None, tag=None, predicted=False):
        """

        :param seqs:  训练的batch块
        :param labels:  实体标签
        :param lr:  学利率
        :param dropout:  活跃的节点数，全连接层
        :return: feed_dict  训练数据
        :return: predicted  测试标志
        """
        print('seqs:', seqs.shape)
        word_ids, seq_len_list = self.pad_sequences(seqs, pad_mark=0, predict=predicted)
        print('word_ids:', np.array(word_ids).shape)
        print('seq_len_list:', seq_len_list)
        feed_dict = {self.word_ids: word_ids,  # embedding到同一长度
                     self.sequence_lengths: seq_len_list,  # 实际长度。
                     }
        if labels is not None:
            feed_dict[self.task_targets_kw] = labels[0]
        if lr is not None:
            feed_dict[self.lr_pl] = lr
        return feed_dict, seq_len_list

    def predict(self):
        """

        :param sess:
        :param seqs:
        :param predicted:
        :return: label_list
                 seq_len_list
        """
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            ckpt_file = tf.train.latest_checkpoint('../model'.__add__('/task/'))
            saver.restore(sess, ckpt_file)
            test_data_loader = DataLoad()
            index_cur = 0
            while index_cur < test_data_loader.batchs:
                sentence_, tasks_ = self.data_inter.next()  #
                sentence_ = np.array(self.pad_sequences(sentence_))
                sentence_ = self.get_word_em_bf(sentence_)
                task_acc = sess.run([self.acc],
                                    feed_dict={
                                         self.input_net: sentence_,
                                         self.targets: tasks_
                                    })
                print(task_acc)
                index_cur += 1


if __name__ == "__main__":
    test_model = ProposedModel()
    test_model.train()
    # test_model.predict()
