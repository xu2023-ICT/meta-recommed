import numpy as np
import tensorflow as tf
import datetime
import ctr_funcs as func
from time import time
import config_dnn as cfg
import os

save_model_ind = 1 # 是否保存模型，1是保存，0是不保存
num_csv_col = 17 # 数据的列数，label一列，特征16列
pre = './data/'
suf = '.tfrecord'
train_file_name = [pre+'big_train_main'+suf] # 大推荐模型的训练数据
test_file_name = [pre+'test_oneshot_a'+suf] # 大推荐模型的测试数据，这里的测试结果不重要
n_ft = 11134 # embedding矩阵的大小
k = 10 # embedding维度的大小
kp_prob = 1.0 # 废弃了的参数
n_epoch = 1 # 推荐模型只能训练一个epoch
rnd_seed = 123 # 随机种子
record_step_size = 2000 # 每2000步打一次日志
layer_dim = [256, 128, 1] # 网络维度
opt_alg = 'Adam' # 优化器
n_one_hot_slot = 6 # 非文本特征个数
n_mul_hot_slot = 2 # 文本特征个数
max_len_per_slot = 5 # 每个文本特征占几列
input_format = 'tfrecord' # 文件格式
eta_range = [1e-3] # 学习率
batch_size_range = [128]  # batch大小
total_num_ft_col = 16 # 除去label的列数

## 尝试不同的参数组合
para_list = []
for i in range(len(eta_range)):
    for ii in range(len(batch_size_range)):
        para_list.append([eta_range[i], batch_size_range[ii]])

## record results
result_list = []

# 加载不同组参数
for item in para_list:
    eta = item[0]
    batch_size = item[1]
    tf.reset_default_graph() # 清空计算图 tensor自己的问题，不清空再次赋值就会报错
    idx_1 = n_one_hot_slot
    idx_2 = idx_1 + n_mul_hot_slot*max_len_per_slot

    print('Loading data start!')
    tf.set_random_seed(rnd_seed)

    # 读取数据
    # ft是特征的意思 
    train_ft, train_label = func.tfrecord_input_pipeline(train_file_name, num_csv_col, batch_size, n_epoch)
    test_ft, test_label = func.tfrecord_input_pipeline_test(test_file_name, num_csv_col, batch_size, 1)
    print('Loading data done!')

    # 根据x_input_one_hot得到其对应的embedding的函数，其中mask操作可做可不做，没啥大用
    def get_masked_one_hot(x_input_one_hot):
        '''
        将one-hot编码的特征映射到embedding空间，并屏蔽掉值为0的部分。
        '''
        data_mask = tf.cast(tf.greater(x_input_one_hot, 0), tf.float32)
        data_mask = tf.expand_dims(data_mask, axis = 2)
        data_mask = tf.tile(data_mask, (1,1,k))
        # output: (?, n_one_hot_slot, k)
        data_embed_one_hot = tf.nn.embedding_lookup(emb_mat, x_input_one_hot)
        data_embed_one_hot_masked = tf.multiply(data_embed_one_hot, data_mask)
        return data_embed_one_hot_masked

    # x_input_mul_hot，其中mask操作可做可不做，没啥大用
    def get_masked_mul_hot(x_input_mul_hot):
        '''
        将多值特征（multi-hot）转换成 embedding 向量，并屏蔽无效值，最后对每个字段的多值 embedding 做加和（池化）。
        '''
        data_mask = tf.cast(tf.greater(x_input_mul_hot, 0), tf.float32)
        data_mask = tf.expand_dims(data_mask, axis = 3)
        data_mask = tf.tile(data_mask, (1,1,1,k))
        # output: (?, n_mul_hot_slot, max_len_per_slot, k)
        data_embed_mul_hot = tf.nn.embedding_lookup(emb_mat, x_input_mul_hot)
        data_embed_mul_hot_masked = tf.multiply(data_embed_mul_hot, data_mask)
        # move reduce_sum here
        data_embed_mul_hot_masked = tf.reduce_sum(data_embed_mul_hot_masked, 2)
        return data_embed_mul_hot_masked

    # embedding连接函数
    def get_concate_embed(x_input_one_hot, x_input_mul_hot):
        '''
        将one-hot编码和多值特征的 embedding 向量连接起来。
        '''
        data_embed_one_hot = get_masked_one_hot(x_input_one_hot)
        data_embed_mul_hot = get_masked_mul_hot(x_input_mul_hot)
        data_embed_concat = tf.concat([data_embed_one_hot, data_embed_mul_hot], 1)
        return data_embed_concat

    # 预测函数，也就是个DNN
    def get_dnn_output(data_embed_concat):
        '''
        构建并执行一个多层全连接神经网络（DNN），用于将 embedding 特征映射成最终的预测输出（点击率或评分等）。
        '''
        # include output layer
        n_layer = len(layer_dim)
        data_embed_dnn = tf.reshape(data_embed_concat, [-1, (n_one_hot_slot + n_mul_hot_slot)*k])
        cur_layer = data_embed_dnn
        # loop to create DNN struct
        for i in range(0, n_layer):
            # output layer, linear activation
            if i == n_layer - 1:
                cur_layer = tf.matmul(cur_layer, weight_dict[i]) #+ bias_dict[i]
            else:
                cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight_dict[i])) # + bias_dict[i])
                cur_layer = tf.nn.dropout(cur_layer, keep_prob)

        y_hat = cur_layer
        return y_hat

    # 定义输入输出的格式
    x_input = tf.placeholder(tf.int32, shape=[None, total_num_ft_col])
    # shape=[None, n_one_hot_slot]
    x_input_one_hot = x_input[:, 0:idx_1]
    x_input_mul_hot = x_input[:, idx_1:idx_2]
    # shape=[None, n_mul_hot_slot, max_len_per_slot]
    x_input_mul_hot = tf.reshape(x_input_mul_hot, (-1, n_mul_hot_slot, max_len_per_slot))

    # target vect
    y_target = tf.placeholder(tf.float32, shape=[None, 1])
    # dropout keep prob
    keep_prob = tf.placeholder(tf.float32)

    # 初始化大的embedding矩阵
    with tf.device('/cpu:0'):
        emb_mat = tf.Variable(tf.random_normal([n_ft + 1, k], stddev=0.01))

    # 定义DNN网络的参数
    n_layer = len(layer_dim)
    in_dim = (n_one_hot_slot + n_mul_hot_slot)*k
    weight_dict={}
    for i in range(0, n_layer):
        out_dim = layer_dim[i]
        cur_range = np.sqrt(6.0 / (in_dim + out_dim))
        weight_dict[i] = tf.Variable(tf.random_uniform([in_dim, out_dim], -cur_range, cur_range))
        in_dim = layer_dim[i]

    ####### DNN计算 ########
    data_embed_concat = get_concate_embed(x_input_one_hot, x_input_mul_hot)
    y_hat = get_dnn_output(data_embed_concat)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=y_target))

    #############################
    # prediction
    #############################
    pred_score = tf.sigmoid(y_hat)

    # 优化器，反向传播，梯度下降，更新参数
    optimizer = tf.train.AdamOptimizer(eta).minimize(loss)

    ########## 接下来就是tensorflow1特有的流程，如果看了视频的话，应该可以看懂 ################
    # 大致就是，往定义好的计算图里填真实的数据，进行真实的训练和预测过程 #
    # 配置Tensorflow Session,动态按需申请GPU
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    
    with tf.Session(config=config) as sess:
        # TensorFlow训练前的流程
        # 启动计算图、初始化所有变量，并开启多线程的数据输入队列，为后续训练做好准备。
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)

        train_loss_list = []

        func.print_time()
        print('Start train loop')

        t1 = time()

        epoch = -1
        try:
            while not coord.should_stop():
                epoch += 1
                train_ft_inst, train_label_inst = sess.run([train_ft, train_label])

                sess.run(optimizer, feed_dict={x_input:train_ft_inst, \
                                               y_target:train_label_inst, keep_prob:kp_prob})

                # record loss and accuracy every step_size generations
                if (epoch+1)%record_step_size == 0:
                    train_loss_temp = sess.run(loss, feed_dict={ \
                                               x_input:train_ft_inst, \
                                               y_target:train_label_inst, keep_prob:1.0})
                    train_loss_list.append(train_loss_temp)

                    auc_and_loss = [epoch+1, train_loss_temp]
                    auc_and_loss = [np.round(xx,4) for xx in auc_and_loss]
                    func.print_time()
                    print('Generation # {}. Train Loss: {:.4f}.'\
                          .format(*auc_and_loss))
        # 训练完成
        except tf.errors.OutOfRangeError:
            func.print_time()
            # whether to save the model
            if save_model_ind == 1:
                save_dict = {}
                save_dict['emb_mat'] = emb_mat
                for i in range(0, n_layer):
                    cur_key = 'weight_dict[' + str(i) + ']'
                    save_dict[cur_key] = weight_dict[i]
#                     cur_key = 'bias_dict[' + str(i) + ']'
#                     save_dict[cur_key] = bias_dict[i]
                saver = tf.train.Saver(save_dict)
                save_path = saver.save(sess, './tmp/dnn/')
                print("Model saved in file: %s" % save_path)
            print('Done training -- epoch limit reached')

        train_time = time() - t1

        # load test data
        test_pred_score_all = []
        test_label_all = []
        test_loss_all = []
        # 测试 得到pre_score和loss
        t2 = time()
        try:
            while True:
                test_ft_inst, test_label_inst = sess.run([test_ft, test_label])

                cur_test_pred_score = sess.run(pred_score, feed_dict={ \
                                        x_input:test_ft_inst, keep_prob:1.0})
                test_pred_score_all.append(cur_test_pred_score.flatten())
                test_label_all.append(test_label_inst)

                cur_test_loss = sess.run(loss, feed_dict={ \
                                        x_input:test_ft_inst, \
                                        y_target:test_label_inst, keep_prob:1.0})
                test_loss_all.append(cur_test_loss)

        except tf.errors.OutOfRangeError:
            func.print_time()
            print('Done testing -- epoch limit reached')
        finally:
            coord.request_stop()

        test_time = time() - t2

        coord.join(threads)

        # calculate auc
        # 评估指标计算
        test_pred_score_re = func.list_flatten(test_pred_score_all)
        test_label_re = func.list_flatten(test_label_all)
        test_auc, _, _ = func.cal_auc(test_pred_score_re, test_label_re)
        test_rmse = func.cal_rmse(test_pred_score_re, test_label_re)
        test_loss = np.mean(test_loss_all)

        # rounding
        test_auc = np.round(test_auc, 4)
        test_rmse = np.round(test_rmse, 4)
        test_loss = np.round(test_loss, 5)
        train_loss_list = [np.round(xx,4) for xx in train_loss_list]

        print('test_auc = ', test_auc)
        print('test_rmse =', test_rmse)
        print('test_loss =', test_loss)
        print('train_loss_list =', train_loss_list)

        # append to result_list
        result_list.append([eta, batch_size, test_auc, test_loss, train_time, test_time])

fmt_str = '{:<6}\t{:<6}\t{:<6}\t{:<6}\t{}\t{}'
header_row = ['eta', 'bs', 'auc', 'loss', 'train_time', 'test_time']
print(fmt_str.format(*header_row))

for i in range(len(result_list)):
    tmp = result_list[i]
    print(fmt_str.format(*tmp))