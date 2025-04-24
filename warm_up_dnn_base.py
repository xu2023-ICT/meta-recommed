import numpy as np
import tensorflow as tf
import ctr_funcs as func
import config_gme as cfg
import os
import logging

# 去掉警告，方便阅读结果
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
logging.getLogger("tensorflow").setLevel(logging.FATAL)
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)

# 一堆配置
warm_up_bool = False  # 是否进行warm up
test_batch_size = cfg.test_batch_size
meta_mode = cfg.meta_mode
alpha = cfg.alpha
gamma = cfg.gamma
train_file_name_warm = cfg.train_file_name_warm
train_file_name_warm_2 = cfg.train_file_name_warm_2
n_epoch = cfg.n_epoch
label_col_idx = 0
num_csv_col_warm = cfg.num_csv_col_warm
total_num_ft_col_warm = num_csv_col_warm - 1
num_csv_col_w_ngb = cfg.num_csv_col_w_ngb
total_num_ft_col_cold = num_csv_col_w_ngb - 1
tar_idx = cfg.tar_idx
attr_idx = cfg.attr_idx
str_txt = cfg.output_file_name
base_path = './tmp'
model_loading_addr = cfg.model_loading_addr
model_saving_addr = base_path + '/meta_' + str_txt + '/'
output_file_name = base_path + '/meta_' + str_txt + '.txt'
save_model_ind = cfg.save_model_ind
train_file_name_a = cfg.train_file_name_a
train_file_name_b = cfg.train_file_name_b
test_file_name = cfg.test_file_name
batch_size = cfg.batch_size
n_ft = cfg.n_ft
k = cfg.k
kp_prob = cfg.kp_prob
n_epoch_meta = cfg.n_epoch_meta
record_step_size = cfg.record_step_size
layer_dim = cfg.layer_dim
att_dim = cfg.att_dim
opt_alg = cfg.opt_alg
n_one_hot_slot = cfg.n_one_hot_slot
n_mul_hot_slot = cfg.n_mul_hot_slot
max_len_per_slot = cfg.max_len_per_slot
input_format = cfg.input_format
n_slot = n_one_hot_slot + n_mul_hot_slot
n_one_hot_slot_ngb = cfg.n_one_hot_slot_ngb
n_mul_hot_slot_ngb = cfg.n_mul_hot_slot_ngb
max_len_per_slot_ngb = cfg.max_len_per_slot_ngb
max_n_ngb_ori = cfg.max_n_ngb_ori
max_n_ngb = cfg.max_n_ngb
n_slot_ngb = n_one_hot_slot_ngb + n_mul_hot_slot_ngb
eta_range = cfg.eta_range
meta_batch_size_range = cfg.meta_batch_size_range
cold_eta_range = cfg.cold_eta_range
meta_eta_range = cfg.meta_eta_range
tar_slot_map = {}


for i in range(len(tar_idx)):
    tar_slot_map[tar_idx[i]] = i

# 构造para_list，用于尝试不同的参数组合
para_list = []
for i in range(len(eta_range)):
    for ii in range(len(meta_batch_size_range)):
        for iii in range(len(cold_eta_range)):
            for iv in range(len(meta_eta_range)):
                para_list.append([eta_range[i], meta_batch_size_range[ii], cold_eta_range[iii], \
                                  meta_eta_range[iv]])

# 用于记录结果
result_list = []

# 尝试不同的参数组合
for item in para_list:
    eta = item[0]
    meta_batch_size = item[1] # batch的大小
    cold_eta = item[2]    # MAML中，一步梯度下降的学习率
    meta_eta = item[3]   # 优化器的学习率

    tf.reset_default_graph()  # tensorflow中的函数，用于刷新图（百度下~）
    tf.set_random_seed(123)

    # 读取数据
    print('Loading data start!')
    if input_format == 'tfrecord':
        # 如果开启了warm up功能，要把两个预热的数据集读进来，进行两次warm up。warm up的意思是当数据不断增多时，对embedding矩阵进行更新
        if warm_up_bool:
            # warm up数据集1
            train_ft_warm, train_label_warm, train_hist_warm = func.tfrecord_input_pipeline_test_zb(['./data_with_hist/test_oneshot_a_hist.tfrecord'], num_csv_col_warm, batch_size, n_epoch)
            train_ft_meta_warm, train_label_meta_warm, train_hist_meta_warm = func.tfrecord_input_pipeline_test_zb(['./data_with_hist/test_oneshot_a_hist.tfrecord'],num_csv_col_warm, batch_size,n_epoch)
            test_ft_warm, test_label_warm, test_hist_warm = func.tfrecord_input_pipeline_test_zb(['./data_with_hist/test_test_w_ngb_hist.tfrecord'], num_csv_col_w_ngb,test_batch_size, 1)
            test_ft_meta_warm, test_label_meta_warm, test_hist_meta_warm = func.tfrecord_input_pipeline_test_zb(['./data_with_hist/test_test_w_ngb_hist.tfrecord'],num_csv_col_w_ngb, test_batch_size, 1)
            test_ft_copy, test_label_copy, test_hist_copy = func.tfrecord_input_pipeline_test_zb(['./data_with_hist/test_test_w_ngb_hist.tfrecord'], num_csv_col_w_ngb,test_batch_size, 1)

            # warm up数据集2
            train_ft_warm_2, train_label_warm_2, train_hist_warm_2 = func.tfrecord_input_pipeline_test_zb(['./data_with_hist/test_oneshot_b_hist.tfrecord'],num_csv_col_warm, batch_size,n_epoch)
            train_ft_meta_warm_2, train_label_meta_warm_2, train_hist_meta_warm_2 = func.tfrecord_input_pipeline_test_zb(['./data_with_hist/test_oneshot_b_hist.tfrecord'],num_csv_col_warm,batch_size, n_epoch)
            test_ft_warm_2, test_label_warm_2, test_hist_warm_2 = func.tfrecord_input_pipeline_test_zb(['./data_with_hist/test_test_w_ngb_hist.tfrecord'], num_csv_col_w_ngb,test_batch_size, 1)
            test_ft_meta_warm_2, test_label_meta_warm_2, test_hist_meta_warm_2 = func.tfrecord_input_pipeline_test_zb(['./data_with_hist/test_test_w_ngb_hist.tfrecord'],num_csv_col_w_ngb,test_batch_size, 1)

        # 读取训练生成器需要的数据集。这就是文档里提到的数据集Da和数据集Db
        train_ft_a, train_label_a, train_hist_a = func.tfrecord_input_pipeline_test_zb(
            ['./data_with_hist/train_oneshot_a_w_ngb_hist.tfrecord'], num_csv_col_w_ngb,
            meta_batch_size, n_epoch_meta)

        train_ft_b, train_label_b, train_hist_b = func.tfrecord_input_pipeline_test_zb(
            ['./data_with_hist/train_oneshot_b_w_ngb_hist.tfrecord'], num_csv_col_w_ngb,
            meta_batch_size, n_epoch_meta)

        test_ft, test_label, test_hist = func.tfrecord_input_pipeline_test_zb(
            ['./data_with_hist/test_test_w_ngb_hist.tfrecord'], num_csv_col_w_ngb, test_batch_size, 1)

        test_ft_meta, test_label_meta, test_hist_meta = func.tfrecord_input_pipeline_test_zb(
            ['./data_with_hist/test_test_w_ngb_hist.tfrecord'], num_csv_col_w_ngb,
            test_batch_size, 1)

    # 划分数据，其中x_input_one_hot是正常的特征，x_input_mul_hot是多值特征（例如title等）
    def partition_input(x_input):
        idx_1 = n_one_hot_slot
        idx_2 = idx_1 + n_mul_hot_slot * max_len_per_slot
        x_input_one_hot = x_input[:, 0:idx_1]
        x_input_mul_hot = x_input[:, idx_1:idx_2]
        x_input_mul_hot = tf.reshape(x_input_mul_hot, (-1, n_mul_hot_slot, max_len_per_slot))
        return x_input_one_hot, x_input_mul_hot

    # 划分数据，包括自己的数据和邻居的数据。这里的邻居指的是最相似的k个老item
    def partition_input_w_ngb(x_input):
        # generate idx_list
        len_list = []
        # tar
        len_list.append(n_one_hot_slot)
        len_list.append(n_mul_hot_slot * max_len_per_slot)

        # ngb
        for _ in range(max_n_ngb_ori):
            len_list.append(n_one_hot_slot_ngb)
            len_list.append(n_mul_hot_slot_ngb * max_len_per_slot_ngb)

        len_list = np.array(len_list)
        idx_list = np.cumsum(len_list)

        x_input_one_hot = x_input[:, 0:idx_list[0]]
        x_input_mul_hot = x_input[:, idx_list[0]:idx_list[1]]
        # shape=[None, n_mul_hot_slot, max_len_per_slot]
        x_input_mul_hot = tf.reshape(x_input_mul_hot, [-1, n_mul_hot_slot, max_len_per_slot])

        #######################
        # ngb
        concat_one_hot_ngb = x_input[:, idx_list[1]:idx_list[2]]
        concat_mul_hot_ngb = x_input[:, idx_list[2]:idx_list[3]]
        for i in range(1, max_n_ngb_ori):
            # one_hot
            temp_1 = x_input[:, idx_list[2 * i + 1]:idx_list[2 * i + 2]]
            concat_one_hot_ngb = tf.concat([concat_one_hot_ngb, temp_1], 1)

            # mul_hot
            temp_2 = x_input[:, idx_list[2 * i + 2]:idx_list[2 * i + 3]]
            concat_mul_hot_ngb = tf.concat([concat_mul_hot_ngb, temp_2], 1)

        concat_one_hot_ngb = tf.reshape(concat_one_hot_ngb, [-1, max_n_ngb_ori, n_one_hot_slot_ngb])

        concat_mul_hot_ngb = tf.reshape(concat_mul_hot_ngb, [-1, max_n_ngb_ori, n_mul_hot_slot_ngb, \
                                                             max_len_per_slot_ngb])

        x_input_one_hot_ngb = concat_one_hot_ngb[:, 0:max_n_ngb, :]
        x_input_mul_hot_ngb = concat_mul_hot_ngb[:, 0:max_n_ngb, :, :]

        return x_input_one_hot, x_input_mul_hot, x_input_one_hot_ngb, x_input_mul_hot_ngb

    # 根据x_input_one_hot得到其对应的embedding的函数，其中mask操作可做可不做，没啥大用
    def get_masked_one_hot(x_input_one_hot):
        data_mask = tf.cast(tf.greater(x_input_one_hot, 0), tf.float32)
        data_mask = tf.expand_dims(data_mask, axis=2)
        data_mask = tf.tile(data_mask, (1, 1, k))
        # output: (?, n_one_hot_slot, k)
        data_embed_one_hot = tf.nn.embedding_lookup(emb_mat, x_input_one_hot)
        data_embed_one_hot_masked = tf.multiply(data_embed_one_hot, data_mask)
        return data_embed_one_hot_masked

    # x_input_mul_hot，其中mask操作可做可不做，没啥大用
    def get_masked_mul_hot(x_input_mul_hot):
        data_mask = tf.cast(tf.greater(x_input_mul_hot, 0), tf.float32)
        data_mask = tf.expand_dims(data_mask, axis=3)
        data_mask = tf.tile(data_mask, (1, 1, 1, k))
        # output: (?, n_mul_hot_slot, max_len_per_slot, k)
        data_embed_mul_hot = tf.nn.embedding_lookup(emb_mat, x_input_mul_hot)
        data_embed_mul_hot_masked = tf.multiply(data_embed_mul_hot, data_mask)
        # move reduce_sum here
        data_embed_mul_hot_masked = tf.reduce_sum(data_embed_mul_hot_masked, 2)
        return data_embed_mul_hot_masked

    # embedding连接函数
    def get_concate_embed(x_input_one_hot, x_input_mul_hot):
        data_embed_one_hot = get_masked_one_hot(x_input_one_hot)
        data_embed_mul_hot = get_masked_mul_hot(x_input_mul_hot)
        data_embed_concat = tf.concat([data_embed_one_hot, data_embed_mul_hot], 1)
        return data_embed_concat

    # 为one_hot向量增加掩码的函数，带邻居节点
    def get_masked_one_hot_ngb(x_input_one_hot_ngb):
        data_mask = tf.cast(tf.greater(x_input_one_hot_ngb, 0), tf.float32)
        data_mask = tf.expand_dims(data_mask, axis=3)
        data_mask = tf.tile(data_mask, (1, 1, 1, k))
        # output: (?, max_n_clk, n_one_hot_slot, k)
        data_embed_one_hot = tf.nn.embedding_lookup(emb_mat, x_input_one_hot_ngb)
        data_embed_one_hot_masked = tf.multiply(data_embed_one_hot, data_mask)
        return data_embed_one_hot_masked

    # 为multi_hot向量增加掩码的函数，带邻居节点
    def get_masked_mul_hot_ngb(x_input_mul_hot_ngb):
        data_mask = tf.cast(tf.greater(x_input_mul_hot_ngb, 0), tf.float32)
        data_mask = tf.expand_dims(data_mask, axis=4)
        data_mask = tf.tile(data_mask, (1, 1, 1, 1, k))
        # output: (?, max_n_clk, n_mul_hot_slot, max_len_per_slot, k)
        data_embed_mul_hot = tf.nn.embedding_lookup(emb_mat, x_input_mul_hot_ngb)
        data_embed_mul_hot_masked = tf.multiply(data_embed_mul_hot, data_mask)
        # output: (?, max_n_clk, n_mul_hot_slot, k)
        data_embed_mul_hot_masked = tf.reduce_sum(data_embed_mul_hot_masked, 3)
        return data_embed_mul_hot_masked

    # embedding连接函数。把自己的embedding和邻居的item id embedding连接在一起
    def get_concate_embed_ngb(x_input_one_hot_ngb, x_input_mul_hot_ngb):
        data_embed_one_hot = get_masked_one_hot_ngb(x_input_one_hot_ngb)
        data_embed_mul_hot = get_masked_mul_hot_ngb(x_input_mul_hot_ngb)
        data_embed_concat = tf.concat([data_embed_one_hot, data_embed_mul_hot], 2)
        return data_embed_concat

    # 替换预热后的嵌入向量（用生成器生成的id emb替换原来不好的id emb）
    def get_sel_col(data_embed_concat, col_idx):
        cur_col_idx = col_idx[0]
        # none * len(col_idx) * k
        ft_emb = data_embed_concat[:, cur_col_idx:cur_col_idx + 1, :]
        for i in range(1, len(col_idx)):
            cur_col_idx = col_idx[i]
            cur_x = data_embed_concat[:, cur_col_idx:cur_col_idx + 1, :]
            ft_emb = tf.concat([ft_emb, cur_x], 1)
        # reshape -> 2D none * total_dim
        ft_emb = tf.reshape(ft_emb, [-1, len(col_idx) * k])
        return ft_emb

    # 替换预热后的嵌入向量，带邻居节点
    def get_sel_col_ngb(data_embed_concat_ngb, col_idx):
        cur_col_idx = col_idx[0]
        # none * max_n_ngb * len(col_idx) * k
        ngb_emb = data_embed_concat_ngb[:, :, cur_col_idx:cur_col_idx + 1, :]
        for i in range(1, len(col_idx)):
            cur_col_idx = col_idx[i]
            cur_x = data_embed_concat_ngb[:, :, cur_col_idx:cur_col_idx + 1, :]
            ngb_emb = tf.concat([ngb_emb, cur_x], 2)
        # reshape -> 3D none * max_n_ngb * total_dim
        ngb_emb = tf.reshape(ngb_emb, [-1, max_n_ngb, len(col_idx) * k])
        return ngb_emb

    # embedding连接函数，用于MAML框架之后
    def get_concate_embed_w_meta(data_embed_concat, pred_emb):
        cur_slot_idx = 0
        if cur_slot_idx in tar_idx:
            cur_col_idx = tar_slot_map[cur_slot_idx]
            final_emb = pred_emb[:, cur_col_idx:cur_col_idx + 1, :]
        else:
            final_emb = data_embed_concat[:, cur_slot_idx:cur_slot_idx + 1, :]

        for i in range(1, n_slot):
            cur_slot_idx = i
            if cur_slot_idx in tar_idx:
                cur_col_idx = tar_slot_map[cur_slot_idx]
                cur_x = pred_emb[:, cur_col_idx:cur_col_idx + 1, :]
            else:
                cur_x = data_embed_concat[:, cur_slot_idx:cur_slot_idx + 1, :]
            final_emb = tf.concat([final_emb, cur_x], 1)
        return final_emb


    # 预测函数，也就是个DNN
    def get_y_hat(final_emb):
        # include output layer
        n_layer = len(layer_dim)
        data_embed_dnn = tf.reshape(final_emb, [-1, n_slot * k])
        cur_layer = data_embed_dnn
        # loop to create DNN struct
        for i in range(0, n_layer):
            # output layer, linear activation
            if i == n_layer - 1:
                cur_layer = tf.matmul(cur_layer, weight_dict[i])  # + bias_dict[i]
            else:
                cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight_dict[i]))  # + bias_dict[i])
                cur_layer = tf.nn.dropout(cur_layer, keep_prob)

        y_hat = cur_layer
        return y_hat

    # 模块一：嵌入生成器，生成良好的id emb。这里实现的是简易版，只输入了item的特征，面试时应该说，输入是item特征+老item的id emb
    def get_new_emebdding(data_embed_concat):
        data_embed_concat = data_embed_concat[:, 5:, :]

        weight = tf.reduce_sum(data_embed_concat, 2)  # (?,10)压缩矩阵，按维度1求和。把一个序列中所有item的emb都加起来
        weight = weight/10

        # weight = 2*gamma * tf.nn.sigmoid(tf.matmul(weight, W_SENET))
        weight = tf.expand_dims(weight, axis=-1)
        data_embed_concat = tf.multiply(data_embed_concat, weight)

        data_embed_concat = tf.reshape(data_embed_concat, [-1, 30])
        pred_emb = gamma * tf.nn.tanh(tf.matmul(data_embed_concat, W_meta))

        pred_emb = tf.reshape(pred_emb, [-1, len(tar_idx), k])

        return pred_emb

    # 模块二：对比学习，优化嵌入空间。实现时也是简易版，但面试时答给你的文档里写的
    def xiangsi_SENET(data_embed_concat):
        tensor = get_sel_col(data_embed_concat, [5, 6, 7])
        mask_1 = tf.ones_like(tensor)
        mask_0 = tf.zeros_like(tensor)
        mask1 = tf.concat([mask_1[:, 0:10], mask_0[:, 10:]], axis=1)
        tensor_qian_half = tf.multiply(tensor, mask1)
        mask2 = tf.concat([mask_0[:, 0:10], mask_1[:, 10:]], axis=1)
        tensor_hou_half = tf.multiply(tensor, mask2)

        tensor_qian_half = tf.reshape(tensor_qian_half, [-1, 3, 10])
        tensor_hou_half = tf.reshape(tensor_hou_half, [-1, 3, 10])

        weight = tf.reduce_sum(tensor_qian_half, 2)  # (?,10)压缩矩阵，按维度1求和。把一个序列中所有item的emb都加起来
        weight = weight / 10
        weight = 2*gamma * tf.nn.sigmoid(tf.matmul(weight, W_SENET))
        weight = tf.expand_dims(weight, axis=-1)
        tensor_qian_half = tf.multiply(tensor_qian_half, weight)
        tensor_qian_half = tf.reshape(tensor_qian_half, [-1, 30])

        weight = tf.reduce_sum(tensor_hou_half, 2)  # (?,10)压缩矩阵，按维度1求和。把一个序列中所有item的emb都加起来
        weight = weight / 10
        weight = 2*gamma * tf.nn.sigmoid(tf.matmul(weight, W_SENET))
        weight = tf.expand_dims(weight, axis=-1)
        tensor_hou_half = tf.multiply(tensor_hou_half, weight)
        tensor_hou_half = tf.reshape(tensor_hou_half, [-1, 30])


        data_embed_concat2 = gamma * tf.nn.tanh(tf.matmul(tensor_qian_half, W_meta))
        data_embed_concat4 = gamma * tf.nn.tanh(tf.matmul(tensor_hou_half, W_meta))

        # 把张量拉成矢量，这是我自己的应用需求
        tensor1 = tf.reshape(data_embed_concat2, shape=(1, -1))
        tensor2 = tf.reshape(data_embed_concat4, shape=(1, -1))
        # 求模长
        tensor1_norm = tf.sqrt(tf.reduce_sum(tf.square(tensor1)))
        tensor2_norm = tf.sqrt(tf.reduce_sum(tf.square(tensor2)))
        # 内积
        tensor1_tensor2 = tf.reduce_sum(tf.multiply(tensor1, tensor2))
        cosin = tensor1_tensor2 / (tensor1_norm * tensor2_norm)
        return tf.exp(-cosin)

    # 模块三：使用user序列嵌入去噪
    def get_hist_mean(x_input_hist, emb):
        data_embed_one_hot = get_masked_one_hot(x_input_hist)
        data_embed_one_hot = tf.concat([data_embed_one_hot, emb], axis=1)
        data_embed_one_hot = tf.reduce_sum(data_embed_one_hot, 1)  # (?,10)压缩矩阵，按维度1求和。把一个序列中所有item的emb都加起来
        hist = data_embed_one_hot/20
        data_embed_dnn = tf.reshape(hist, [-1, 10])
        # DNN操作
        cur_layer = tf.nn.relu(tf.matmul(data_embed_dnn, hist_vars_1))
        cur_layer = tf.expand_dims(cur_layer, 1)
        return cur_layer

    # 评价指标
    def get_metric(test_pred_score_all, test_label_all):
        test_pred_score_re = func.list_flatten(test_pred_score_all)
        test_label_re = func.list_flatten(test_label_all)
        test_auc, _, _ = func.cal_auc(test_pred_score_re, test_label_re)
        return test_auc

    # 预热池
    if warm_up_bool:
        x_input_warm = tf.placeholder(tf.int32, shape=[None, total_num_ft_col_warm])
        x_input_one_hot_warm, x_input_mul_hot_warm = partition_input(x_input_warm)
        y_target_warm = tf.placeholder(tf.float32, shape=[None, 1])

    # 占位符（这个是tensorflow1的一种机制）
    x_input_a = tf.placeholder(tf.int32, shape=[None, total_num_ft_col_cold])
    x_input_a_hist = tf.placeholder(tf.int32, shape=[None, 20])
    x_input_one_hot_a, x_input_mul_hot_a, x_input_one_hot_ngb_a, x_input_mul_hot_ngb_a = partition_input_w_ngb(x_input_a)
    y_target_a = tf.placeholder(tf.float32, shape=[None, 1])

    x_input_b = tf.placeholder(tf.int32, shape=[None, total_num_ft_col_cold])
    x_input_b_hist = tf.placeholder(tf.int32, shape=[None, 20])
    x_input_one_hot_b, x_input_mul_hot_b, _, _ \
        = partition_input_w_ngb(x_input_b)
    y_target_b = tf.placeholder(tf.float32, shape=[None, 1])

    # dropout keep prob
    keep_prob = tf.placeholder(tf.float32)

    ############################
    # emb_mat dim add 1 -> for padding (idx = 0)
    with tf.device('/cpu:0'):
        emb_mat = tf.Variable(tf.random_normal([n_ft + 1, k], stddev=0.01))

    if warm_up_bool:
        # placeholder for new emb_mat
        emb_mat_input = tf.placeholder(tf.float32, shape=[n_ft + 1, k])
        emb_mat_assign_op = tf.assign(emb_mat, emb_mat_input)

    # 生成器的参数初始化

    W_meta = tf.Variable(tf.random_uniform([30, 10], -np.sqrt(6.0 / (30 + 10)), np.sqrt(6.0 / (30 + 10))))
    W_SENET = tf.Variable(tf.random_uniform([3, 3], -np.sqrt(6.0 / (3 + 3)), np.sqrt(6.0 / (3 + 3))))
    hist_vars_1 = tf.Variable(tf.random_uniform([10, 10], -np.sqrt(6.0 / (10 + 10)), np.sqrt(6.0 / (10 + 10))))
    # 设置元学习需要更新的参数
    meta_vars = [W_meta, W_SENET, hist_vars_1]

    ####### 参数设置与初始化 ############
    n_layer = len(layer_dim)
    in_dim = n_slot * k
    weight_dict = {}

    # 主模型的参数，不参与元学习的更新，直接从训练好的主模型里读取即可
    for i in range(0, n_layer):
        out_dim = layer_dim[i]
        cur_range = np.sqrt(6.0 / (in_dim + out_dim))
        weight_dict[i] = tf.Variable(tf.random_uniform([in_dim, out_dim], -cur_range, cur_range))
        in_dim = layer_dim[i]

    ####### DNN ########
    if warm_up_bool:
        data_embed_concat_warm = get_concate_embed(x_input_one_hot_warm, x_input_mul_hot_warm)
        y_hat_warm = get_y_hat(data_embed_concat_warm)
        # used for training
        warm_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat_warm, labels=y_target_warm))
        warm_vars = [emb_mat]

    ###########  MAML框架训练过程  ###########
    # 第一步：对冷启动阶段进行评估
    data_embed_concat_a = get_concate_embed(x_input_one_hot_a, x_input_mul_hot_a)
    y_hat = get_y_hat(data_embed_concat_a)
    # used for eval only
    eval_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=y_target_a))
    # 用三个模块，生成良好的id嵌入
    pred_emb_a = get_new_emebdding(data_embed_concat_a)
    pred_emb_a = get_hist_mean(x_input_a_hist, pred_emb_a)
    contrastive_loss = xiangsi_SENET(data_embed_concat_a)

    final_emb_a = get_concate_embed_w_meta(data_embed_concat_a, pred_emb_a)
    cold_y_hat_a = get_y_hat(final_emb_a)
    cold_loss_a = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=cold_y_hat_a, labels=y_target_a))

    ###############
    # 第二步：更新一步梯度
    cold_emb_grads = tf.gradients(cold_loss_a, pred_emb_a)[0]
    pred_emb_a_new = pred_emb_a - cold_eta * cold_emb_grads

    ###############
    # 第三步：用一步更新后的movie id emb，在其他mini-batch上做预测
    data_embed_concat_b = get_concate_embed(x_input_one_hot_b, x_input_mul_hot_b)
    final_emb_b = get_concate_embed_w_meta(data_embed_concat_b, pred_emb_a_new)
    cold_y_hat_b = get_y_hat(final_emb_b)
    cold_loss_b = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=cold_y_hat_b, labels=y_target_b))

    ###############
    # 第四步，计算总损失，并用CEM调优
    meta_loss = cold_loss_a * alpha + cold_loss_b * (1 - alpha) + contrastive_loss

    pred_score = tf.sigmoid(y_hat)  # using ori emb
    pred_score_a = tf.sigmoid(cold_y_hat_a)  # using new, gen emb

    if opt_alg == 'Adam':
        meta_optimizer = tf.train.AdamOptimizer(meta_eta).minimize(meta_loss, var_list=meta_vars)
        if warm_up_bool:
            warm_optimizer = tf.train.AdamOptimizer(eta).minimize(warm_loss, var_list=warm_vars)


    ########################################
    # Launch the graph.
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.3

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        print('Start train loop')

        ######################################
        test_pred_score_all = []
        test_label_all = []

        test_pred_score_all_meta = []
        test_label_all_meta = []

        if warm_up_bool:
            test_pred_score_all_warm = []
            test_label_all_warm = []

            test_pred_score_all_meta_warm = []
            test_label_all_meta_warm = []

            test_pred_score_all_warm_2 = []
            test_label_all_warm_2 = []

            test_pred_score_all_meta_warm_2 = []
            test_label_all_meta_warm_2 = []

        save_dict = {}
        save_dict['emb_mat'] = emb_mat
        for i in range(0, n_layer):
            cur_key = 'weight_dict[' + str(i) + ']'
            save_dict[cur_key] = weight_dict[i]
        saver = tf.train.Saver(save_dict)

        saver.restore(sess, './tmp/dnn/')

        ######################################
        # A: test directly
        ######################################
        try:
            while True:
                test_ft_inst, test_label_inst = sess.run([test_ft, test_label])
                cur_test_pred_score = sess.run(pred_score, feed_dict={ \
                    x_input_a: test_ft_inst, keep_prob: 1.0})

                test_pred_score_all.append(cur_test_pred_score.flatten())
                test_label_all.append(test_label_inst)

        except tf.errors.OutOfRangeError:
            print('Done direct testing')

        if warm_up_bool:
            ######################################
            # B1: warm up training - update emb_mat
            ######################################
            try:
                while True:
                    train_ft_inst_warm, train_label_inst_warm = sess.run([train_ft_warm, train_label_warm])
                    # run warm optimizer
                    sess.run(warm_optimizer,
                             feed_dict={x_input_warm: train_ft_inst_warm, y_target_warm: train_label_inst_warm, \
                                        keep_prob: kp_prob})
            except tf.errors.OutOfRangeError:
                print('Done warm up training')

            ######################################
            # B2: warm up testing
            ######################################
            try:
                while True:
                    test_ft_inst, test_label_inst = sess.run([test_ft_warm, test_label_warm])
                    cur_test_pred_score = sess.run(pred_score, feed_dict={ \
                        x_input_a: test_ft_inst, keep_prob: 1.0})
                    test_pred_score_all_warm.append(cur_test_pred_score.flatten())
                    test_label_all_warm.append(test_label_inst)

            except tf.errors.OutOfRangeError:
                print('Done warm up testing')

            ######################################
            # B3: warm up training - update emb_mat
            ######################################
            try:
                while True:
                    train_ft_inst_warm, train_label_inst_warm = sess.run([train_ft_warm_2, train_label_warm_2])
                    # run warm optimizer
                    sess.run(warm_optimizer,
                             feed_dict={x_input_warm: train_ft_inst_warm, y_target_warm: train_label_inst_warm, \
                                        keep_prob: kp_prob})
            except tf.errors.OutOfRangeError:
                print('Done warm up training 2')

            ######################################
            # B4: warm up testing
            ######################################
            try:
                while True:
                    test_ft_inst, test_label_inst = sess.run([test_ft_warm_2, test_label_warm_2])
                    cur_test_pred_score = sess.run(pred_score, feed_dict={ \
                        x_input_a: test_ft_inst, keep_prob: 1.0})
                    test_pred_score_all_warm_2.append(cur_test_pred_score.flatten())
                    test_label_all_warm_2.append(test_label_inst)

            except tf.errors.OutOfRangeError:
                print('Done warm up testing 2')

        ######################################
        # C1: meta training - update GME params
        ######################################
        # reload model params before warm up
        saver.restore(sess, './tmp/dnn/')

        epoch = -1
        try:
            while True:
                epoch += 1
                train_ft_inst_a, train_label_inst_a, train_hist_inst_a = sess.run([train_ft_a, train_label_a, train_hist_a])
                train_ft_inst_b, train_label_inst_b, train_hist_inst_b = sess.run([train_ft_b, train_label_b, train_hist_b])
                sess.run(meta_optimizer, feed_dict={x_input_a: train_ft_inst_a, y_target_a: train_label_inst_a, \
                                                    x_input_b: train_ft_inst_b, y_target_b: train_label_inst_b, \
                                                    x_input_a_hist: train_hist_inst_a,
                                                    x_input_b_hist: train_hist_inst_b,
                                                    keep_prob: kp_prob})

        except tf.errors.OutOfRangeError:
            print('Done meta training')

        ######################################
        # C2: meta testing
        ######################################
        try:
            while True:
                test_ft_inst, test_label_inst, test_hist_inst = sess.run([test_ft_meta, test_label_meta, test_hist_meta])
                cur_test_pred_score = sess.run(pred_score_a, feed_dict={ \
                    x_input_a: test_ft_inst, keep_prob: 1.0, x_input_a_hist: test_hist_inst})
                test_pred_score_all_meta.append(cur_test_pred_score.flatten())
                test_label_all_meta.append(test_label_inst)

        except tf.errors.OutOfRangeError:
            print('Done meta testing')

        if warm_up_bool:
            ######################################
            # D0: update emb_mat of new ads by GME
            ######################################
            try:
                # get emb_mat
                emb_mat_val = sess.run(emb_mat)
                while True:
                    test_ft_inst_copy, test_label_inst_copy, test_hist_inst_copy = sess.run([test_ft_copy, test_label_copy, test_hist_copy])
                    # get new ID embs
                    pred_emb_w_val = sess.run(pred_emb_a, feed_dict={x_input_a: test_ft_inst_copy, x_input_a_hist: test_hist_inst_copy})
                    # assume only 1 tar idx
                    id_col = test_ft_inst_copy[:, tar_idx]
                    for iter_ee in range(len(id_col)):
                        cur_ft_id = id_col[iter_ee]
                        cur_emb = pred_emb_w_val[iter_ee, :]
                        emb_mat_val[cur_ft_id, :] = cur_emb
            except tf.errors.OutOfRangeError:
                # update emb_mat
                sess.run(emb_mat_assign_op, feed_dict={emb_mat_input: emb_mat_val})
                print('Done update emb_mat with meta')

            ######################################
            # D1: warm up training after meta
            ######################################
            try:
                while True:
                    train_ft_inst_warm, train_label_inst_warm = sess.run([train_ft_meta_warm, train_label_meta_warm])
                    # run warm optimizer
                    sess.run(warm_optimizer,
                             feed_dict={x_input_warm: train_ft_inst_warm, y_target_warm: train_label_inst_warm,
                                        keep_prob: kp_prob})
            except tf.errors.OutOfRangeError:
                print('Done warm up training after meta')

            ######################################
            # D2: warm up testing after meta
            ######################################
            try:
                while True:
                    test_ft_inst, test_label_inst = sess.run([test_ft_meta_warm, test_label_meta_warm])
                    cur_test_pred_score = sess.run(pred_score, feed_dict={ \
                        x_input_a: test_ft_inst, keep_prob: 1.0})
                    test_pred_score_all_meta_warm.append(cur_test_pred_score.flatten())
                    test_label_all_meta_warm.append(test_label_inst)

            except tf.errors.OutOfRangeError:
                print('Done warm up testing after meta')

            ######################################
            # D3: warm up training after meta
            ######################################
            try:
                while True:
                    train_ft_inst_warm, train_label_inst_warm = sess.run(
                        [train_ft_meta_warm_2, train_label_meta_warm_2])
                    # run warm optimizer
                    sess.run(warm_optimizer,
                             feed_dict={x_input_warm: train_ft_inst_warm, y_target_warm: train_label_inst_warm, \
                                        keep_prob: kp_prob})
            except tf.errors.OutOfRangeError:
                print('Done warm up training after meta 2')

            ######################################
            # D4: warm up testing after meta
            ######################################
            try:
                while True:
                    test_ft_inst, test_label_inst = sess.run([test_ft_meta_warm_2, test_label_meta_warm_2])
                    cur_test_pred_score = sess.run(pred_score, feed_dict={ \
                        x_input_a: test_ft_inst, keep_prob: 1.0})
                    test_pred_score_all_meta_warm_2.append(cur_test_pred_score.flatten())
                    test_label_all_meta_warm_2.append(test_label_inst)

            except tf.errors.OutOfRangeError:
                print('Done warm up testing after meta 2')

        #############################
        # dummy opt to pass syntax check
        # otherwise, you have
        # try
        # except
        # if [-> syntax error]
        # finally
        try:
            (aa, bb) = (2, 1)
            cc = aa / bb
        except ZeroDivisionError:
            print('divide by zero')
        #############################

        finally:
            coord.request_stop()
        coord.join(threads)

        # calculate metric
        test_auc = get_metric(test_pred_score_all, test_label_all)
        test_auc_meta = get_metric(test_pred_score_all_meta, test_label_all_meta)

        if warm_up_bool:
            test_auc_warm = get_metric(test_pred_score_all_warm, test_label_all_warm,)
            test_auc_meta_warm = get_metric(test_pred_score_all_meta_warm,test_label_all_meta_warm)
            test_auc_warm_2 = get_metric(test_pred_score_all_warm_2, test_label_all_warm_2)
            test_auc_meta_warm_2 = get_metric(test_pred_score_all_meta_warm_2,test_label_all_meta_warm_2)

            result_list.append([batch_size, meta_batch_size, eta, cold_eta, meta_eta, \
                                test_auc, \
                                test_auc_meta, \
                                test_auc_warm, \
                                test_auc_meta_warm, \
                                test_auc_warm_2, \
                                test_auc_meta_warm_2])
        else:
            result_list.append([meta_batch_size, cold_eta, meta_eta, \
                                test_auc, \
                                test_auc_meta])


# 记录实验结果
# auc指不用生成器，冷启动item就用随机初始化的id embedding，所得到的推荐结果
# meta auc指用生成器为item生成id embedding后，所得到的推荐效果
if warm_up_bool:
    header_row = ['bs', 'mbs', 'eta', 'cold_eta', 'meta_eta', \
                  'auc', \
                  'meta auc',  \
                  'warm1 auc',  \
                  'warm1 meta auc',  \
                  'warm1 auc',  \
                  'warm2 meta auc']
else:
    header_row = ['meta_bs', 'cold_eta', 'meta_eta', \
                  'auc',  \
                  'meta auc']
print('*' * 20)
print('meta_mode: ' + meta_mode)
fmt_str = '{:<10}' * len(header_row)
print(fmt_str.format(*header_row))
fmt_str = '{:<10.5f}' * len(header_row)
for i in range(len(result_list)):
    tmp = result_list[i]
    print(fmt_str.format(*tmp))

