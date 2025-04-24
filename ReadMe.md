# 基于元学习的冷启动推荐系统

本项目基于 TensorFlow 实现了一个用于**推荐系统冷启动问题**的解决方案，融合了 MAML 元学习框架、嵌入生成器、对比学习与用户历史去噪技术，在点击率预测任务中实现了显著的效果提升。

---

## 🧠 项目核心思想

- **DNN 模型**：训练一个基础的推荐模型（Embedding + 多层感知机），作为知识提取器。
- **MAML 框架**：模拟任务（新 item），通过少量样本快速更新冷启动 embedding。
- **冷启动 Embedding 生成器**：
  - 输入：item 的属性 + 相似 item 的聚合特征
  - 输出：新 item 的 embedding，替代随机初始化的 embedding
- **对比损失**：通过对比拆分的相似特征，增强 embedding 判别性。
- **去噪机制**：利用用户历史交互 embedding 平均，缓解长尾用户点击噪声的影响。

---

## 🏗️ 项目结构概览

项目目录结构：
├── ReadMe.md               # 项目说明文件  
├── base_dnn.py             # 主推荐模型训练脚本（基于 DNN）  
├── warm_up_dnn_base.py     # 元学习 + 冷启动嵌入生成器训练脚本  
├── test.py                 # 测试脚本（用于示例或调试）  
├── config_dnn.py           # DNN 模型的配置参数  
├── config_gme.py           # 元学习阶段的配置参数  
├── ctr_funcs.py            # 公共工具函数（包括数据读取和指标计算）  
├── data/                   # 原始数据目录（包含 TFRecord 和 CSV 文件）  
│   ├── big_train_main.csv  
│   ├── big_train_main.tfrecord  
│   ├── test_oneshot_a.csv  
│   └── test_oneshot_a.tfrecord  
├── data_with_hist/         # 含历史点击行为的冷启动数据目录  
│   ├── test_oneshot_a_hist.csv  
│   ├── test_oneshot_a_hist.tfrecord  
│   ├── test_oneshot_b_hist.csv  
│   ├── test_oneshot_b_hist.tfrecord  
│   ├── test_test_w_ngb_hist.tfrecord  
│   ├── train_oneshot_a_w_ngb_hist.csv  
│   ├── train_oneshot_a_w_ngb_hist.tfrecord  
│   ├── train_oneshot_b_w_ngb_hist.csv  
│   └── train_oneshot_b_w_ngb_hist.tfrecord  
├── tmp/                    # 模型保存目录及结果输出目录  
│   ├── dnn/                # 保存的 DNN 模型参数  
│   └── dnn_0801_0900.txt   # 输出的结果或日志文件  


---

## 🔍 模型训练流程

### 📌 第一步：基础推荐模型训练（`base_dnn.py`）

- **输入**：One-hot 类别特征、多值特征（如 `title`, `genres`）  
- **流程**：  
  1. 特征映射到 embedding 空间  
  2. 多层感知机（MLP）用于点击率预测  
- **输出**：  
  - 训练后的 embedding 矩阵（`emb_mat`）  
  - DNN 参数（权重矩阵）  

### 📌 第二步：冷启动嵌入生成器训练（`warm_up_dnn.py`）

- 冻结已训练好的 DNN 参数  
- 构造任务对（A/B 数据集），分别用于梯度更新和验证  
- 计算：  
  - 冷启动 loss（预测误差）  
  - 对比 loss（相似性区分）  
- **总损失**：
  ```text
  L_total = α * L_cold_A + (1 - α) * L_cold_B + L_contrastive


