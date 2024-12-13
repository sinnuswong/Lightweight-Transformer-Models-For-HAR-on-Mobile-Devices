import numpy as np
import mindspore
from mindspore import nn, Tensor, ops, context
from mindspore.dataset import GeneratorDataset

import myutil

import os
from mindspore import save_checkpoint, load_checkpoint, load_param_into_net
from mindspore import nn, context, export

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

# config
window_size = 130
feature_size = 6
num_classes = 2

hidden_size = 32
output_size = num_classes
input_shape = (window_size, feature_size)
batch_size = 128
num_epochs = 5
learning_rate = 1e-3

hit_data_path = './train_hit_c25'
save_model_base_path = current_directory + os.sep + 'left_model'
model_name = 'left_lstm_hit'
save_model_path_no_extension = save_model_base_path + os.sep + model_name


class CustomDataset:
    def __init__(self):
        x, y = myutil.build_badminton_hit_data(hit_data_path=hit_data_path)
        print(x.shape)
        self.data = x.astype(np.float32)  # 确保数据类型匹配
        self.labels = y.astype(np.int32)

        # 定义划分比例
        train_ratio = 0.85
        test_ratio = 0.15

        # 计算训练集的大小
        num_samples = self.data.shape[0]  # 1000
        num_train = int(num_samples * train_ratio)

        # 生成随机打乱的索引
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        # 划分数据集
        train_indices = indices[:num_train]
        test_indices = indices[num_train:]

        self.train_data = self.data[train_indices]
        self.train_labels = self.labels[train_indices]
        self.test_data = self.data[test_indices]
        self.test_labels = self.labels[test_indices]

        # 输出数据集的形状以验证
        print(f'Train Data Shape: {self.train_data.shape}')
        print(f'Train Labels Shape: {self.train_labels.shape}')
        print(f'Test Data Shape: {self.test_data.shape}')
        print(f'Test Labels Shape: {self.test_labels.shape}')

        self.current_index = 0
        num_samples = self.train_data.shape[0]
        self.num_batches = num_samples // batch_size  # 每个 epoch 的批次数

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)

    def get_next_batch(self):
        num_samples = self.train_data.shape[0]

        if self.current_index >= num_samples:
            self.current_index = 0  # 重新开始
            # 如果需要打乱数据，可以在这里打乱
            # np.random.shuffle(train_indices) # 需要重新产生索引
        batch_indices = range(self.current_index, min(self.current_index + batch_size, num_samples))
        self.current_index += batch_size
        print(self.train_data[batch_indices].shape)
        print(self.train_labels[batch_indices].shape)

        return self.train_data[batch_indices], self.train_labels[batch_indices]


dataset = CustomDataset()


class LSTMModel(nn.Cell):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=feature_size, hidden_size=hidden_size, batch_first=True)
        self.fc1 = nn.Dense(hidden_size, hidden_size)
        self.fc2 = nn.Dense(hidden_size, num_classes)

    def construct(self, x):
        _, (hn, _) = self.lstm(x)
        hn = hn[-1]  # 获取最后一层的输出
        x = self.fc1(hn)
        x = ops.relu(x)
        x = self.fc2(x)
        return x


model = LSTMModel()

# loss_fn = nn.CrossEntropyLoss()
loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
optimizer = nn.Adam(params=model.trainable_params(), learning_rate=learning_rate)


def train_loop(model, dataset, loss_fn, optimizer):
    def forward_fn(data, label):
        logits = model(data)
        # print("hahahahah "+logits)
        loss = loss_fn(logits, label)
        return loss, logits

    grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    def train_step(data, label):
        (loss, logits), grads = grad_fn(data, label)
        loss = ops.depend(loss, optimizer(grads))
        return loss, logits

    size = len(dataset)
    model.set_train()
    correct = 0
    total = 0

    data, label = dataset.data, dataset.labels
    data = Tensor(data, mindspore.float32)
    label = Tensor(label.squeeze(), mindspore.int32)  # 确保标签是 1D
    loss, logits = train_step(data, label)
    # print(logits)

    # 计算准确率
    predictions = logits.argmax(axis=1)
    correct += (predictions == label).asnumpy().sum()
    total += label.shape[0]

    accuracy = 100 * correct / total
    loss_value = loss.asnumpy()
    print(f"Training loss: {loss_value:>7f}, accuracy: {accuracy:>0.1f}%")

    # for i in range(dataset.num_batches):
    #     data, label = dataset.get_next_batch()
    #     data = Tensor(data, mindspore.float32)
    #     label = Tensor(label.squeeze(), mindspore.int32)  # 确保标签是 1D
    #     loss, logits = train_step(data, label)
    #
    #     # 计算准确率
    #     predictions = logits.argmax(axis=1)
    #     correct += (predictions == label).asnumpy().sum()
    #     total += label.shape[0]
    #
    #     accuracy = 100 * correct / total
    #     loss_value = loss.asnumpy()
    #     print(f"loss: {loss_value:>7f}, accuracy: {accuracy:>0.1f}%")


def test_loop(model, dataset, loss_fn):
    num_batches = len(dataset)
    model.set_train(False)
    total, test_loss, correct = 0, 0, 0

    data, label = dataset.test_data, dataset.test_labels
    data = Tensor(data, mindspore.float32)
    label = Tensor(label.squeeze(), mindspore.int32)  # 确保标签是 1D
    pred = model(data)
    total += len(data)
    test_loss += loss_fn(pred, label).asnumpy()
    correct += (pred.argmax(1) == label).asnumpy().sum()

    # test_loss /= num_batches
    correct /= total
    print(f"Verify: loss: {test_loss:>8f}, accuracy: {(100 * correct):>0.1f}%")


# 训练
for t in range(num_epochs):
    print(f"Epoch {t + 1} -------------------------------")
    train_loop(model, dataset, loss_fn, optimizer)
    test_loop(model, dataset, loss_fn)
print("Done!")

# 保存模型
checkpoint_file = save_model_path_no_extension + ".ckpt"
save_checkpoint(model, checkpoint_file)
print(f"checkpoints saved to {checkpoint_file}")

# 加载模型
loaded_model = LSTMModel()
param_dict = load_checkpoint(checkpoint_file)
load_param_into_net(loaded_model, param_dict)

print("checkpoints loaded")
input = np.random.uniform(0.0, 1.0, size=[32, 130, 6]).astype(np.float32)  # Lenet模型的size为32,1,32,32
export(loaded_model, Tensor(input), file_name=save_model_path_no_extension, file_format='MINDIR')  # file_name指定转换后文件的文件名
print("convert to MINDIR success")
