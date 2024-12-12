import numpy as np
import mindspore
from mindspore import nn, Tensor, ops, context
from mindspore.dataset import GeneratorDataset

import myutil_huawei

import os

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
hit_data_path = '/Users/sinnus/Desktop/ActivityData/badminton/c130/0921right/hit'
save_model_base_path = current_directory + os.sep + 'right_model'
model_name = 'right_dense_hit'
save_model_path_no_extension = save_model_base_path + os.sep + model_name
# 设置执行上下文
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class CustomDataset:
    def __init__(self):
        x, y = myutil_huawei.build_badminton_hit_data(hit_data_path=hit_data_path)
        print(x.shape)
        self.data = x.astype(np.float32)  # 确保数据类型匹配
        self.labels = y.astype(np.int32)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)

    def get_data(self, batch_size):
        num = int(len(self.data) / batch_size)
        res = []
        for i in range(num):
            res.append((i, self.data[i * batch_size:(i + 1) * batch_size, :, :],
                        self.labels[i * batch_size:(i + 1) * batch_size]))
        return res

    # 初始化数据集


dataset = CustomDataset()


# 模型定义
class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.sequential = nn.SequentialCell(
            nn.LSTM(6, 32, batch_first=True),
            nn.Dense(32, 32),
            nn.ReLU(),
            nn.Dense(32, 32),
            nn.ReLU(),
            nn.Dense(32, 2)
        )

    def construct(self, x):
        logits = self.sequential(x)
        return logits


class LSTMClassifier(nn.Cell):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc1 = nn.Dense(hidden_size, 32)
        self.fc2 = nn.Dense(32, num_classes)

    def construct(self, x):
        _, (hn, _) = self.lstm(x)
        hn = hn[-1]  # 获取最后一层的输出
        x = ops.relu(self.fc1(hn))
        x = self.fc2(x)
        return x


model = Network()
# model = LSTMClassifier(input_size=6, hidden_size=32, num_classes=2)

# 超参数
epochs = 10
batch_size = 32
learning_rate = 1e-2

# 损失函数和优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = nn.SGD(model.trainable_params(), learning_rate=learning_rate)


# 训练函数
def train_loop(model, dataset, loss_fn, optimizer):
    def forward_fn(data, label):
        logits = model(data)
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
    for batch,data, label in dataset.get_data(batch_size):
        data = Tensor(data, mindspore.float32)
        label = Tensor(label.squeeze(), mindspore.int32)  # 确保标签是 1D
        loss, logits = train_step(data, label)

        # 计算准确率
        predictions = logits.argmax(axis=1)
        correct += (predictions == label).asnumpy().sum()
        total += label.shape[0]

        if batch % 10 == 0:
            accuracy = 100 * correct / total
            loss_value = loss.asnumpy()
            print(f"loss: {loss_value:>7f}, accuracy: {accuracy:>0.1f}%  [{batch:>3d}/{size:>3d}]")


# 测试函数
def test_loop(model, dataset, loss_fn):
    num_batches = dataset.get_dataset_size()
    model.set_train(False)
    total, test_loss, correct = 0, 0, 0
    for data, label in dataset.create_tuple_iterator():
        data = Tensor(data, mindspore.float32)
        label = Tensor(label.squeeze(), mindspore.int32)  # 确保标签是 1D
        pred = model(data)
        total += len(data)
        test_loss += loss_fn(pred, label).asnumpy()
        correct += (pred.argmax(1) == label).asnumpy().sum()
    test_loss /= num_batches
    correct /= total
    print(f"Test: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# 训练
epochs = 30
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(model, dataset, loss_fn, optimizer)
print("Done!")
