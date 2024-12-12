import numpy as np
import mindspore as ms
from mindspore import nn, context, Tensor, ops
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


# 数据生成器
def generate_data(num_samples=1000, input_shape=(130, 6)):
    for _ in range(num_samples):
        x = np.random.rand(*input_shape).astype(np.float32)
        y = np.random.randint(0, 2)  # y 是标量
        yield x, y


class CustomDataset:
    def __init__(self, num_samples=1000, input_shape=(130, 6)):
        self.data = list(generate_data(num_samples, input_shape))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


# 模型定义
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


# 超参数
input_size = 6
sequence_length = 130
hidden_size = 32
num_classes = 2
batch_size = 16
num_epochs = 5
learning_rate = 0.001

# 数据加载
data = CustomDataset(num_samples=1000, input_shape=(sequence_length, input_size))
train_dataset = GeneratorDataset(data, column_names=["data", "label"], shuffle=True)
train_dataset = train_dataset.batch(batch_size, drop_remainder=True)  # batch_size 对齐

# 初始化模型
model = LSTMClassifier(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
optimizer = nn.Adam(model.trainable_params(), learning_rate=learning_rate)

# 设置训练
net_with_loss = nn.WithLossCell(model, loss_fn)
train_network = nn.TrainOneStepCell(net_with_loss, optimizer)
train_network.set_train()

# 训练
for epoch in range(num_epochs):
    for batch, (x, y) in enumerate(train_dataset.create_tuple_iterator()):
        x = Tensor(x, ms.float32)
        y = Tensor(y, ms.int32).squeeze()  # 确保标签是 1D
        loss = train_network(x, y)
        if batch % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{batch}], Loss: {loss.asnumpy():.4f}")

# 测试
model.set_train(False)
test_sample = np.random.rand(1, sequence_length, input_size).astype(np.float32)
test_tensor = Tensor(test_sample, ms.float32)
output = model(test_tensor)
predicted_class = ops.argmax(output, axis=1)
print("Test sample prediction:", predicted_class.asnumpy())
