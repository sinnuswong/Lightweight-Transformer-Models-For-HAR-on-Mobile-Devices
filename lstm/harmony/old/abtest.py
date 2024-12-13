import mindspore
from mindspore import nn, Tensor
import mindspore.dataset as ds
from mindspore.train import Model
import numpy as np

# 定义LSTM模型
class LSTMModel(nn.Cell):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Dense(hidden_size, 64)  # 第一层全连接
        self.fc2 = nn.Dense(64, num_classes)  # 第二层全连接
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def construct(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        x = self.fc1(lstm_out)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# 生成训练数据
def generate_data(num_samples=1000):
    np.random.seed(0)
    X = np.random.rand(num_samples, 6, 130).astype(np.float32)
    y = np.random.randint(0, 2, size=(num_samples,)).astype(np.int32)
    return X, y

X_train, y_train = generate_data()

# 创建MindSpore Dataset
train_ds = ds.NumpySlicesDataset({"data": X_train, "label": y_train}, shuffle=True)

# 创建模型实例
input_size = 130
hidden_size = 32
num_classes = 2
model = LSTMModel(input_size, hidden_size, num_classes)

# 定义损失函数和优化器
loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
optimizer = nn.Adam(params=model.trainable_params(), learning_rate=0.001)
train_model = Model(model, loss_fn, optimizer)

# 计算准确率的函数
def calculate_accuracy(data_iterator):
    correct = 0
    total = 0
    for data in data_iterator:
        inputs = Tensor(data["data"])
        labels = Tensor(data["label"])
        outputs = model(inputs)
        predictions = np.argmax(outputs.asnumpy(), axis=1)
        correct += np.sum(predictions == labels.asnumpy())
        total += labels.shape[0]
    return correct / total if total > 0 else 0

# 训练模型并输出损失和准确率
def train(train_dataset, epochs=10):
    train_dataset = train_dataset.batch(32)  # 设置批次大小
    for epoch in range(1, epochs + 1):  # 从1开始
        # 训练一个 epoch
        train_loss = train_model.train(epoch, train_dataset)

        # 计算当前epoch的准确率
        accuracy = calculate_accuracy(train_dataset.create_dict_iterator())

        # 输出损失与准确率
        print(f'Epoch [{epoch}/{epochs}], Loss: {train_loss}, Accuracy: {accuracy:.2f}')

# 开始训练
train(train_ds, epochs=10)