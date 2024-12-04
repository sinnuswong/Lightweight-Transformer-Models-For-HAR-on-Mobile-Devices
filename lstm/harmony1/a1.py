import mindspore
from mindspore import nn, context
from mindspore import Tensor
from mindspore import save_checkpoint, load_checkpoint, load_param_into_net
from mindspore.train import Model
import mindspore.dataset as ds
import numpy as np

class LSTMModel(nn.Cell):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Dense(hidden_size, output_size)

    def construct(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step_output = lstm_out[:, -1, :]
        output = self.fc(last_time_step_output)
        return output

# 设置运行模式
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

# 参数设置
batch_size = 32
seq_length = 10
input_size = 5
hidden_size = 64
num_layers = 2
output_size = 1

# 创建随机数据
x_data = np.random.randn(batch_size, seq_length, input_size).astype(np.float32)
y_data = np.random.randn(batch_size, output_size).astype(np.float32)

# 转换为Tensor
x = Tensor(x_data)
y = Tensor(y_data)

# 使用 GeneratorDataset 创建数据集
def data_generator():
    for i in range(x.shape[0]):
        yield (x[i], y[i])  # 按行遍历 x 和 y 数据

# 使用 GeneratorDataset 创建数据集
dataset = ds.GeneratorDataset(data_generator, column_names=["x", "y"])

# 设置批次大小
dataset = dataset.batch(batch_size)

# 初始化模型、损失函数和优化器
model = LSTMModel(input_size, hidden_size, num_layers, output_size)
loss_fn = nn.MSELoss()
optimizer = nn.Adam(params=model.get_parameters(), learning_rate=0.001)

# 创建 Model 实例
model_wrapper = Model(model, loss_fn=loss_fn, optimizer=optimizer)

# 训练模型
def train(model_wrapper, dataset, epochs=10):
    for epoch in range(epochs):
        loss = model_wrapper.train(1, dataset)  # 传递数据集
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss}")

train(model_wrapper, dataset, epochs=10)

# 保存模型
checkpoint_file = "lstm_model.ckpt"
save_checkpoint(model, checkpoint_file)
print(f"模型已保存至 {checkpoint_file}")

# 加载模型
loaded_model = LSTMModel(input_size, hidden_size, num_layers, output_size)
param_dict = load_checkpoint(checkpoint_file)
load_param_into_net(loaded_model, param_dict)
print("模型已加载")

