import mindspore
from mindspore import nn, context
from mindspore import Tensor
from mindspore import dtype as mstype
import numpy as np
from mindspore import save_checkpoint

class LSTMModel(nn.Cell):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Dense(hidden_size, output_size)

    def construct(self, x):
        # LSTM层返回两个输出：输出序列和最后一个隐藏状态
        lstm_out, _ = self.lstm(x)
        # 选择最后时间步的输出进行后续处理
        last_time_step_output = lstm_out[:, -1, :]
        output = self.fc(last_time_step_output)
        return output


# 假设输入数据的形状为 (batch_size, seq_length, input_size)
batch_size = 32
seq_length = 10
input_size = 5
hidden_size = 64
num_layers = 2
output_size = 1  # 回归问题示例，输出一个值

# 创建随机数据
x_data = np.random.randn(batch_size, seq_length, input_size).astype(np.float32)
y_data = np.random.randn(batch_size, output_size).astype(np.float32)

# 转换为Tensor
x = Tensor(x_data, mstype.float32)
y = Tensor(y_data, mstype.float32)


context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
loss_fn = nn.MSELoss()
optimizer = nn.Adam(params=LSTMModel(input_size, hidden_size, num_layers, output_size).get_parameters(), learning_rate=0.001)


def train(model, x, y, loss_fn, optimizer, epochs=10):
    for epoch in range(epochs):
        # 前向传播
        output = model(x)

        # 计算损失
        loss = loss_fn(output, y)

        # 反向传播和优化
        # optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 输出训练进度
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.asnumpy()}")


# 创建模型实例
model = LSTMModel(input_size, hidden_size, num_layers, output_size)

# 训练模型
train(model, x, y, loss_fn, optimizer, epochs=10)


checkpoint_file = "lstm_model.ckpt"
save_checkpoint(model, checkpoint_file)
print(f"模型已保存至 {checkpoint_file}")

# 加载模型
loaded_model = LSTMModel(input_size, hidden_size, num_layers, output_size)
param_dict = load_checkpoint(checkpoint_file)
load_param_into_net(loaded_model, param_dict)
print("模型已加载")
