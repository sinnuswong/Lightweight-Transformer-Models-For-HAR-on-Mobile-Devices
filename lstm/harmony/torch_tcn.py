import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import weight_norm


# 定义 TCN Block
class TCNBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, dilation, padding, dropout):
        super(TCNBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(input_channels, output_channels, kernel_size, stride=stride, dilation=dilation, padding=padding)
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = weight_norm(
            nn.Conv1d(output_channels, output_channels, kernel_size, stride=stride, dilation=dilation, padding=padding)
        )
        self.net = nn.Sequential(self.conv1, self.relu, self.dropout, self.conv2, self.relu, self.dropout)
        self.downsample = nn.Conv1d(input_channels, output_channels, 1) if input_channels != output_channels else None
        self.relu_final = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu_final(out + res)


# 定义 TCN 模型
class TCNModel(nn.Module):
    def __init__(self, input_size, num_channels, num_classes, kernel_size, dropout):
        super(TCNModel, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(
                TCNBlock(
                    in_channels, out_channels, kernel_size=kernel_size, stride=1, dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size // 2, dropout=dropout
                )
            )
        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(num_channels[-1], num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # TCN expects (batch_size, input_channels, sequence_length)
        x = x.transpose(1, 2)  # (batch, sequence, features) -> (batch, features, sequence)
        x = self.network(x)
        x = x.mean(dim=2)  # Global Average Pooling (optional)
        x = self.output_layer(x)
        return self.softmax(x)

# 配置
# window_size = 130
# feature_size = 6
# num_classes = 3
# hidden_size = 256
# L2 = 0.000001
# learning_rate = 0.001
# batch_size = 128
# epochs = 200

# 配置参数
feature_size = 6

input_size = feature_size
num_classes = 3
num_channels = [128, 256, 256]  # TCN hidden channels
kernel_size = 3
dropout = 0.2
learning_rate = 0.0001
batch_size = 64
epochs = 50

# 初始化模型
model = TCNModel(input_size=input_size, num_channels=num_channels, num_classes=num_classes,
                 kernel_size=kernel_size, dropout=dropout)

# 打印模型结构
print(model)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 数据准备
import myutil_huawei  # 自定义模块
data, labels = myutil_huawei.build_badminton_kill_data(kill_data_path='/Users/sinnus/Desktop/ActivityData/badminton/c130/1020left/kill_high_long_hit')
data = torch.tensor(data, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.long)

# 划分数据集
X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

# 创建 DataLoader
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    train_accuracy = 100 * correct / total
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    model.eval()

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += targets.size(0)
            val_correct += (predicted == targets).sum().item()

    val_accuracy = 100 * val_correct / val_total
    print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {running_loss / len(train_loader):.4f}, "
          f"Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss / len(val_loader):.4f}, "
          f"Val Acc: {val_accuracy:.2f}%")

# 保存模型
torch.save(model.state_dict(), "tcn_model.pth")
