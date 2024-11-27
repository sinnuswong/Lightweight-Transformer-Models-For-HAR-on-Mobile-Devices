import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# 配置
window_size = 130
feature_size = 6
num_classes = 3
hidden_size = 256
L2 = 0.000001
learning_rate = 0.001
batch_size = 128
epochs = 200


# 定义模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.dense2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # LSTM layer
        x, _ = self.lstm(x)
        # We only care about the output of the last time step
        x = x[:, -1, :]
        # Fully connected layers
        x = self.relu(self.dense1(x))
        x = self.relu(self.dense2(x))
        x = self.softmax(self.output(x))
        return x


# 初始化模型
model = LSTMModel(input_size=feature_size, hidden_size=hidden_size, num_classes=num_classes)

# 打印模型结构
print(model)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=L2)

# 准备数据
import myutil_huawei  # 自定义数据处理模块

data, labels = myutil_huawei.build_badminton_kill_data(
    kill_data_path='/Users/sinnus/Desktop/ActivityData/badminton/c130/1020left/kill_high_long_hit')
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
torch.save(model.state_dict(), "lstm_model.pth")
