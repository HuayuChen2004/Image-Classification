import torch
import torch.nn as nn
import torch.optim as optim

# 设置随机种子，保证结果可复现
torch.manual_seed(0)

# 1. 构造训练数据：y = 2x + 3 + 噪声
num_samples = 100
x = torch.linspace(-10, 10, num_samples).unsqueeze(1)           # 形状 (100, 1)
noise = torch.randn(num_samples, 1) * 2.0                       # 噪声
y = 2.0 * x + 3.0 + noise                                       # 形状 (100, 1)

# 2. 定义线性回归模型（一个线性层）
class SimpleLinearRegression(nn.Module):
    def __init__(self):
        super(SimpleLinearRegression, self).__init__()
        self.linear = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        return self.linear(x)

model = SimpleLinearRegression()

# 3. 定义损失函数和优化器
criterion = nn.MSELoss()              # 均方误差损失
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4. 训练循环
num_epochs = 200
for epoch in range(1, num_epochs + 1):
    model.train()
    optimizer.zero_grad()

    outputs = model(x)                # 前向传播
    loss = criterion(outputs, y)      # 计算损失
    loss.backward()                   # 反向传播
    optimizer.step()                  # 更新参数

    if epoch % 20 == 0:
        w = model.linear.weight.item()
        b = model.linear.bias.item()
        print(f"Epoch [{epoch}/{num_epochs}]  Loss: {loss.item():.4f}    w: {w:.4f}    b: {b:.4f}")

# 5. 训练结束后，打印最终结果
w_final = model.linear.weight.item()
b_final = model.linear.bias.item()
print(f"\n训练结束：\n学到的权重 w = {w_final:.4f}, 偏置 b = {b_final:.4f}")
