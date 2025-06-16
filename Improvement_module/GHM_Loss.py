import torch
import torch.nn.functional as F
from ultralytics import YOLO


class GHM_Loss(torch.nn.Module):
    def __init__(self, bins=10, momentum=0):
        super(GHM_Loss, self).__init__()
        self.bins = bins
        self.momentum = momentum
        self.edges = [i / bins for i in range(bins + 1)]
        self.edges[-1] += 1e-6
        if momentum > 0:
            self.acc_sum = [0.0 for _ in range(bins)]
        self.valid_bins = bins

    def forward(self, logits, target):
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(logits)

        logits = F.softmax(logits, dim=1)

        # 计算每个样本的梯度
        g = torch.abs(logits[:, 1] - target.float())

        total = target.numel()

        if mmt > 0:
            self.acc_sum = [mmt * acc_sum + (1 - mmt) * g.sum().item() for acc_sum, g in zip(self.acc_sum, g.unbind())]

        # 计算样本所在梯度区间
        inds = torch.bucketize(g, torch.tensor(edges)).clamp_(min=0, max=self.valid_bins - 1)

        # 计算权重
        weights = total / (self.valid_bins * inds.numel())
        weights /= weights.sum()

        # 计算 GHM 损失
        loss = F.cross_entropy(logits, target, weight=weights, reduction='none')

        return loss.mean()

# 使用示例
# 定义模型和优化器
model = YOLO()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = GHM_Loss()

# 训练循环中使用 GHM 损失
# for epoch in range(num_epochs):
#     for inputs, labels in train_loader:
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
