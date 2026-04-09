"""
dgcnn.py - Dynamic Graph CNN for point cloud classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class STN3d(nn.Module):
    """Spatial transformer that learns a 3x3 rotation to align input points"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 512, 1)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 9)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(256)

    def forward(self, x):
        B = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2)[0]
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.fc2(x)
        iden = torch.eye(3, dtype=x.dtype, device=x.device).flatten().unsqueeze(0).repeat(B, 1)
        return (x + iden).view(-1, 3, 3)


def knn(x, k):
    """Find k nearest neighbours using pairwise squared distances"""
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    return (-xx - inner - xx.transpose(2, 1)).topk(k=k, dim=-1)[1]


def get_graph_feature(x, k=20, idx=None):
    """Build edge features by concatenating each point with its neighbour differences"""
    B, C, N = x.shape
    device = x.device

    if idx is None:
        idx = knn(x, k)

    idx_base = torch.arange(0, B, device=device).view(-1, 1, 1) * N
    idx = (idx + idx_base).view(-1)
    x = x.transpose(2, 1).contiguous()
    neighbours = x.view(B * N, -1)[idx].view(B, N, k, C)
    x = x.view(B, N, 1, C).repeat(1, 1, k, 1)
    return torch.cat([x, neighbours - x], dim=3).permute(0, 3, 1, 2).contiguous()


class EdgeConv(nn.Module):
    """Edge convolution that rebuilds the graph dynamically at each layer"""
    def __init__(self, in_ch, out_ch, k=20):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch * 2, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, x):
        return self.conv(get_graph_feature(x, self.k)).max(dim=-1)[0]


class DGCNN(nn.Module):
    """DGCNN with spatial transformer and combined max+avg pooling"""
    def __init__(self, num_classes=10, k=20, emb_dims=1024):
        super().__init__()
        # spatial transformer for input alignment
        self.stn = STN3d()
        # EdgeConv layers
        self.ec1 = EdgeConv(3, 64, k)
        self.ec2 = EdgeConv(64, 64, k)
        self.ec3 = EdgeConv(64, 128, k)
        self.ec4 = EdgeConv(128, 256, k)

        # fuse all EdgeConv outputs into a single embedding
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, emb_dims, 1, bias=False),
            nn.BatchNorm1d(emb_dims),
            nn.LeakyReLU(0.2, True)
        )

        # classifier
        self.cls = nn.Sequential(
            nn.Linear(emb_dims * 2, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.6),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.6),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # align input points using learned spatial transform
        x_t = x.transpose(2, 1)
        trans = self.stn(x_t)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)

        x1 = self.ec1(x)
        x2 = self.ec2(x1)
        x3 = self.ec3(x2)
        x4 = self.ec4(x3)

        # multi-scale aggregation captures both fine and coarse structure
        x = self.conv5(torch.cat([x1, x2, x3, x4], dim=1))

        # max pooling captures strongest features, avg pooling captures overall shape
        x = torch.cat([
            F.adaptive_max_pool1d(x, 1).squeeze(-1),
            F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        ], dim=1)

        return self.cls(x), None

    def get_loss(self, pred, target, trans_feat=None):
        """Cross entropy with label smoothing to reduce overconfidence"""
        return F.cross_entropy(pred, target, label_smoothing=0.2)
