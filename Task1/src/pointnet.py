"""
pointnet.py - PointNet baseline for point cloud classification
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
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        B = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # global max pooling over all points
        x = torch.max(x, 2)[0]
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        iden = torch.eye(3, dtype=x.dtype, device=x.device).flatten().unsqueeze(0).repeat(B, 1)
        return (x + iden).view(-1, 3, 3)


class STNkd(nn.Module):
    """Spatial transformer for k-dimensional feature space"""
    def __init__(self, k=64):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        B = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2)[0]
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        iden = torch.eye(self.k, dtype=x.dtype, device=x.device).flatten().unsqueeze(0).repeat(B, 1)
        return (x + iden).view(-1, self.k, self.k)


class PointNetEncoder(nn.Module):
    """Encoder with input and feature transforms followed by global max pooling"""
    def __init__(self):
        super().__init__()
        self.stn = STN3d()
        self.fstn = STNkd(k=64)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x):
        # align raw points using learned 3x3 transform
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)

        x = F.relu(self.bn1(self.conv1(x)))

        # align features in 64-d space for better invariance
        trans_feat = self.fstn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans_feat)
        x = x.transpose(2, 1)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        # global max pooling aggregates all points into one descriptor
        x = torch.max(x, 2)[0]
        return x, trans_feat


def feature_transform_regularizer(trans):
    """Penalise deviation from orthogonal to keep the transform well-behaved"""
    d = trans.size(1)
    I = torch.eye(d, device=trans.device).unsqueeze(0)
    return torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))


class PointNet(nn.Module):
    """PointNet classifier with spatial and feature transforms"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.encoder = PointNetEncoder()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = x.transpose(2, 1)
        x, trans_feat = self.encoder(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return x, trans_feat

    def get_loss(self, pred, target, trans_feat, reg_weight=0.001):
        """Cross entropy with feature transform regularisation"""
        loss = F.cross_entropy(pred, target)
        if trans_feat is not None:
            loss += reg_weight * feature_transform_regularizer(trans_feat)
        return loss
