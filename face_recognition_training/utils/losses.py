import torch
from torch import nn
import torch.nn.functional as F
import math


class NonLinearHeader(nn.Module):
    """non linear header between representation and loss"""

    def __init__(self, emb_dim, proj_dim) -> None:
        super().__init__()
        # self.w1 = nn.Linear(emb_dim, emb_dim, bias=True)
        # self.bn1 = nn.BatchNorm1d(emb_dim)
        self.relu = nn.ReLU()
        self.w2 = nn.Linear(emb_dim, proj_dim, bias=False)
        self.bn2 = nn.BatchNorm1d(proj_dim)

    def forward(self, x):
        # x = self.w1(x)
        # x = self.bn1(x)
        x = self.relu(x)
        x = self.w2(x)
        return self.bn2(x)


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.50):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.kernel = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.normal_(self.kernel, std=0.01)

    def forward(self, embbedings, label):
        embbedings = l2_norm(embbedings, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(
            index.size()[0], cos_theta.size()[1], device=cos_theta.device
        )
        m_hot.scatter_(1, label[index, None], self.m)
        cos_theta.acos_()
        cos_theta[index] += m_hot
        cos_theta.cos_().mul_(self.s)
        return cos_theta


class CosFace(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.35):
        super(CosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.kernel = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.normal_(self.kernel, std=0.01)

    def forward(self, embbedings, label):
        embbedings = l2_norm(embbedings, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(
            index.size()[0], cos_theta.size()[1], device=cos_theta.device
        )
        m_hot.scatter_(1, label[index, None], self.m)
        cos_theta[index] -= m_hot
        ret = cos_theta * self.s
        return ret


class ElasticCosFace(nn.Module):
    def __init__(
        self, in_features, out_features, s=64.0, m=0.35, std=0.0125, plus=False
    ):
        super(ElasticCosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.kernel = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.normal_(self.kernel, std=0.01)
        self.std = std
        self.plus = plus

    def forward(self, embbedings, label):
        embbedings = l2_norm(embbedings, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(
            index.size()[0], cos_theta.size()[1], device=cos_theta.device
        )
        margin = torch.normal(
            mean=self.m,
            std=self.std,
            size=label[index, None].size(),
            device=cos_theta.device,
        )  # Fast converge .clamp(self.m-self.std, self.m+self.std)
        if self.plus:
            with torch.no_grad():
                distmat = cos_theta[index, label.view(-1)].detach().clone()
                _, idicate_cosie = torch.sort(distmat, dim=0, descending=True)
                margin, _ = torch.sort(margin, dim=0)
            m_hot.scatter_(1, label[index, None], margin[idicate_cosie])
        else:
            m_hot.scatter_(1, label[index, None], margin)
        cos_theta[index] -= m_hot
        ret = cos_theta * self.s
        return ret


class AdaFace(nn.Module):
    def __init__(
        self,
        embedding_size=512,
        classnum=70722,
        m=0.4,
        h=0.333,
        s=64.0,
        t_alpha=1.0,
    ):
        super(AdaFace, self).__init__()
        self.classnum = classnum
        self.kernel = nn.Parameter(torch.Tensor(embedding_size, classnum))

        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.m = m
        self.eps = 1e-3
        self.h = h
        self.s = s

        # ema prep
        self.t_alpha = t_alpha
        self.register_buffer("t", torch.zeros(1))
        self.register_buffer("batch_mean", torch.ones(1) * (20))
        self.register_buffer("batch_std", torch.ones(1) * 100)

        print("\n\AdaFace with the following property")
        print("self.m", self.m)
        print("self.h", self.h)
        print("self.s", self.s)
        print("self.t_alpha", self.t_alpha)

    def forward(self, embbedings, norms, label):

        kernel_norm = l2_norm(self.kernel, axis=0)
        cosine = torch.mm(embbedings, kernel_norm)
        cosine = cosine.clamp(-1 + self.eps, 1 - self.eps)  # for stability

        safe_norms = torch.clip(norms, min=0.001, max=100)  # for stability
        safe_norms = safe_norms.clone().detach()

        # update batchmean batchstd
        with torch.no_grad():
            mean = safe_norms.mean().detach()
            std = safe_norms.std().detach()
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            self.batch_std = std * self.t_alpha + (1 - self.t_alpha) * self.batch_std

        margin_scaler = (safe_norms - self.batch_mean) / (
            self.batch_std + self.eps
        )  # 66% between -1, 1
        margin_scaler = margin_scaler * self.h  # 68% between -0.333 ,0.333 when h:0.333
        margin_scaler = torch.clip(margin_scaler, -1, 1)
        # ex: m=0.5, h:0.333
        # range
        #       (66% range)
        #   -1 -0.333  0.333   1  (margin_scaler)
        # -0.5 -0.166  0.166 0.5  (m * margin_scaler)

        # g_angular
        m_arc = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_arc.scatter_(1, label.reshape(-1, 1), 1.0)
        g_angular = self.m * margin_scaler * -1
        m_arc = m_arc * g_angular
        theta = cosine.acos()
        theta_m = torch.clip(theta + m_arc, min=self.eps, max=math.pi - self.eps)
        cosine = theta_m.cos()

        # g_additive
        m_cos = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_cos.scatter_(1, label.reshape(-1, 1), 1.0)
        g_add = self.m + (self.m * margin_scaler)
        m_cos = m_cos * g_add
        cosine = cosine - m_cos

        # scale
        scaled_cosine_m = cosine * self.s
        return scaled_cosine_m
