# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import torch
import torch.nn as nn
from utils.losses import NonLinearHeader
import numpy as np


@torch.no_grad()
def sim_matrix(a, b, eps=1e-8):
    """
    cosine similarity between each element in two matrices
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(
        self,
        base_encoder,
        dim=128,
        K=65536,
        m=0.999,
        T=0.07,
        mlp=False,
        margin=0.0,
        queue_type="normal",
        reset_queue_every=5,
        morphing=False,
        dropout=0.0,
        domain_ada=False,
    ):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.dim = dim
        self.mlp = mlp
        self.margin = margin
        self.queue_type = queue_type
        self.queue_full = False
        self.reset_queue_every = reset_queue_every
        self.reset_queue = True
        self.epoch = 0
        self.morphing = morphing
        self.domain_ada = domain_ada

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_features=dim, dropout=dropout)
        self.encoder_k = base_encoder(num_features=dim, dropout=dropout)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]  # dim = 25088
            self.encoder_q.fc = nn.Sequential(
                self.encoder_q.fc, nn.ReLU(), nn.Linear(dim, dim)
            )
            self.encoder_k.fc = nn.Sequential(
                self.encoder_k.fc, nn.ReLU(), nn.Linear(dim, dim)
            )

        # if mlp:
        #     logging.info("=> Model with MLP header")
        #     self.header_q = NonLinearHeader(emb_dim=dim, proj_dim=dim)
        #     self.header_k = NonLinearHeader(emb_dim=dim, proj_dim=dim)

        #     for param_q, param_k in zip(self.header_q.parameters(), self.header_k.parameters()):
        #         param_k.data.copy_(param_q.data)  # initialize
        #         param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("labels", torch.zeros(K, dtype=torch.int))

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

        if False:  # self.mlp:
            for param_q, param_k in zip(
                self.header_q.parameters(), self.header_k.parameters()
            ):
                param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _reset_queue(self):
        if self.reset_queue and self.epoch % self.reset_queue_every == 0:
            self.queue[:, :] = nn.functional.normalize(
                torch.randn(self.dim, self.K), dim=0
            )
            self.labels[:] = torch.zeros(self.K, dtype=torch.int)
            self.queue_ptr[0] = 0
            self.queue_full = False
            self.reset_queue = False
            logging.info("Reset queue")
        elif self.epoch % self.reset_queue_every != 0:
            self.reset_queue = True

    @torch.no_grad()
    def _sim_queue(self, keys):
        if self.queue_type not in ["simQ", "dissimQ"]:
            return
        # cosine similarity between each key and queue sample
        cos_sims = sim_matrix(keys, self.queue.T)
        replace_idx = []
        if self.queue_type == "simQ":
            # get idx of most similar queue sample for each key
            _, replace_idx = torch.max(cos_sims, dim=1)
        else:
            _, replace_idx = torch.min(cos_sims, dim=1)
        # replace most similar or most dissimilar queue samples
        self.queue[:, replace_idx] = keys.T

    @torch.no_grad()
    def _center_queue(self, keys):
        cos_sims = sim_matrix(keys, self.queue.T)
        _, max_idx = torch.max(cos_sims, dim=1)
        class_centers = (self.queue[:, max_idx] + keys.T) / 2
        self.queue[:, max_idx] = class_centers
        return max_idx

    @torch.no_grad()
    def _class_queue(self, keys, labs):
        self._reset_queue()
        # only replace the queue samples where queue label == key label
        occurs_in_q = torch.stack([(self.labels == i) for i in labs], dim=0)
        for i, occurrence in enumerate(occurs_in_q):
            idx = occurrence.nonzero(as_tuple=False)
            # if key label does not occur queue labels -> skip
            if idx.shape[0] == 0:
                continue
            # replace queue sample with average of old queue sample and new key
            old_new_emb = torch.stack([self.queue[:, idx[0][0]], keys[i]], dim=0)
            class_center = torch.mean(old_new_emb, dim=0)
            self.queue[:, idx[0][0]] = nn.functional.normalize(class_center, dim=0)

    @torch.no_grad()
    def _fill_queue(self, keys, labs, batch_size, ptr):
        # fill queue and labels until full
        self.queue[:, ptr : ptr + batch_size] = keys.T
        if len(labs) > 0:
            self.labels[ptr : ptr + batch_size] = labs

        if ptr + batch_size >= self.K:
            self.queue_full = True
            logging.info("Queue filled, now replace")

        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labs=[]):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        if self.queue_type == "normal":  # default queue
            # replace the keys at ptr (dequeue and enqueue)
            self.queue[:, ptr : ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.K  # move pointer
            self.queue_ptr[0] = ptr
            return np.arange(ptr, ptr + batch_size, 1)

        if len(labs) > 0:
            labs = concat_all_gather(labs)

        if not self.queue_full:
            self._fill_queue(keys, labs, batch_size, ptr)
            return

        if self.queue_type == "classQ" and self.queue_full:
            self._class_queue(keys, labs)
        elif self.queue_type == "centerQ" and self.queue_full:
            return self._center_queue(keys)

        self._sim_queue(keys)

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(
        self, im_q, im_k, morphed_q=[], label=[], weights=[], only_emb=False, epoch=0
    ):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
            morphed_q: batch of morphed query
            label: a batch of key labels (indices)
            weights: weights from morphing
            epoch: current epoch
        Output:
            logits, targets
        """
        self.epoch = epoch
        # compute query features
        if self.morphing:
            m_q = self.encoder_q(morphed_q)
            # if self.mlp:
            #     m_q = self.header_q(m_q)
            m_q = nn.functional.normalize(m_q, dim=1)
        else:
            q = self.encoder_q(im_q)  # queries: NxC
            # if self.mlp:
            #     q = self.header_q(q)
            q = nn.functional.normalize(q, dim=1)
        if only_emb:
            return q

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            # if self.mlp:
            #     k = self.header_k(k)
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

            if len(morphed_q) > 0:
                im_q, idx_unshuffle = self._batch_shuffle_ddp(im_q)
                q = self.encoder_k(im_q)  # keys: NxC
                # if self.mlp:
                #     q = self.header_k(q)
                q = nn.functional.normalize(q, dim=1)
                q = self._batch_unshuffle_ddp(q, idx_unshuffle)

        queue = self.queue.clone().detach()
        # dequeue and enqueue
        max_idx = self._dequeue_and_enqueue(k, label)
        if max_idx is not None:
            mask = torch.ones(self.K, dtype=torch.bool)
            mask[max_idx] = False
            queue = queue[:, mask]

        # new morph
        if self.morphing:
            m_q_expand = torch.repeat_interleave(m_q, 2, dim=0)
            l_pos = torch.einsum("nc,nc->n", [m_q_expand, q])
            l_pos = l_pos * weights
            l_pos = l_pos.unsqueeze(-1)

            l_neg = torch.einsum("nc,ck->nk", [m_q_expand, queue])
            logits = torch.cat([l_pos, l_neg], dim=1)

            logits /= self.T

            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
            return logits, labels

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1) - self.margin

        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, queue])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        if self.domain_ada:
            return logits, labels, q
        return logits, labels


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
