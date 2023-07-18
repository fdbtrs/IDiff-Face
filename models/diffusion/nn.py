import math

import torch
from einops import rearrange

from utils.checkpoint import checkpoint
from utils.helpers import zero_module


class SpatialEmbeddingCrossAttentionBlock(torch.nn.Module):
    """
    Block that applies self-attention to the image input and cross-attention based on an additional context embedding
    vector. Before applying the attention layers, both of these input tensors are transformed into a suitable shape
    (B, token_length, n_tokens) for the attention mechanism: The image-like tensor of shape (B, C, H, W) is mapped to
    shape (B, L, H*W), where L is the number of inner channels of this block. The embedding tensor of shape (B, D) is
    first mapped to (B, D * d) where D is the context_dim and d is a chosen constant n_context_tokens. After that, the
    tensor of shape (B, D * d) is reshaped to (B, D, d).
    """

    def __init__(self, in_channels, context_dim, inner_channels=None, n_context_tokens=None,
                 n_heads=4, head_channels=32, use_checkpoint=True):
        super().__init__()

        self.in_channels = in_channels
        self.context_dim = context_dim

        if n_heads is None or n_heads <= 0:
            n_heads = in_channels // head_channels

        self.n_heads = n_heads
        self.head_channels = head_channels
        self.inner_channels = n_heads * head_channels if inner_channels is None else inner_channels

        self.x_proj_in = torch.nn.Conv2d(in_channels, self.inner_channels, 1)
        self.x_proj_out = zero_module(torch.nn.Conv2d(self.inner_channels, in_channels, 1))

        self.self_attention = MultiHeadAttention(in_channels=self.inner_channels, n_heads=n_heads,
                                                 head_channels=head_channels, use_checkpoint=use_checkpoint)

        d = head_channels if n_context_tokens is None else n_context_tokens
        self.c_proj_in = torch.nn.Linear(context_dim, self.context_dim * d)

        self.cross_attention = MultiHeadAttention(in_channels=self.inner_channels, key_value_channels=context_dim,
                                                  n_heads=n_heads, head_channels=head_channels,
                                                  use_checkpoint=use_checkpoint)

        self.x_norm = torch.nn.GroupNorm(32, self.inner_channels)

        # TODO: c_norm might not be necessary
        self.c_norm = torch.nn.GroupNorm(32, self.context_dim)

        self.ff = torch.nn.Sequential(
            torch.nn.Linear(self.inner_channels, self.inner_channels * 4),
            torch.nn.GELU(),
            torch.nn.Linear(self.inner_channels * 4, self.inner_channels)
        )

    def forward(self, x, c):
        # x is batch of image tensors or feature maps
        # c is batch of context embedding vectors

        b, n, h, w = x.shape
        x_in = x

        # turn image to appropriate shape
        x = self.x_proj_in(x)
        x = rearrange(x, "b n h w -> b n (h w)")

        # turn embedding vector to appropriate shape
        c = self.c_proj_in(c)
        c = rearrange(c, "b (n d) -> b n d", n=self.context_dim)
        c = self.c_norm(c)

        # apply self-attention
        x = self.self_attention(self.x_norm(x)) + x

        # apply cross-attention
        x = self.cross_attention(self.x_norm(x), c) + x

        # apply tiny feedforward network
        x = rearrange(self.ff(rearrange(self.x_norm(x), "b n d -> b d n").contiguous()), "b d n -> b n d").contiguous() + x

        # turn result back to an image-like shape
        x = rearrange(x, "b n (h w) -> b n h w", h=h, w=w)
        x = self.x_proj_out(x)

        return x + x_in


class SpatialSelfAttentionBlock(torch.nn.Module):

    def __init__(self, in_channels, n_heads=4, head_channels=32, use_checkpoint=False):
        super().__init__()
        self.in_channels = in_channels
        
        if n_heads is None or n_heads <= 0:
            n_heads = in_channels // head_channels

        self.head_channels = head_channels

        self.attention = MultiHeadAttention(in_channels=in_channels, n_heads=n_heads, head_channels=head_channels,
                                            use_checkpoint=use_checkpoint)

        self.norm = torch.nn.GroupNorm(32, in_channels)

    def forward(self, x, c):
        b, c, h, w = x.shape
        x_in = x

        # turn image to appropriate shape
        x = self.norm(x)
        x = rearrange(x, "b c h w -> b c (h w)")

        # apply attention
        x = self.attention(x)

        # turn result back to an image-like shape
        x = rearrange(x, "b c (h w) -> b c h w", h=h)

        return x + x_in


class MultiHeadAttention(torch.nn.Module):

    def __init__(self, in_channels, key_value_channels=None, n_heads=4, head_channels=32, use_checkpoint=False):
        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.n_heads = n_heads
        self.head_channels = head_channels

        self.inner_dim = n_heads * head_channels

        if key_value_channels is None:
            key_value_channels = in_channels

        self.to_q = torch.nn.Conv1d(in_channels, self.inner_dim, 1)
        self.to_k = torch.nn.Conv1d(key_value_channels, self.inner_dim, 1)
        self.to_v = torch.nn.Conv1d(key_value_channels, self.inner_dim, 1)

        self.proj_out = torch.nn.Conv1d(self.inner_dim, in_channels, 1)

    def forward(self, x, c: torch.Tensor = None):
        c = x if c is None else c

        if self.use_checkpoint:
            return checkpoint(self._forward, (x, c), self.parameters(), True)
        else:
            return self._forward(x, c)

    def _forward(self, x, c):
        q, k, v = self.to_q(x), self.to_k(c), self.to_v(c)
        x = self.qkv_attention(q, k, v, n_heads=self.n_heads)
        return self.proj_out(x)

    @staticmethod
    def qkv_attention(q, k, v, n_heads):
        # shapes of query, key and value tensors
        bq, wq, lq = q.shape
        bk, wk, lk = k.shape
        bv, wv, lv = v.shape

        # check if batch_size and width are the same for all of them
        assert bq == bk == bv
        assert wq == wk == wv

        width = wq
        bs = bq

        assert width % n_heads == 0

        # reshape each from (bs, width, length) to (bs * n_heads, width // n_heads, length)
        q = q.reshape(bs * n_heads, width // n_heads, lq)
        k = k.reshape(bs * n_heads, width // n_heads, lk)
        v = v.reshape(bs * n_heads, width // n_heads, lv)

        # more stable with f16 than dividing afterwards
        scale = 1 / math.sqrt(math.sqrt(width // n_heads))

        # (bs * n_heads, channels, lq), (bs * n_heads, channels, lq) -> (bs * n_heads, lq, lk)
        weight = torch.einsum("b c t , b c s -> b t s", q * scale, k * scale)

        # (bs * n_heads, lq, lk) -> (bs * n_heads, lq, lk)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)

        # (bs * n_heads, lq, lk), (bs * n_heads, channels, lv) -> (bs * n_heads, channels, lq)
        a = torch.einsum("b t s, b c s -> b c t", weight, v)

        # (bs * n_heads, channels, lq) -> (bs, width, lq)
        return rearrange(a, "(b n) c t -> b (n c) t", b=bs).contiguous()


if __name__ == "__main__":

    """
    x = torch.ones((3, 128, 16, 16))
    c = torch.ones((3, 256))
    
    block = SpatialEmbeddingCrossAttentionBlock(head_channels=32, n_heads=4, in_channels=128, context_dim=256)

    trainable_params, _, total_params = count_model_parameters(block)
    print(f"#Params Model: {trainable_params} (Total: {total_params})")

    block = SpatialTransformer(in_channels=128, n_heads=4, d_head=32, context_dim=256)

    trainable_params, _, total_params = count_model_parameters(block)
    print(f"#Params Model: {trainable_params} (Total: {total_params})")

    # with open("SpatialTransformer.txt", "w") as f:
    #    f.write(str(block))
    """
