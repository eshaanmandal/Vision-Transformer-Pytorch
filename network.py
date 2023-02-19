import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary

# Residual Connection
class SkipConnection(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x
    
# Feed forward neural network
class FFNN(nn.Module):
    def __init__(
            self,
            emb_size,
            expansion: int = 4,
            drop_p: float = 0
    ):
        super().__init__()
        self.ffnn = nn.Sequential(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size)
        )

    def forward(self, x):
        return self.ffnn(x)
    
class PatchEmbedding(nn.Module):
    def __init__(
      self,
      in_channels: int = 3,
      patch_size: int = 16,
      emb_size: int = 768,
      image_size: int = 224      
    ):
        super().__init__()

        # Projection layer
        # Converts the image into patches aka flattened layers
        self.projection = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=emb_size,
                kernel_size=patch_size,
                stride=patch_size
            ),
            Rearrange('b e h w -> b (h w) e')
        )

        # A cls token that is added in front of the flattened
        # layer this is learnt by the neural network
        self.cls_token = nn.Parameter(torch.randn(1,1,emb_size))

        #A position embedding vector that is used for storing the 
        # position of every flattened vector that is generated 
        self.position = nn.Parameter(
            torch.randn((image_size//patch_size)**2+1, emb_size)
        )

    def forward(self, x):
        b, _, _, _ = x.shape # getting the batch size of the input
        x = self.projection(x)

        cls_token = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_token, x], dim=1)
        x += self.position
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            emb_size: int = 512,
            num_heads: int = 8,
            dropout: float = 0
    ):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads=num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)

        self.attention_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x, mask=None):
        # Rearranging the keys, queries and values to divide
        # everything in a multi-headed attention
        keys = rearrange(
            self.keys(x),
            'b n (h d) -> b h n d',
            h = self.num_heads
        )

        queries = rearrange(
            self.queries(x),
            'b n (h d) -> b h n d',
            h = self.num_heads
        )

        values = rearrange(
            self.values(x),
            'b n (h d) -> b h n d',
            h = self.num_heads
        )

        energy = torch.einsum(
            'bhqd, bhkd -> bhqk',
            queries,
            keys
        )

        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        # scaling the components of the matrix multiplication
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1)/scaling
        att = self.attention_drop(att)

        # summing over the third axis
        out = torch.einsum('bhal, bhlv -> bhav', att, values)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.projection(out)
        return out
    
class TransformerEncoderBlock(nn.Sequential):
    def __init__(
      self, 
      emb_size: int = 768,
      drop_p: float = 0,
      forward_expansion: int = 4,
      forward_drop_p: float = 0,
      **kwargs      
    ):
        super().__init__(
            SkipConnection(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    MultiHeadAttention(emb_size, **kwargs),
                    nn.Dropout(drop_p)
                )
            ),
            SkipConnection(
                nn.Sequential(
                   nn.LayerNorm(emb_size),
                   FFNN(
                        emb_size,
                        forward_expansion,
                        forward_drop_p
                   ),
                   nn.Dropout(drop_p)     
                )
            )
        )

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth=12, **kwargs):
        super().__init__(
            *[
                TransformerEncoderBlock(**kwargs)
                for _ in range(depth)
            ]
        )

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes: int = 1000):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )

# Finally our vision transformer network

class ViT(nn.Sequential):
    def __init__(
            self,
            in_channels: int = 3,
            patch_size: int = 16,
            emb_size: int = 768,
            img_size: int = 224,
            depth: int = 12,
            n_classes: int = 1000,
            **kwargs
    ):
        super().__init__(
            PatchEmbedding(
                in_channels=in_channels,
                patch_size=patch_size,
                emb_size=emb_size,
                image_size=img_size
            ),
            TransformerEncoder(depth=depth),
            ClassificationHead(emb_size=emb_size, n_classes=n_classes)
        )

# print(summary(ViT(), (3, 224, 224), device='cpu'))
