import torch
import torch.nn as nn

class GePrompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, top_k=None, batchwise_prompt=False):
        super().__init__()

        self.length = length
        self.embed_dim = embed_dim
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt

        self.task_embed = None
        task_len = self.task_embed.shape[1]
        self.generation_layer_1 = nn.Linear(768, 256, bias=True)
        self.generation_activation = nn.Relu()
        self.generation_layer_2 = nn.Linear(256, 768*self.top_k, bias=True)

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    
    def forward(self, x_embed, prompt_mask=None, cls_features=None):
        out = dict()

        if self.prompt_pool:
            if self.embedding_key == 'mean':
                x_embed_mean = torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'max':
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            elif self.embedding_key == 'mean_max':
                x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'cls':
                if cls_features is None:
                    x_embed_mean = torch.max(x_embed, dim=1)[0] # B, C
                else:
                    x_embed_mean = cls_features
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")

            prompt_norm = self.l2_normalize(self.prompt_key, dim=1) # Pool_size, C
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=1) # B, C

            similarity = torch.matmul(x_embed_norm, prompt_norm.t()) # B, Pool_size

            idx = prompt_mask # B, top_k

        x_task_embed = torch.cat((x_embed, self.task_embed), dim=1)
        a = self.generation_layer_1(x_task_embed)
        b = self.generation_activation(a)
        c = self.generation_layer_2(b)