import torch
import torch.nn as nn


class promptGe(nn.Module):
    def __init__(self, length=1, embed_dim=768):
        super().__init__()

        self.length = length
        self.embed_dim = embed_dim
    
        self.task_embed_layer = nn.Embedding(10, self.embed_dim)
        self.task_embed_layer.cuda()
        #Generation network
        self.generation_layer_1 = nn.Linear(768, 256, bias=True)
        self.generation_layer_1.cuda()

        self.generation_activation = nn.ReLU()
        self.generation_activation.cuda()
        
        self.generation_layer_2 = nn.Linear(256, 768*5, bias=True)
        self.generation_layer_2.cuda()

    def forward(self, x_embed, task_id=None):
        out = dict()

        task_embed = torch.Tensor([task_id]).to(x_embed.device)
        m = task_embed.expand(x_embed.shape[0], -1).long()
        n = self.task_embed_layer(m).to(x_embed.device)

        x_task_embed = torch.cat((x_embed, n), dim=1)
        x_task_embed_mean = x_task_embed.mean(dim=1).unsqueeze(1)
        a = self.generation_layer_1(x_task_embed_mean)
        b = self.generation_activation(a)
        batched_prompt_raw = self.generation_layer_2(b)     # B, 1, C * 5
        batched_prompt_raw = batched_prompt_raw.reshape((x_embed.shape[0], 5, 768))

        out['total_prompt_len'] = 5
        out['prompted_embedding'] = torch.cat([batched_prompt_raw, x_embed], dim=1)
        return out
        

# Batch , number of prompts, Embedding