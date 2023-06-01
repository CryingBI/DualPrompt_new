import torch
import torch.nn as nn


class promptGe(nn.Module):
    def __init__(self, length=1, embed_dim=768):
        super().__init__()

        self.length = length
        self.embed_dim = embed_dim
    
        self.task_embed_layer = nn.Embedding(10, self.embed_dim)

        #Generation network
        self.generation_layer_1 = nn.Linear(768, 256, bias=True)
        self.generation_activation = nn.ReLU()
        self.generation_layer_2 = nn.Linear(256, 768*5, bias=True)

    def forward(self, x_embed, task_id=None):
        cuda = torch.device('cuda')
        out = dict()

        task_embed = torch.Tensor([task_id])
        m = task_embed.expand(x_embed.shape[0], -1).long()   
        n = self.task_embed_layer(m)

        x_task_embed = torch.cat((x_embed, n), dim=1).to(device=cuda)
        a = self.generation_layer_1(x_task_embed)
        b = self.generation_activation(a)
        batched_prompt_raw = self.generation_layer_2(b)     # B, length, C * 5

        # prompt_1 = batched_prompt_raw[:, :, :768]
        # prompt_2 = batched_prompt_raw[:, :, 768:768*2]
        # prompt_3 = batched_prompt_raw[:, :, 768*2:768*3]
        # prompt_4 = batched_prompt_raw[:, :, 768*3:768*4]
        # prompt_5 = batched_prompt_raw[:, :, 768*4:768*5]

        out['total_prompt_len'] = 5
        out['prompted_embedding'] = torch.cat([batched_prompt_raw, x_embed], dim=1)
        return out
        