import logging
import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from timm.models.layers import DropPath

logger = logging.getLogger(__name__)


def init_params(module, num_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(num_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
    

class GraphNodeFeature(nn.Module):
    
    def __init__(self, num_heads, num_atoms, num_in_degree, num_out_degree, hidden_dim, num_layers):
        super(GraphNodeFeature, self).__init__()
        self.num_heads = num_heads
        self.num_atoms = num_atoms
        
        self.atom_encoder = nn.Embedding(num_atoms + 1, hidden_dim, padding_idx=0)
        self.in_degree_encoder = nn.Embedding(num_in_degree, hidden_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(
            num_out_degree, hidden_dim, padding_idx=0
        )
        self.graph_token = nn.Embedding(1, hidden_dim)
        self.apply(lambda module: init_params(module, num_layers=num_layers))
    
    def forward(self, batched_data):
        x, in_degree, out_degree = (
            batched_data["x"],
            batched_data["in_degree"],
            batched_data["out_degree"],
        )
        n_graph, n_node = x.size()[:2]  # [B, T, 9]

        # node feauture + graph token
        node_feature = self.atom_encoder(x).sum(dim=-2)  # [n_graph, n_node, n_hidden]
        
        node_feature = (
                node_feature
                + self.in_degree_encoder(in_degree)  # [n_graph, n_node, n_hidden]
                + self.out_degree_encoder(out_degree)  # [n_graph, n_node, n_hidden]
        )
        
        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)
        graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)
        
        return graph_node_feature


class GraphEdgeFeature(nn.Module):
    
    def __init__(self, num_heads, num_edges, num_spatial, num_edge_dist, edge_type,
                 multi_hop_max_dist, num_layers, edge_dim):
        super(GraphEdgeFeature, self).__init__()
        self.num_heads = num_heads
        self.multi_hop_max_dist = multi_hop_max_dist
        self.edge_embedding_dim = edge_dim
        
        self.edge_encoder = nn.Embedding(num_edges + 1, edge_dim, padding_idx=0)
        self.edge_type = edge_type
        if self.edge_type == "multi_hop":
            self.edge_dis_encoder = nn.Embedding(
                num_edge_dist * edge_dim * edge_dim, 1)
        self.spatial_pos_encoder = nn.Embedding(num_spatial, edge_dim, padding_idx=0)
        
        self.graph_token_virtual_distance = nn.Embedding(1, edge_dim)
        self.apply(lambda module: init_params(module, num_layers=num_layers))
    
    def forward(self, batched_data):
        attn_bias, spatial_pos, x = (
            batched_data["attn_bias"],
            batched_data["spatial_pos"],
            batched_data["x"],
        )
        attn_bias = torch.zeros_like(attn_bias) # avoid nan
        # in_degree, out_degree = batched_data.in_degree, batched_data.in_degree
        edge_input, attn_edge_type = (
            batched_data["edge_input"],
            batched_data["attn_edge_type"],
        )
        n_graph, n_node = x.size()[:2]
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
            1, self.edge_embedding_dim, 1, 1
        )  # [n_graph, edge_dim, n_node+1, n_node+1]

        # spatial pos
        # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias
        
        # reset spatial pos here
        t = self.graph_token_virtual_distance.weight.view(1, self.edge_embedding_dim, 1)
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

        # edge feature
        if self.edge_type == "multi_hop": # here
            spatial_pos_ = spatial_pos.clone()
            spatial_pos_[spatial_pos_ == 0] = 1  # set pad to 1
            # set 1 to 1, node_embeds > 1 to node_embeds - 1
            spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
            if self.multi_hop_max_dist > 0:
                spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
                edge_input = edge_input[:, :, :, : self.multi_hop_max_dist, :]
            # [n_graph, n_node, n_node, max_dist, n_head]
            edge_input = self.edge_encoder(edge_input).mean(-2)
            max_dist = edge_input.size(-2)
            edge_input_flat = edge_input.permute(3, 0, 1, 2, 4).reshape(
                max_dist, -1, self.edge_embedding_dim)
            edge_input_flat = torch.bmm(
                edge_input_flat,
                self.edge_dis_encoder.weight.reshape(
                    -1, self.edge_embedding_dim, self.edge_embedding_dim
                )[:max_dist, :, :])
            edge_input = edge_input_flat.reshape(
                max_dist, n_graph, n_node, n_node, self.edge_embedding_dim
            ).permute(1, 2, 3, 0, 4)
            edge_input = (
                edge_input.sum(-2) / (spatial_pos_.float().unsqueeze(-1))
            ).permute(0, 3, 1, 2)
        else:
            # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
            edge_input = self.edge_encoder(attn_edge_type).mean(-2).permute(0, 3, 1, 2)
        
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + edge_input.to(graph_attn_bias.dtype)
        
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # [B, n_heads, N, N]
        return graph_attn_bias


class GraphPropagationAttention(nn.Module):
    def __init__(self, node_dim, edge_dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = node_dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(node_dim, node_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(node_dim, node_dim)
        
        self.reduce = nn.Conv2d(edge_dim, num_heads, kernel_size=1)
        self.expand = nn.Conv2d(num_heads, edge_dim, kernel_size=1)
        if edge_dim != node_dim:
            self.fc = nn.Linear(edge_dim, node_dim)
        else:
            self.fc = nn.Identity()
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, node_embeds, edge_embeds, padding_mask):
        # node-to-node propagation
        B, N, C = node_embeds.shape
        qkv = self.qkv(node_embeds).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale # [B, n_head, 1+N, 1+N]
        attn_bias = self.reduce(edge_embeds) # [B, C, 1+N, 1+N] -> [B, n_head, 1+N, 1+N]
        attn = attn + attn_bias # [B, n_head, 1+N, 1+N]
        residual = attn

        attn = attn.masked_fill(padding_mask, float("-inf"))
        attn = attn.softmax(dim=-1) # [B, C, N, N]
        attn = self.attn_drop(attn)
        node_embeds = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # node-to-edge propagation
        edge_embeds = self.expand(attn + residual)  # [B, n_head, 1+N, 1+N] -> [B, C, 1+N, 1+N]

        # edge-to-node propagation
        w = edge_embeds.masked_fill(padding_mask, float("-inf"))
        w = w.softmax(dim=-1)
        w = (w * edge_embeds).sum(-1).transpose(-1, -2)
        node_embeds = node_embeds + self.fc(w)
        node_embeds = self.proj(node_embeds)
        node_embeds = self.proj_drop(node_embeds)

        return node_embeds, edge_embeds


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., drop_act=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.drop_act = nn.Dropout(drop_act)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop_act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GPTransBlock(nn.Module):
    def __init__(self, node_dim, edge_dim, num_heads, mlp_ratio=1., qkv_bias=True, drop=0., drop_act=0.,
                 with_cp=False, attn_drop=0., drop_path=0., init_values=None):
        super().__init__()
        self.with_cp = with_cp
        self.norm1 = nn.LayerNorm(node_dim)
        self.gpa = GraphPropagationAttention(node_dim, edge_dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                             attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(node_dim)
        self.ffn = FFN(in_features=node_dim, hidden_features=int(node_dim * mlp_ratio),
                       drop=drop, drop_act=drop_act)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if init_values is not None:
            self.gamma1 = nn.Parameter(init_values * torch.ones((node_dim)), requires_grad=True)
            self.gamma2 = nn.Parameter(init_values * torch.ones((node_dim)), requires_grad=True)
        else:
            self.gamma1 = None
            self.gamma2 = None

    def forward(self, node_embeds, edge_embeds, padding_mask):
        
        def _inner_forward(x, edge_embeds):
            if self.gamma1 is not None:
                attn, edge_embeds_ = self.gpa(self.norm1(x), edge_embeds, padding_mask)
                edge_embeds = edge_embeds + edge_embeds_
                x = x + self.drop_path(self.gamma1 * attn)
                x = x + self.drop_path(self.gamma2 * self.ffn(self.norm2(x)))
            else:
                attn, edge_embeds_ = self.gpa(self.norm1(x), edge_embeds, padding_mask)
                edge_embeds = edge_embeds + edge_embeds_
                x = x + self.drop_path(attn)
                x = x + self.drop_path(self.ffn(self.norm2(x)))
            return x, edge_embeds
        
        if self.with_cp and node_embeds.requires_grad:
            node_embeds, edge_embeds = cp.checkpoint(_inner_forward, node_embeds, edge_embeds)
        else:
            node_embeds, edge_embeds = _inner_forward(node_embeds, edge_embeds)
        
        return node_embeds, edge_embeds


class GraphEmbedding(nn.Module):
    def __init__(self, num_atoms, num_in_degree, num_out_degree, num_edges, num_spatial,
                 num_edge_dist, edge_type, multi_hop_max_dist, num_layers, node_dim,
                 edge_dim, num_heads, dropout):
        
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.embedding_dim = node_dim
        self.emb_layer_norm = nn.LayerNorm(node_dim)
        self.graph_node_feature = GraphNodeFeature(
            num_heads=num_heads,
            num_atoms=num_atoms,
            num_in_degree=num_in_degree,
            num_out_degree=num_out_degree,
            hidden_dim=node_dim,
            num_layers=num_layers,
        )
        
        self.graph_edge_feature = GraphEdgeFeature(
            num_heads=num_heads,
            num_edges=num_edges,
            num_spatial=num_spatial,
            num_edge_dist=num_edge_dist,
            edge_type=edge_type,
            multi_hop_max_dist=multi_hop_max_dist,
            num_layers=num_layers,
            edge_dim=edge_dim,
        )
 
    def forward(self, batched_data, perturb=None):
        # compute padding mask. This is needed for multi-head attention
        data_x = batched_data["x"]  # [B, 18 or 19, 9]
        n_graph, n_node = data_x.size()[:2]
        padding_mask = (data_x[:, :, 0]).eq(0)  # B node_embeds T
        padding_mask_cls = torch.zeros(  # not mask
            n_graph, 1, device=padding_mask.device, dtype=padding_mask.dtype
        )  # B node_embeds 1
        padding_mask = torch.cat((padding_mask_cls, padding_mask), dim=1)
        # B node_embeds (1+T)
        
        node_embeds = self.graph_node_feature(batched_data)
        if perturb is not None:  # perturb is None
            node_embeds[:, 1:, :] += perturb
        
        # node_embeds: B node_embeds T node_embeds C
        edge_embeds = self.graph_edge_feature(batched_data)
        node_embeds = self.emb_layer_norm(node_embeds)
        node_embeds = self.dropout(node_embeds)
        
        return node_embeds, edge_embeds, padding_mask


class GPTrans(nn.Module):
    def __init__(self, num_layers=24, num_heads=23, node_dim=736, edge_dim=92, layer_scale=1.0, mlp_ratio=1.0,
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, num_atoms=4608, num_edges=1536,
                 num_in_degree=512, num_out_degree=512, num_spatial=512, num_edge_dist=128, multi_hop_max_dist=20,
                 edge_type='multi_hop', qkv_bias=True, num_classes=1, task_type="graph_regression",
                 random_feature=False, with_cp=False):
        super(GPTrans, self).__init__()
        logger.info(f"drop: {drop_rate}, drop_path_rate: {drop_path_rate}, attn_drop_rate: {attn_drop_rate}")
        
        self.task_type = task_type
        self.random_feature = random_feature
        self.graph_embedding = GraphEmbedding(
            num_atoms=num_atoms,
            num_in_degree=num_in_degree,
            num_out_degree=num_out_degree,
            num_edges=num_edges,
            num_spatial=num_spatial,
            num_edge_dist=num_edge_dist,
            edge_type=edge_type,
            multi_hop_max_dist=multi_hop_max_dist,
            num_layers=num_layers,
            node_dim=node_dim,
            edge_dim=edge_dim,
            num_heads=num_heads,
            dropout=drop_rate,
        )
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            GPTransBlock(node_dim=node_dim, edge_dim=edge_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                         drop_act=drop_rate, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                         qkv_bias=qkv_bias, init_values=layer_scale, with_cp=with_cp) for i in range(num_layers)
        ])
    
        self.fc_layer = nn.Sequential(
            nn.Linear(node_dim + edge_dim, node_dim),
            nn.LayerNorm(node_dim),
            nn.GELU(),
        )
        self.head = nn.Linear(node_dim, num_classes, bias=True)

    def forward(self, batched_data, perturb=None):
        node_embeds, edge_embeds, padding_mask = self.graph_embedding(
            batched_data,
            perturb=perturb,
        )
        if self.random_feature and self.training:
            node_embeds += torch.rand_like(node_embeds)
            edge_embeds += torch.rand_like(edge_embeds)
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool)

        for blk in self.blocks:
            node_embeds, edge_embeds = blk(node_embeds,  # [B, 1+N, C]
                                           edge_embeds,  # [B, C, 1+N, 1+N]
                                           padding_mask) # [B, 1+N, 1]
        if self.task_type == "graph_regression" or self.task_type == "graph_classification":
            x = torch.cat([node_embeds[:, :1, :], edge_embeds[:, :, 0:1, 0].transpose(-1, -2)], dim=2)
            x = self.fc_layer(x)
            x = self.head(x)[:, 0, :] # select the virtual node
            if x.size(-1) == 1: x = x.squeeze(-1)
        elif self.task_type == "node_classification":
            diag = torch.diagonal(edge_embeds, dim1=-1, dim2=-2).transpose(-1, -2)
            x = torch.cat([node_embeds, diag], dim=2)[:, 1:, :]
            x = x.reshape(-1, x.shape[-1])
            x = self.fc_layer(x)
            x = self.head(x)
        return x

    @torch.jit.ignore
    def lr_decay_keywords(self, decay_ratio=0.87):
        lr_ratios = {}
        depth = len(self.blocks) + 1
        for k, v in self.named_parameters():
            if "graph_embedding." in k:
                lr_ratios[k] = decay_ratio ** depth
            elif "blocks." in k:
                block_id = int(k.split(".")[1])
                lr_ratios[k] = decay_ratio ** (depth - block_id - 1)
            elif "fc_layer." in k or "head." in k:
                lr_ratios[k] = 1
        return lr_ratios