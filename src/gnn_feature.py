from re import X
import jittor as jt
import jittor.nn as nn
import numpy as np

from jittor_geometric.ops import cootocsr,cootocsc



from jittor_geometric.nn.conv.gcn_conv import gcn_norm

from jittor_geometric.nn import GCNConv
# from torch_geometric.nn import GCNConv, global_mean_pool

import jittor as jt

def bincount(input, weights=None, minlength=0):
    if len(input.shape) != 1:
        raise ValueError("bincount only supports 1-d tensors")

    input = input.int32()
    if input.numel() == 0:
        size = minlength
    else:
        size = int(input.max().item()) + 1
        size = max(size, minlength)

    one_hot = jt.nn.one_hot(input, num_classes=size).float32()
    if weights is not None:
        weights = weights.reshape([-1, 1])
        return (one_hot * weights).sum(dim=0)
    else:
        return one_hot.sum(dim=0)

def segment_sum(data, segment_ids, num_segments):
    """
    data: [N, F]
    segment_ids: [N]，每个元素是 [0, num_segments-1] 的整数
    num_segments: int，输出的 segment 数
    返回: [num_segments, F]
    """
    segment_ids = segment_ids.int32()
    out = jt.zeros([num_segments, data.shape[1]], dtype=data.dtype)
    out = out.index_add_(0, segment_ids, data)
    return out


class AttentionPooling(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(AttentionPooling, self).__init__()
        self.query = jt.randn(1, 1, input_dim)
        self.query.requires_grad = True
        self.attn = jt.attention.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)

    def execute(self, x, key_padding_mask=None):
        # x shape: [batch_size, seq_len, input_dim]
        # key_padding_mask shape: [batch_size, seq_len]
        
        # Reshape x to [seq_len, batch_size, input_dim]
        x = x.transpose(0, 1)
        
        batch_size = x.shape[1]
        # query shape: [1, batch_size, input_dim]
        query = self.query.repeat(1, batch_size, 1)
        
        # key_padding_mask shape: [batch_size, seq_len]
        attn_output, _ = self.attn(query, x, x, key_padding_mask=key_padding_mask)
        # attn_output shape: [1, batch_size, output_dim]
        
        # Reshape output to [batch_size, output_dim]
        return attn_output.squeeze(0)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = jt.attention.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def execute(self, src, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(nn.relu(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = jt.attention.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = jt.attention.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def execute(self, tgt, memory, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(nn.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([
            type(encoder_layer)(
                encoder_layer.self_attn.embed_dim,
                encoder_layer.self_attn.num_heads,
                encoder_layer.linear1.out_features,
                encoder_layer.dropout.p
            ) for _ in range(num_layers)
        ])
        self.norm = norm

    def execute(self, src, src_key_padding_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([
            type(decoder_layer)(
                decoder_layer.self_attn.embed_dim,
                decoder_layer.self_attn.num_heads,
                decoder_layer.linear1.out_features,
                decoder_layer.dropout.p
            ) for _ in range(num_layers)
        ])
        self.norm = norm

    def execute(self, tgt, memory, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, 
                          tgt_key_padding_mask=tgt_key_padding_mask,
                          memory_key_padding_mask=memory_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self.d_model = d_model
        self.nhead = nhead

    def execute(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None, 
                memory_key_padding_mask=None):
        memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt, memory, 
                             tgt_key_padding_mask=tgt_key_padding_mask,
                             memory_key_padding_mask=memory_key_padding_mask)
        return output

# def repeat_interleave(input, repeats, dim=None):
#     if dim is None:
#         input = input.reshape(-1)
#         dim = 0
    
#     if dim < 0:
#         dim = len(input.shape) + dim
    
#     if isinstance(repeats, int):
#         indices = []
#         for i in range(input.shape[dim]):
#             indices.extend([i] * repeats)
#     else:
#         if isinstance(repeats, jt.Var):
#             repeats_list = jt.tolist(repeats)
#         elif hasattr(repeats, 'tolist'):
#             repeats_list = repeats.tolist()
#         else:
#             repeats_list = list(repeats)
        
#         indices = []
#         for i, r in enumerate(repeats_list):
#             indices.extend([i] * int(r))
    
#     indices_tensor = jt.array(indices, dtype='int32')
    
#     full_indices = [slice(None)] * len(input.shape)
#     full_indices[dim] = indices_tensor
    
#     return input[tuple(full_indices)]

def repeat_interleave(input, repeats, dim=0):
    """
    Jittor-native implementation of repeat_interleave that is fully differentiable
    and does not rely on jt.Var.repeat(tensor), which is not supported.
    """
    if dim is None:
        input = input.reshape(-1)
        dim = 0
    if dim < 0:
        dim += input.ndim

    if isinstance(repeats, int):
        repeats_var = jt.full([input.shape[dim]], repeats, dtype='int32')
    else:
        repeats_var = repeats if isinstance(repeats, jt.Var) else jt.array(repeats)
        repeats_var = repeats_var.reshape([-1]).int32()
        if repeats_var.ndim != 1 or repeats_var.shape[0] != input.shape[dim]:
            raise ValueError(f"repeats length must match input.shape[dim]. Got {repeats_var.shape[0]} and {input.shape[dim]}.")

    # Create an index tensor for gathering
    # Example: input.shape[dim]=3, repeats=[2,1,3]
    # arange -> [0, 1, 2]
    # This part needs a Jittor-native way to repeat elements of a tensor by counts in another tensor.
    # We can do this by creating a target index array.
    
    # 1. Get cumulative sum of repeats to know where each block starts
    # cumsum -> [2, 3, 6]
    # end_pos -> [2, 3, 6]
    end_pos = repeats_var.cumsum(0)
    # start_pos -> [0, 2, 3]
    start_pos = end_pos - repeats_var

    # 2. Create the index array
    total_len = repeats_var.sum().item()
    indices = jt.zeros(total_len, dtype='int32')
    
    # This loop is on CPU and creates ops, it's not a differentiable part of the graph itself.
    for i in range(input.shape[dim]):
        s = start_pos[i].item()
        e = end_pos[i].item()
        if s < e:
            indices[s:e] = i
    
    return input.index_select(dim, indices)



class PolygonGCN(nn.Module):
    def __init__(self, out_feature):
        super(PolygonGCN, self).__init__()
        self.initLin = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU()
        )
        
        globalEncNum = 8
        d_model = 64
        
        self.conv1 = GCNConv(16, 16)
        self.conv2 = GCNConv(32, 16)
        self.conv3 = GCNConv(48, 16)
        self.conv4 = GCNConv(64, d_model - globalEncNum)
        self.global_fc = nn.Linear(1, globalEncNum) # in feature is 1, out feature is 16
        
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=8)#, batch_first=True
        encoder_norm = nn.LayerNorm(d_model)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=2, norm=encoder_norm)
        
        self.att_pool = AttentionPooling(d_model, 8)
        
    def paddingToEachBatch(self, x, batch):
        inf = 1e9
        
        batch_size = int(batch.max().item()) + 1
        batch_counts = bincount(batch)
        max_batch_size = int(batch_counts.max().item())
        
        feature_dim = int(x.shape[-1])
        
        # --- Start of change ---
        # 创建一个索引，用于将扁平的 x 映射到批处理后的位置
        # cum_counts: [0, count_0, count_0+count_1, ...]
        cum_counts = jt.concat([jt.zeros(1, dtype=batch_counts.dtype), batch_counts.cumsum(0)], dim=0)
        
        # 为每个 batch 内的元素创建索引 (0, 1, 2, ...)
        inner_batch_indices = jt.arange(x.shape[0]) - repeat_interleave(cum_counts[:-1], batch_counts)
        
        # 创建最终的 batched_data 张量
        batched_data = jt.zeros((batch_size, max_batch_size, feature_dim), dtype=x.dtype)
        
        # 使用高级索引直接赋值，这会保持梯度
        # batch 是每个元素所属的 batch_id
        # inner_batch_indices 是每个元素在 batch 内的 id
        batched_data[batch, inner_batch_indices] = x
        
        # 创建 padding mask
        range_vec = jt.arange(max_batch_size)
        mask = range_vec < batch_counts.unsqueeze(1)
        paddingMask = jt.full((batch_size, max_batch_size), -inf, dtype='float32')
        paddingMask = paddingMask.masked_fill(mask, 0.0)
        # --- End of change ---
        
        return batched_data, paddingMask
    
            
    # def paddingToEachBatch(self, x, batch):
    #     inf = 1e9
        
    #     batch_size = int(batch.max().item()) + 1
    #     batch_counts = bincount(batch)
    #     max_batch_size = int(batch_counts.max().item())
        
    #     feature_dim = int(x.shape[-1])
        
    #     range_vec = jt.arange(max_batch_size)  # [max_batch_size]
    #     range_tensor = range_vec.unsqueeze(0).repeat(batch_size, 1)  # [batch_size, max_batch_size]
    #     batch_counts_expanded = batch_counts.unsqueeze(1).repeat(1, max_batch_size)  # [batch_size, max_batch_size]
    #     mask = range_tensor < batch_counts_expanded
        
    #     # batched_data: [batch_size, max_batch_size, feature_dim]
    #     batched_data = jt.zeros((batch_size, max_batch_size, feature_dim), dtype=x.dtype)
    #     current_idx = 0
    #     for i in range(batch_size):
    #         batch_len = int(batch_counts[i].item())
    #         batched_data[i, :batch_len, :] = x[current_idx:current_idx + batch_len, :]
    #         current_idx += batch_len
    #     # paddingMask: [batch_size, max_batch_size]
    #     paddingMask = jt.ones((batch_size, max_batch_size), dtype='float32') * (-inf)
    #     paddingMask[mask] = 0.0
        
    #     return batched_data, paddingMask
    
    def execute(self, data):
        x, edge_index, batch, area, perm = data.x, data.edge_index, data.batch, data.area, data.perm
        
        x.requires_grad = True

        global_features = jt.stack([perm], dim=1) # shape = batch, 1
        x0 = self.initLin(x) # 32

        edge_weight = jt.ones(edge_index.shape[1], dtype=jt.float32)
        csc = cootocsc(edge_index, edge_weight, x.shape[0])
        csr = cootocsr(edge_index, edge_weight, x.shape[0])

        x1 = nn.relu(self.conv1(x0, csc, csr)) # 32
        x = jt.cat([x0, x1], dim=1) # 32 + 32 = 64
        x2 = nn.relu(self.conv2(x, csc, csr)) # 64
        x = jt.cat([x0, x1, x2], dim=1) # 32 + 32 + 64 = 128
        x3 = nn.relu(self.conv3(x, csc, csr)) # 128
        x = jt.cat([x0, x1, x2, x3], dim=1) # 32 + 32 + 64 + 128 = 256
        x = nn.relu(self.conv4(x, csc, csr)) # 64
        global_features = self.global_fc(global_features) # shape = batch, 16
        batchCounts = bincount(batch)
        

        global_features = repeat_interleave(global_features, batchCounts, dim=0).reshape(-1, 8)
        
        x = jt.cat([x, global_features], dim=1)
        
        batchedX, paddingMask = self.paddingToEachBatch(x, batch)

        key_padding_mask = (paddingMask < 0)
        
        # 转置输入
        batchedX = batchedX.transpose(0, 1)
        x = self.transformer_encoder(batchedX, src_key_padding_mask=key_padding_mask)
        
        # 转回 [batch_size, max_batch_size, feature_dim]
        x = x.transpose(0, 1)
        
        x = self.att_pool(x, key_padding_mask=key_padding_mask)
        
        return x
