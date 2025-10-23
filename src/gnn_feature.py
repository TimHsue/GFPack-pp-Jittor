from re import X
import jittor as jt
import jittor.nn as nn
import numpy as np

from jittor_geometric.ops import cootocsr,cootocsc



from jittor_geometric.nn.conv.gcn_conv import gcn_norm

from jittor_geometric.nn import GCNConv

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
        x = x.transpose(0, 1)
        batch_size = x.shape[1]
        query = self.query.repeat(1, batch_size, 1)
        attn_output, _ = self.attn(query, x, x, key_padding_mask=key_padding_mask)
        
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


def repeat_interleave(input, repeats, dim=0):
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

    end_pos = repeats_var.cumsum(0)
    start_pos = end_pos - repeats_var

    total_len = repeats_var.sum().item()
    indices = jt.zeros(total_len, dtype='int32')
    
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
        
        cum_counts = jt.concat([jt.zeros(1, dtype=batch_counts.dtype), batch_counts.cumsum(0)], dim=0)
        
        inner_batch_indices = jt.arange(x.shape[0]) - repeat_interleave(cum_counts[:-1], batch_counts)
        
        batched_data = jt.zeros((batch_size, max_batch_size, feature_dim), dtype=x.dtype)
        
        batched_data[batch, inner_batch_indices] = x
        
        range_vec = jt.arange(max_batch_size)
        mask = range_vec < batch_counts.unsqueeze(1)
        paddingMask = jt.full((batch_size, max_batch_size), -inf, dtype='float32')
        paddingMask = paddingMask.masked_fill(mask, 0.0)
        
        return batched_data, paddingMask
    
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
        
        batchedX = batchedX.transpose(0, 1)
        x = self.transformer_encoder(batchedX, src_key_padding_mask=key_padding_mask)
        
        x = x.transpose(0, 1)
        
        x = self.att_pool(x, key_padding_mask=key_padding_mask)
        
        return x
