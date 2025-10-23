import jittor as jt
import jittor.nn as nn
from .gnn_feature import PolygonGCN,Transformer,repeat_interleave,bincount
import numpy as np

class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        self.W = nn.Parameter(jt.randn(embed_dim // 2) * scale, requires_grad=False)

    def execute(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return jt.cat([jt.sin(x_proj), jt.cos(x_proj)], dim=-1)

class PolygonPackingTransformer(nn.Module):
    def __init__(self, marginal_prob_std_func, feature_dim=64, hidden_dim=128, nhead=16, maxInputLength=50, num_encoder_layers=8, num_decoder_layers=8):
        super(PolygonPackingTransformer, self).__init__()
        
        self.marginal_prob_std = marginal_prob_std_func
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.maxInputLength = maxInputLength
        
        self.t_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU()
        )
        
        self.action_encoder = nn.Sequential(
            nn.Linear(4, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
        )

        self.geo_feature = PolygonGCN(self.feature_dim)
        
        self.transformer_model = Transformer(
            d_model=hidden_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers
        )

        self.output_layer = nn.Linear(hidden_dim, 4)  # 输出层，将Transformer的输出映射到2个值，即x和y速度

    def execute(self, actions, polyIds, paddingMask, batch, t, gnnFeatureData, polyFeatures=None):
        batchCounts = bincount(batch)
        batchSize = int(batch.max() + 1)
        
        sigma_feature = self.t_embed(t.unsqueeze(-1))
        actions_feature = self.action_encoder(actions).reshape(batchSize, -1, self.feature_dim)
        
        if polyFeatures is None:
            geo_feature_all = self.geo_feature(gnnFeatureData) # polyCnt, e * fd
        else:
            geo_feature_all = polyFeatures

        polyIds = polyIds.unsqueeze(1).expand(-1, self.feature_dim) 
        geo_feature = jt.gather(geo_feature_all, 0, polyIds).reshape(batchSize, -1, self.feature_dim)

        semantic_geo_pos = jt.cat([geo_feature, actions_feature], dim=-1) # batch, maxCnt, fd * 2
        combined_feature_input = jt.cat([semantic_geo_pos, sigma_feature], dim=1) # batch, maxCnt + 1, fd * 2
        # paddingMask is a float tensor, where -inf means masked
        # paddingMask is used for conbine feature input, but with shape batch * maxCnt, 1
        paddingMask = paddingMask.reshape(batchSize, -1)
        paddingMaskInput = jt.cat([paddingMask, jt.zeros(batchSize, 1)], dim=1)

        key_padding_mask = (paddingMaskInput < 0)  # [batch_size, seq_len]
        
        # 转置输入：从 [batch, seq, dim] 到 [seq, batch, dim]
        combined_feature_input_t = combined_feature_input.transpose(0, 1)
        
        transformer_out = self.transformer_model(
            combined_feature_input_t, 
            combined_feature_input_t,
            src_key_padding_mask=key_padding_mask,
            tgt_key_padding_mask=key_padding_mask,
            memory_key_padding_mask=key_padding_mask
        )
        
        # 转回 [batch, seq, dim]
        transformer_out = transformer_out.transpose(0, 1)
        
        mu, std = self.marginal_prob_std(jt.zeros_like(t), t)
        std = repeat_interleave(std, batchCounts).view(-1, 1) + 1e-5
        
        velocities = self.output_layer(transformer_out)[:, :-1].reshape(-1, 4)
        return velocities / std