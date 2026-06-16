import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, seq_len, patch_len, stride, d_model, dropout):
        super().__init__()

        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = int((seq_len - patch_len) / stride) + 1

        self.proj = nn.Linear(patch_len, d_model)

        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, d_model) * 0.02
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (batch * channels, seq_len)
        return: (batch * channels, num_patches, d_model)
        """

        patches = x.unfold(
            dimension=-1,
            size=self.patch_len,
            step=self.stride
        )

        x = self.proj(patches)
        x = x + self.pos_embed
        x = self.dropout(x)

        return x


class PatchTSTBackbone(nn.Module):
    def __init__(
        self,
        seq_len,
        patch_len=7,
        stride=3,
        d_model=64,
        n_heads=4,
        e_layers=2,
        d_ff=128,
        dropout=0.1,
    ):
        super().__init__()

        self.patch_embedding = PatchEmbedding(
            seq_len=seq_len,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            dropout=dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=e_layers
        )

        self.num_patches = self.patch_embedding.num_patches
        self.d_model = d_model

    def forward(self, x):
        """
        x: (batch, seq_len, channels)

        return:
        seq_feature: (batch, channels * num_patches * d_model)
        """

        batch_size, seq_len, channels = x.shape

        # channel-independent PatchTST
        x = x.permute(0, 2, 1)          # (batch, channels, seq_len)
        x = x.reshape(batch_size * channels, seq_len)

        x = self.patch_embedding(x)     # (batch * channels, patches, d_model)
        x = self.encoder(x)

        x = x.reshape(
            batch_size,
            channels,
            self.num_patches * self.d_model
        )

        x = x.flatten(start_dim=1)

        return x


class DiseasePatchTSTFusion(nn.Module):
    """
    PatchTST + 过程特征融合模型

    输入：
    x_seq: (batch, 28, seq_feature_dim)
    x_tab: (batch, tab_feature_dim)

    输出：
    y: (batch, 6)
    """

    def __init__(
        self,
        seq_len,
        seq_feature_dim,
        tab_feature_dim,
        output_dim=6,
        patch_len=7,
        stride=3,
        d_model=64,
        n_heads=4,
        e_layers=2,
        d_ff=128,
        dropout=0.1,
    ):
        super().__init__()

        self.backbone = PatchTSTBackbone(
            seq_len=seq_len,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            dropout=dropout
        )

        seq_out_dim = seq_feature_dim * self.backbone.num_patches * d_model

        self.seq_head = nn.Sequential(
            nn.Linear(seq_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.tab_head = nn.Sequential(
            nn.Linear(tab_feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.fusion_head = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )

    def forward(self, x_seq, x_tab):
        seq_feat = self.backbone(x_seq)
        seq_feat = self.seq_head(seq_feat)

        tab_feat = self.tab_head(x_tab)

        feat = torch.cat([seq_feat, tab_feat], dim=1)

        out = self.fusion_head(feat)

        return out