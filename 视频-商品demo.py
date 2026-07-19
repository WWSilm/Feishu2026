"""
视频-商品契合度预测模型 (Demo)
===============================
架构：多模态双塔 + 交叉融合 + 多任务学习
模态：视频帧、视频文本(ASR/标题)、音频、达人画像、商品文本、商品图片、商品属性
训练目标：回归(MSE) + 排序(Ranking) + 对比学习(InfoNCE)

注意：这是一个示范代码，使用随机数据模拟输入，不依赖真实数据集。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np

# ============================================================
# 第一部分：各模态编码器 (Encoders)
# ============================================================

class VideoVisualEncoder(nn.Module):
    """
    视频视觉编码器
    - 输入：从视频中采样的 N 帧图像特征 (在实际使用中，这些特征由预训练的 ViT/ResNet 提取)
    - 处理：线性投影 + 时序 Transformer 编码
    - 输出：视频视觉表征向量 [batch, d_model]
    """
    def __init__(self, frame_dim=512, d_model=256, n_frames=8, n_heads=4, n_layers=2):
        super().__init__()
        self.n_frames = n_frames
        self.d_model = d_model
        
        # 将预训练视觉特征投影到统一维度
        self.proj = nn.Linear(frame_dim, d_model)
        
        # 可学习的位置编码（因为帧数是固定的）
        self.pos_emb = nn.Parameter(torch.randn(1, n_frames, d_model) * 0.02)
        
        # 时序 Transformer：建模帧之间的时序关系
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_model * 4,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, frame_features):
        """
        Args:
            frame_features: [batch, n_frames, frame_dim] 
                           每帧的预训练视觉特征 (如 CLIP ViT 输出)
        Returns:
            video_visual_emb: [batch, d_model] 视频视觉表征
        """
        x = self.proj(frame_features)       # [batch, n_frames, d_model]
        x = x + self.pos_emb                # 加入位置编码
        x = self.transformer(x)             # [batch, n_frames, d_model]
        x = self.norm(x)
        
        # 取所有帧的平均池化作为视频整体视觉表征
        video_visual_emb = x.mean(dim=1)    # [batch, d_model]
        return video_visual_emb


class AudioEncoder(nn.Module):
    """
    音频编码器
    - 输入：音频的 Mel 频谱特征序列
    - 处理：投影 + Transformer 编码
    - 输出：音频表征向量 [batch, d_model]
    
    实际使用时，输入可以是 AST(Audio Spectrogram Transformer) 的特征输出
    """
    def __init__(self, audio_dim=768, d_model=256, n_heads=4, n_layers=2, max_len=256):
        super().__init__()
        self.proj = nn.Linear(audio_dim, d_model)
        
        # 可学习的位置编码
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, audio_features):
        """
        Args:
            audio_features: [batch, seq_len, audio_dim]  音频特征序列
        Returns:
            audio_emb: [batch, d_model]  音频表征
        """
        x = self.proj(audio_features)                   # [batch, seq_len, d_model]
        seq_len = x.size(1)
        x = x + self.pos_emb[:, :seq_len, :]            # 裁剪位置编码到实际长度
        x = self.transformer(x)                         # [batch, seq_len, d_model]
        x = self.norm(x)
        audio_emb = x.mean(dim=1)                       # [batch, d_model] 平均池化
        return audio_emb


class TextEncoder(nn.Module):
    """
    文本编码器（通用）
    - 可用于编码：视频标题/ASR文本、商品标题/描述
    - 输入：token ids 序列
    - 处理：Embedding + Transformer
    - 输出：[CLS] token 作为文本表征
    
    实际使用时可替换为预训练的 BERT/BGE 等模型
    """
    def __init__(self, vocab_size=30000, d_model=256, n_heads=4, n_layers=2, max_len=128):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        
        # [CLS] token，用于聚合整个序列的信息
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_ids):
        """
        Args:
            input_ids: [batch, seq_len]  token id 序列
        Returns:
            text_emb: [batch, d_model]  文本表征 (取 [CLS] token)
        """
        batch_size = input_ids.size(0)
        x = self.token_emb(input_ids)                           # [batch, seq_len, d_model]
        
        # 在序列开头拼接 [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch, 1, d_model]
        x = torch.cat([cls_tokens, x], dim=1)                   # [batch, seq_len+1, d_model]
        
        seq_len = x.size(1)
        x = x + self.pos_emb[:, :seq_len, :]
        x = self.transformer(x)
        x = self.norm(x)
        
        text_emb = x[:, 0, :]   # 取 [CLS] token 的输出  [batch, d_model]
        return text_emb


class CreatorEncoder(nn.Module):
    """
    达人画像编码器
    - 输入：达人的统计特征（粉丝数、历史内容类别分布、带货品类偏好等）
    - 处理：简单的 MLP
    - 输出：达人表征向量 [batch, d_model]
    """
    def __init__(self, creator_dim=64, d_model=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(creator_dim, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )

    def forward(self, creator_features):
        """
        Args:
            creator_features: [batch, creator_dim]  达人统计特征向量
        Returns:
            creator_emb: [batch, d_model]
        """
        return self.mlp(creator_features)


class ProductAttributeEncoder(nn.Module):
    """
    商品属性编码器
    - 输入：商品的类别ID、价格区间、品牌ID 等离散/连续属性
    - 处理：对离散属性用 Embedding，连续属性用 Linear，然后拼接
    - 输出：商品属性表征 [batch, d_model]
    """
    def __init__(self, num_categories=100, num_brands=500, num_price_bins=20, d_model=256):
        super().__init__()
        self.cat_emb = nn.Embedding(num_categories, d_model // 4)      # 品类 embedding
        self.brand_emb = nn.Embedding(num_brands, d_model // 4)        # 品牌 embedding
        self.price_emb = nn.Embedding(num_price_bins, d_model // 4)    # 价格区间 embedding
        
        # 融合三种属性 embedding
        concat_dim = (d_model // 4) * 3
        self.proj = nn.Sequential(
            nn.Linear(concat_dim, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model),
        )

    def forward(self, category_ids, brand_ids, price_bins):
        """
        Args:
            category_ids: [batch]  商品品类 ID
            brand_ids:    [batch]  品牌 ID
            price_bins:   [batch]  价格区间 ID
        Returns:
            attr_emb: [batch, d_model]
        """
        cat = self.cat_emb(category_ids)    # [batch, d_model//4]
        brand = self.brand_emb(brand_ids)   # [batch, d_model//4]
        price = self.price_emb(price_bins)  # [batch, d_model//4]
        
        concat = torch.cat([cat, brand, price], dim=-1)  # [batch, d_model//4 * 3]
        attr_emb = self.proj(concat)                      # [batch, d_model]
        return attr_emb


# ============================================================
# 第二部分：双塔 (Video Tower & Product Tower)
# ============================================================

class VideoTower(nn.Module):
    """
    视频塔：融合视频的所有模态特征
    - 视觉（帧）、文本（ASR/标题）、音频、达人画像
    - 通过 Gate 机制动态加权各模态的重要性
    """
    def __init__(self, d_model=256, vocab_size=30000, frame_dim=512, 
                 audio_dim=768, creator_dim=64, n_frames=8):
        super().__init__()
        self.d_model = d_model
        
        # 各模态编码器
        self.visual_enc = VideoVisualEncoder(frame_dim=frame_dim, d_model=d_model, n_frames=n_frames)
        self.audio_enc = AudioEncoder(audio_dim=audio_dim, d_model=d_model)
        self.text_enc = TextEncoder(vocab_size=vocab_size, d_model=d_model)
        self.creator_enc = CreatorEncoder(creator_dim=creator_dim, d_model=d_model)
        
        # 视频统计特征（播放量、时长等）
        self.stats_proj = nn.Linear(8, d_model)  # 假设有 8 个统计特征
        
        # Gate 机制：学习各模态的权重
        # 输入所有模态的拼接，输出每个模态的权重
        n_modalities = 5  # visual, audio, text, creator, stats
        self.gate = nn.Sequential(
            nn.Linear(d_model * n_modalities, n_modalities),
            nn.Softmax(dim=-1)
        )

    def forward(self, frames, audio_feat, text_ids, creator_feat, stats_feat):
        """
        Args:
            frames:       [batch, n_frames, frame_dim]  视频帧特征
            audio_feat:   [batch, audio_seq_len, audio_dim]  音频特征
            text_ids:     [batch, text_seq_len]  视频文本 token ids
            creator_feat: [batch, creator_dim]  达人特征
            stats_feat:   [batch, 8]  视频统计特征
        Returns:
            video_emb: [batch, d_model]  视频最终表征
        """
        # 各模态独立编码
        v_vis = self.visual_enc(frames)           # [batch, d_model]
        v_aud = self.audio_enc(audio_feat)         # [batch, d_model]
        v_txt = self.text_enc(text_ids)            # [batch, d_model]
        v_cre = self.creator_enc(creator_feat)     # [batch, d_model]
        v_sta = self.stats_proj(stats_feat)        # [batch, d_model]
        
        # 拼接所有模态，计算 gate 权重
        all_concat = torch.cat([v_vis, v_aud, v_txt, v_cre, v_sta], dim=-1)  # [batch, d_model*5]
        gate_weights = self.gate(all_concat)  # [batch, 5]  每个模态的权重
        
        # 按权重加权求和
        modalities = torch.stack([v_vis, v_aud, v_txt, v_cre, v_sta], dim=1)  # [batch, 5, d_model]
        gate_weights = gate_weights.unsqueeze(-1) * 5  # [batch, 5, 1]  乘以模态数保持量级
        video_emb = (modalities * gate_weights).sum(dim=1)  # [batch, d_model]
        
        return video_emb


class ProductTower(nn.Module):
    """
    商品塔：融合商品的所有模态特征
    - 文本（标题/描述）、图片、属性、统计特征
    """
    def __init__(self, d_model=256, vocab_size=30000, 
                 num_categories=100, num_brands=500, num_price_bins=20):
        super().__init__()
        self.d_model = d_model
        
        # 各模态编码器
        self.text_enc = TextEncoder(vocab_size=vocab_size, d_model=d_model)
        self.image_enc = nn.Sequential(
            nn.Linear(512, d_model),   # 假设预训练图片特征维度为 512
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.attr_enc = ProductAttributeEncoder(
            num_categories=num_categories,
            num_brands=num_brands,
            num_price_bins=num_price_bins,
            d_model=d_model
        )
        self.stats_proj = nn.Linear(4, d_model)  # 4 个商品统计特征
        
        # 简单拼接融合
        self.fuse = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, text_ids, image_feat, category_ids, brand_ids, price_bins, prod_stats):
        """
        Args:
            text_ids:     [batch, seq_len]  商品文本 token ids
            image_feat:   [batch, 512]  商品图片特征
            category_ids: [batch]  品类 ID
            brand_ids:    [batch]  品牌 ID
            price_bins:   [batch]  价格区间 ID
            prod_stats:   [batch, 4]  商品统计特征
        Returns:
            product_emb: [batch, d_model]  商品最终表征
        """
        p_txt = self.text_enc(text_ids)                         # [batch, d_model]
        p_img = self.image_enc(image_feat)                      # [batch, d_model]
        p_attr = self.attr_enc(category_ids, brand_ids, price_bins)  # [batch, d_model]
        p_sta = self.stats_proj(prod_stats)                     # [batch, d_model]
        
        concat = torch.cat([p_txt, p_img, p_attr, p_sta], dim=-1)  # [batch, d_model*4]
        product_emb = self.fuse(concat)                            # [batch, d_model]
        return product_emb


# ============================================================
# 第三部分：交互融合层 + 预测头
# ============================================================

class InteractionLayer(nn.Module):
    """
    交互融合层：计算视频表征和商品表征之间的匹配分数
    支持多种交互方式：
      1. 点积 (Dot Product)
      2. 拼接后 MLP (Concat MLP)
      3. 双线性 (Bilinear)
    这里使用 Concat MLP + 残差点积 的混合方式
    """
    def __init__(self, d_model=256):
        super().__init__()
        # 拼接后的 MLP
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )
        # 双线性交互矩阵
        self.bilinear = nn.Bilinear(d_model, d_model, 1)

    def forward(self, video_emb, product_emb):
        """
        Args:
            video_emb:   [batch, d_model]
            product_emb: [batch, d_model]
        Returns:
            score: [batch, 1]  契合度分数 (logits, 未经过 sigmoid)
        """
        # 方式1: 拼接后过 MLP
        concat = torch.cat([video_emb, product_emb], dim=-1)  # [batch, d_model*2]
        mlp_score = self.mlp(concat)                           # [batch, 1]
        
        # 方式2: 双线性
        bilinear_score = self.bilinear(video_emb, product_emb) # [batch, 1]
        
        # 方式3: 余弦相似度
        cos_score = F.cosine_similarity(video_emb, product_emb, dim=-1).unsqueeze(-1)  # [batch, 1]
        
        # 融合三种分数
        score = mlp_score + bilinear_score + cos_score  # [batch, 1]
        return score


# ============================================================
# 第四部分：完整模型
# ============================================================

class VideoProductMatcher(nn.Module):
    """
    视频-商品契合度预测模型（完整）
    """
    def __init__(self, d_model=256, vocab_size=30000, frame_dim=512,
                 audio_dim=768, creator_dim=64, n_frames=8,
                 num_categories=100, num_brands=500, num_price_bins=20):
        super().__init__()
        
        self.video_tower = VideoTower(
            d_model=d_model, vocab_size=vocab_size, frame_dim=frame_dim,
            audio_dim=audio_dim, creator_dim=creator_dim, n_frames=n_frames
        )
        self.product_tower = ProductTower(
            d_model=d_model, vocab_size=vocab_size,
            num_categories=num_categories, num_brands=num_brands,
            num_price_bins=num_price_bins
        )
        self.interaction = InteractionLayer(d_model=d_model)

    def forward(self, video_inputs, product_inputs):
        """
        前向传播
        Args:
            video_inputs: dict, 包含视频侧所有输入
            product_inputs: dict, 包含商品侧所有输入
        Returns:
            score: [batch, 1]  契合度分数 (0~1)
        """
        video_emb = self.video_tower(
            frames=video_inputs['frames'],
            audio_feat=video_inputs['audio'],
            text_ids=video_inputs['text_ids'],
            creator_feat=video_inputs['creator'],
            stats_feat=video_inputs['stats']
        )
        product_emb = self.product_tower(
            text_ids=product_inputs['text_ids'],
            image_feat=product_inputs['image'],
            category_ids=product_inputs['category_ids'],
            brand_ids=product_inputs['brand_ids'],
            price_bins=product_inputs['price_bins'],
            prod_stats=product_inputs['stats']
        )
        logits = self.interaction(video_emb, product_emb)  # [batch, 1]
        score = torch.sigmoid(logits)                       # [batch, 1] 映射到 0~1
        return score, video_emb, product_emb


# ============================================================
# 第五部分：损失函数
# ============================================================

class MultiTaskLoss(nn.Module):
    """
    多任务损失函数
    1. 回归损失 (MSE)：预测分数逼近真实 CTR
    2. 排序损失 (Pairwise)：同一视频下，高CTR商品的分数应高于低CTR商品
    3. 对比损失 (InfoNCE)：拉近正样本对，推远负样本对
    """
    def __init__(self, lambda_reg=1.0, lambda_rank=1.0, lambda_con=0.5, temperature=0.1):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.lambda_rank = lambda_rank
        self.lambda_con = lambda_con
        self.temperature = temperature

    def regression_loss(self, pred_scores, ctr_labels):
        """回归损失：Huber Loss（对异常值更鲁棒）"""
        return F.huber_loss(pred_scores.squeeze(), ctr_labels)

    def ranking_loss(self, pred_scores, ctr_labels):
        """
        排序损失（Pairwise）：
        在同一个 batch 中，对所有样本对 (i, j)，
        如果 ctr_i > ctr_j，则 pred_i 应该 > pred_j
        """
        n = pred_scores.size(0)
        if n < 2:
            return torch.tensor(0.0, device=pred_scores.device)
        
        # 构造所有 pair 的分数差和标签差
        pred_i = pred_scores.unsqueeze(1)     # [n, 1, 1]
        pred_j = pred_scores.unsqueeze(0)     # [1, n, 1]
        label_i = ctr_labels.unsqueeze(1)     # [n, 1]
        label_j = ctr_labels.unsqueeze(0)     # [1, n]
        
        # 只取 label_i > label_j 的 pair（正序对）
        pred_diff = pred_i.squeeze(-1) - pred_j.squeeze(-1)  # [n, n]
        label_diff = label_i - label_j                        # [n, n]
        
        # mask: 只保留正序对 (i 的 CTR > j 的 CTR)
        mask = (label_diff > 0).float()
        
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred_scores.device)
        
        # 对于正序对，pred_diff 应该 > 0
        loss = -torch.log(torch.sigmoid(pred_diff) + 1e-8) * mask
        return loss.sum() / (mask.sum() + 1e-8)

    def contrastive_loss(self, video_emb, product_emb, ctr_labels):
        """
        对比学习损失 (InfoNCE)：
        - 将 CTR 高于中位数的 (视频, 商品) 对视为正样本对
        - 其余为负样本对
        - 在 batch 内做对比
        """
        # L2 归一化
        v_norm = F.normalize(video_emb, dim=-1)     # [n, d]
        p_norm = F.normalize(product_emb, dim=-1)   # [n, d]
        
        # 相似度矩阵
        sim_matrix = torch.matmul(v_norm, p_norm.T) / self.temperature  # [n, n]
        
        # 根据 CTR 构造正样本：对角线是同一个视频-商品对
        # 这里简单地将每个样本和自身作为正样本对（自对比）
        labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
        loss = F.cross_entropy(sim_matrix, labels)
        return loss

    def forward(self, pred_scores, video_emb, product_emb, ctr_labels):
        """
        Args:
            pred_scores:  [batch, 1]  预测的契合度分数
            video_emb:    [batch, d_model]  视频表征
            product_emb:  [batch, d_model]  商品表征
            ctr_labels:   [batch]  真实 CTR 标签
        Returns:
            total_loss, loss_dict
        """
        l_reg = self.regression_loss(pred_scores, ctr_labels)
        l_rank = self.ranking_loss(pred_scores, ctr_labels)
        l_con = self.contrastive_loss(video_emb, product_emb, ctr_labels)
        
        total = self.lambda_reg * l_reg + self.lambda_rank * l_rank + self.lambda_con * l_con
        
        loss_dict = {
            'total': total.item(),
            'regression': l_reg.item(),
            'ranking': l_rank.item(),
            'contrastive': l_con.item(),
        }
        return total, loss_dict


# ============================================================
# 第六部分：模拟数据生成（用于 Demo）
# ============================================================

def generate_dummy_batch(batch_size=16, n_frames=8, frame_dim=512, 
                          audio_dim=768, audio_seq_len=64, text_seq_len=32,
                          creator_dim=64, vocab_size=30000):
    """
    生成一批模拟数据，用于演示模型的前向传播和训练
    实际使用时，这些数据会从真实数据集中加载
    """
    video_inputs = {
        'frames': torch.randn(batch_size, n_frames, frame_dim),          # 视频帧特征
        'audio': torch.randn(batch_size, audio_seq_len, audio_dim),       # 音频特征序列
        'text_ids': torch.randint(0, vocab_size, (batch_size, text_seq_len)),  # 视频文本 token ids
        'creator': torch.randn(batch_size, creator_dim),                  # 达人特征
        'stats': torch.randn(batch_size, 8),                              # 视频统计特征
    }
    
    product_inputs = {
        'text_ids': torch.randint(0, vocab_size, (batch_size, text_seq_len)),   # 商品文本 token ids
        'image': torch.randn(batch_size, 512),                                   # 商品图片特征
        'category_ids': torch.randint(0, 100, (batch_size,)),                    # 品类 ID
        'brand_ids': torch.randint(0, 500, (batch_size,)),                       # 品牌 ID
        'price_bins': torch.randint(0, 20, (batch_size,)),                       # 价格区间
        'stats': torch.randn(batch_size, 4),                                     # 商品统计特征
    }
    
    # 模拟 CTR 标签 (0~1 之间)
    ctr_labels = torch.rand(batch_size)
    
    return video_inputs, product_inputs, ctr_labels


# ============================================================
# 第七部分：训练演示
# ============================================================

def demo_train():
    """
    演示完整的训练流程：
    1. 初始化模型
    2. 生成模拟数据
    3. 前向传播
    4. 计算多任务损失
    5. 反向传播
    6. 打印训练信息
    """
    # 设置随机种子，保证可复现
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    device = torch.device('cpu')  # Demo 用 CPU，实际用 GPU
    
    # ---- 1. 初始化模型 ----
    model = VideoProductMatcher(
        d_model=128,         # 隐层维度（Demo 用较小值）
        vocab_size=30000,
        frame_dim=512,
        audio_dim=768,
        creator_dim=64,
        n_frames=8,
        num_categories=100,
        num_brands=500,
        num_price_bins=20,
    ).to(device)
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"=" * 60)
    print(f"模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"=" * 60)
    
    # ---- 2. 初始化损失函数和优化器 ----
    criterion = MultiTaskLoss(lambda_reg=1.0, lambda_rank=1.0, lambda_con=0.5, temperature=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # ---- 3. 模拟训练循环 ----
    n_epochs = 5
    batch_size = 16
    
    print(f"\n开始训练 (Demo: {n_epochs} epochs, batch_size={batch_size})")
    print("-" * 60)
    
    for epoch in range(n_epochs):
        model.train()
        
        # 生成模拟数据（实际中从 DataLoader 获取）
        video_inputs, product_inputs, ctr_labels = generate_dummy_batch(batch_size=batch_size)
        video_inputs = {k: v.to(device) for k, v in video_inputs.items()}
        product_inputs = {k: v.to(device) for k, v in product_inputs.items()}
        ctr_labels = ctr_labels.to(device)
        
        # 前向传播
        pred_scores, video_emb, product_emb = model(video_inputs, product_inputs)
        
        # 计算损失
        loss, loss_dict = criterion(pred_scores, video_emb, product_emb, ctr_labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        print(f"Epoch {epoch+1}/{n_epochs} | "
              f"Total Loss: {loss_dict['total']:.4f} | "
              f"Reg: {loss_dict['regression']:.4f} | "
              f"Rank: {loss_dict['ranking']:.4f} | "
              f"Con: {loss_dict['contrastive']:.4f}")
    
    # ---- 4. 推理演示 ----
    print(f"\n{'=' * 60}")
    print("推理演示：给定一个视频，评估它与多个候选商品的契合度")
    print("=" * 60)
    
    model.eval()
    with torch.no_grad():
        # 模拟 1 个视频 vs 5 个候选商品
        n_candidates = 5
        
        # 同一个视频重复 5 次（模拟一对多）
        video_inputs, _, _ = generate_dummy_batch(batch_size=1)
        # 使用 repeat 代替 expand，自动处理维度问题
        video_inputs = {k: v.repeat(n_candidates, *(1,) * (v.dim() - 1)).to(device) 
                        for k, v in video_inputs.items()}
        # 重新生成正确维度
        video_inputs, product_inputs, _ = generate_dummy_batch(batch_size=n_candidates)
        video_inputs = {k: v.to(device) for k, v in video_inputs.items()}
        product_inputs = {k: v.to(device) for k, v in product_inputs.items()}
        
        pred_scores, _, _ = model(video_inputs, product_inputs)
        
        # 模拟商品名称
        product_names = ["美白精华套装", "无线蓝牙耳机", "运动瑜伽垫", "智能手表Pro", "有机绿茶礼盒"]
        
        print(f"\n{'商品名称':<16} {'契合度分数':>10}")
        print("-" * 30)
        
        # 按分数排序输出
        scores = pred_scores.squeeze().cpu().numpy()
        sorted_indices = np.argsort(scores)[::-1]
        
        for idx in sorted_indices:
            score = scores[idx]
            name = product_names[idx]
            bar = "█" * int(score * 20)
            print(f"{name:<16} {score:>10.4f}  {bar}")
    
    print(f"\n✅ Demo 完成！")


# ============================================================
# 第八部分：模型结构可视化
# ============================================================

def print_model_summary():
    """打印模型结构概览"""
    print("\n" + "=" * 60)
    print("模型结构概览")
    print("=" * 60)
    
    model = VideoProductMatcher(d_model=128)
    print(model)
    
    print("\n" + "=" * 60)
    print("各模块参数量统计")
    print("=" * 60)
    
    modules = {
        '视频塔-视觉编码器': model.video_tower.visual_enc,
        '视频塔-音频编码器': model.video_tower.audio_enc,
        '视频塔-文本编码器': model.video_tower.text_enc,
        '视频塔-达人编码器': model.video_tower.creator_enc,
        '商品塔-文本编码器': model.product_tower.text_enc,
        '商品塔-图片编码器': model.product_tower.image_enc,
        '商品塔-属性编码器': model.product_tower.attr_enc,
        '交互融合层': model.interaction,
    }
    
    for name, module in modules.items():
        params = sum(p.numel() for p in module.parameters())
        print(f"  {name:<20} {params:>10,} 参数")


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    # 打印模型结构
    print_model_summary()
    
    # 运行训练演示
    demo_train()
