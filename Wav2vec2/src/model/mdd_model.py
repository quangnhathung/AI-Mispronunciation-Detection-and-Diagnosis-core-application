import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class MDDModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=768, num_heads=8, freeze_feature_extractor=True, num_frozen_transformer_layers=10):
        super(MDDModel, self).__init__()
        
        # ==========================================
        # 1. ĐÔI TAI: Wav2Vec 2.0 (The Listener)
        # ==========================================
        # Load bản pre-train chuẩn 16kHz
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        
        # Đóng băng (Freeze) khối CNN để giữ nguyên khả năng cảm thụ âm thanh thô
        if freeze_feature_extractor:
            self.wav2vec2.feature_extractor._freeze_parameters()
            
        # Đóng băng các lớp Transformer đầu tiên, chỉ fine-tune (mở khóa) 2 lớp cuối
        for i in range(num_frozen_transformer_layers):
            for param in self.wav2vec2.encoder.layers[i].parameters():
                param.requires_grad = False
                
        # ==========================================
        # 2. VỊ GIÁM KHẢO: Phoneme Cross-Attention
        # ==========================================
        # Chuyển đổi ID âm vị thành Vector 768 chiều (Query)
        # padding_idx=0 giúp mạng lờ đi các giá trị đệm (PAD)
        self.phoneme_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0)
        
        # Khối Cross-Attention: 
        # - Query (Q): Chuỗi âm vị chuẩn
        # - Key (K), Value (V): Sóng âm thực tế từ Wav2Vec 2.0
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            dropout=0.1, 
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # ==========================================
        # 3. BÚT CHẤM ĐIỂM: Scoring Head
        # ==========================================
        self.scoring_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
            # Chú ý: Cố tình KHÔNG dùng nn.Sigmoid() ở lớp cuối.
            # Vì hàm Loss (BCEWithLogitsLoss) trong tập train.py đã tự động tính Sigmoid rồi.
        )

    def forward(self, input_values, attention_mask, canonical_ids):
        """
        Đầu vào:
        - input_values: (Batch, T_max) -> Sóng âm thanh thô
        - attention_mask: (Batch, T_max) -> Mặt nạ độ dài âm thanh
        - canonical_ids: (Batch, N_max) -> Nhãn ID âm vị chuẩn
        """
        
        # 1. Đưa âm thanh qua Wav2Vec 2.0
        # Đầu ra latent_audio có kích thước: (Batch, T_out, 768)
        w2v_output = self.wav2vec2(input_values=input_values, attention_mask=attention_mask)
        latent_audio = w2v_output.last_hidden_state 
        
        # Xử lý mặt nạ cho hàm Cross Attention của PyTorch 
        # (PyTorch quy định True là Pad bị bỏ qua, False là dữ liệu thật)
        if attention_mask is not None:
            audio_pad_mask = self._get_feat_extract_output_lengths(attention_mask.sum(dim=1))
            T_out = latent_audio.shape[1]
            key_padding_mask = torch.arange(T_out, device=latent_audio.device)[None, :] >= audio_pad_mask[:, None]
        else:
            key_padding_mask = None

        # 2. Nhúng chuỗi ID âm vị thành Vector
        # Đầu ra query_phonemes: (Batch, N_max, 768)
        query_phonemes = self.phoneme_embedding(canonical_ids)
        
        # 3. Tiến hành gióng hàng (Soft-Alignment) qua Cross-Attention
        aligned_features, attn_weights = self.cross_attention(
            query=query_phonemes,
            key=latent_audio,
            value=latent_audio,
            key_padding_mask=key_padding_mask
        )
        
        # Cơ chế Residual (cộng dồn) + LayerNorm để model học ổn định hơn
        aligned_features = self.layer_norm(query_phonemes + aligned_features)
        
        # 4. Chấm điểm từng âm vị
        # Đầu ra: (Batch, N_max) -> Các con số thực logit
        logits = self.scoring_head(aligned_features).squeeze(-1)
        
        return logits, attn_weights

    def _get_feat_extract_output_lengths(self, input_lengths):
        """Hàm tính lại độ dài của tensor âm thanh sau khi bị nén bởi khối CNN của Wav2Vec 2.0."""
        return self.wav2vec2._get_feat_extract_output_lengths(input_lengths)