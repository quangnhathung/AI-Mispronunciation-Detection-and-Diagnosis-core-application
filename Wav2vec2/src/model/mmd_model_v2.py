import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class MDDModelV2(nn.Module):
    def __init__(self, vocab_size, embed_dim=768, num_heads=8, freeze_feature_extractor=True, num_frozen_transformer_layers=10):
        super(MDDModelV2, self).__init__()
        
        # ==========================================
        # 1. ĐÔI TAI: Wav2Vec 2.0 (The Listener)
        # ==========================================
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        
        if freeze_feature_extractor:
            self.wav2vec2.feature_extractor._freeze_parameters()
            
        for i in range(num_frozen_transformer_layers):
            for param in self.wav2vec2.encoder.layers[i].parameters():
                param.requires_grad = False
                
        # ==========================================
        # 2. VỊ GIÁM KHẢO: Phoneme Encoder (V3 - THÊM GRU)
        # ==========================================
        self.phoneme_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0)
        
        # MẠNG GRU HAI CHIỀU (Bi-directional GRU)
        # Tác dụng: Cấp nhận thức về "vị trí" và "ngữ cảnh xung quanh" cho từng âm vị.
        self.phoneme_rnn = nn.GRU(
            input_size=embed_dim, 
            hidden_size=embed_dim // 2, # Chia 2 vì 2 chiều gộp lại sẽ thành embed_dim (768)
            num_layers=1, 
            batch_first=True, 
            bidirectional=True
        )
        
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
        )

    def forward(self, input_values, attention_mask, canonical_ids):
        # 1. Audio Features (Key, Value)
        w2v_output = self.wav2vec2(input_values=input_values, attention_mask=attention_mask)
        latent_audio = w2v_output.last_hidden_state 
        
        if attention_mask is not None:
            audio_pad_mask = self._get_feat_extract_output_lengths(attention_mask.sum(dim=1))
            T_out = latent_audio.shape[1]
            key_padding_mask = torch.arange(T_out, device=latent_audio.device)[None, :] >= audio_pad_mask[:, None]
        else:
            key_padding_mask = None

        # 2. Phoneme Features (Query) - NÂNG CẤP
        # phoneme_embeds: (Batch, Seq_Len, 768) - Vector Tĩnh
        phoneme_embeds = self.phoneme_embedding(canonical_ids)
        
        # Đưa qua GRU để tạo Vector Động (có chứa thông tin vị trí)
        # query_phonemes: (Batch, Seq_Len, 768)
        query_phonemes, _ = self.phoneme_rnn(phoneme_embeds)
        
        # 3. Tiến hành gióng hàng (Cross-Attention)
        aligned_features, attn_weights = self.cross_attention(
            query=query_phonemes,
            key=latent_audio,
            value=latent_audio,
            key_padding_mask=key_padding_mask
        )
        
        aligned_features = self.layer_norm(query_phonemes + aligned_features)
        
        # 4. Chấm điểm
        logits = self.scoring_head(aligned_features).squeeze(-1)
        
        return logits, attn_weights

    def _get_feat_extract_output_lengths(self, input_lengths):
        return self.wav2vec2._get_feat_extract_output_lengths(input_lengths)