import torch
from torch import nn
import torch.nn.functional as F

from models.coco_caption_utils import initialize_clip
from models.coco_caption_layers import TransformerDecoder, MemoryAdapterLayer


class CocoCaptioner(nn.Module):
    def __init__(self, tokenizer = None, word_embedder = None, config = None):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer 
        self.word_embedder = word_embedder
        self.positional_embedding = nn.Embedding(config['max_words_per_cap'], 768)

        self.visual_encoder, _ = initialize_clip(config)
        self.text_decoder = TransformerDecoder(
            config['num_layers'], 
            768, 
            config['nhead'], 
            config['dropout']
        )
        self.memory_adapter = MemoryAdapterLayer(dim_query=768, dim_mem=512)
        self.fc = nn.Linear(768, config['vocab_size'])
        self.img_emb_transform = nn.Linear(512, 768)
        
    def _cal_loss(self, prediction, caption_ids):
        # Dịch caption sang trái và loại bỏ từ cuối cùng
        shifted_caption_ids = caption_ids[:, :-1]

        # Mask padding tokens trong shifted_caption_ids
        shifted_caption_ids = shifted_caption_ids.masked_fill(shifted_caption_ids == self.tokenizer.pad_token_id, -100)

        # Cắt bỏ phần tử cuối cùng từ mỗi batch trong prediction
        prediction = prediction[:, :-1, :]

        # Reshape prediction và shifted_caption_ids để tính toán mất mát
        prediction = prediction.contiguous().view(-1, prediction.size(2))
        shifted_caption_ids = shifted_caption_ids.contiguous().view(-1)

        # Tính toán mất mát cross-entropy
        loss = F.cross_entropy(prediction, shifted_caption_ids, ignore_index=-100, reduction='mean')
        return loss

            
    def forward(self, image, caption=None):
        # image: (batch_size, 3, 224, 224)
        image = image.to(dtype=next(self.parameters()).dtype) 
        # image_embeds: (batch_size, 512)
        image_embeds = self.visual_encoder.get_image_features(image)
        image_embeds = image_embeds.unsqueeze(1) # (batch_size, 1, 512)   

        # caption_embeds: (batch_size, max_words_per_cap, 768)
        caption_embeds = self.word_embedder(caption.input_ids, attention_mask=caption.attention_mask).last_hidden_state 
        position_ids = torch.arange(0, caption.input_ids.size(1)).unsqueeze(0)
        positional_embeddings = self.positional_embedding(position_ids)
        caption_embeds = caption_embeds + positional_embeddings

        # caption_embeds, image_embeds = self.memory_adapter(caption_embeds, image_embeds)
        image_embeds = self.img_emb_transform(image_embeds)

        # output: (batch_size, max_words_per_cap, 768)
        output = self.text_decoder(caption_embeds, image_embeds)

        # output: (batch_size, max_words_per_cap, vocab_size)
        output = self.fc(output)
        prediction = F.softmax(output, dim=2)
        print("prediction:", prediction.shape, prediction)
        cap = caption.input_ids
        print("caption:", cap.shape, cap)
        loss = self._cal_loss(prediction, caption.input_ids)
        print(loss)
        return loss
    
    def simple_gen(self, image, temperature=1.0, max_length=50):
        # image: (batch_size, 3, 224, 224)
        image = image.to(dtype=next(self.parameters()).dtype) 
        # image_embeds: (batch_size, 512)
        image_embeds = self.visual_encoder.get_image_features(image)
        image_embeds = image_embeds.unsqueeze(1) # (batch_size, 1, 512)
        image_embeds = self.img_emb_transform(image_embeds)

        initial_tokens = [self.tokenizer.cls_token_id] * image_embeds.size(0)
        tokens = torch.tensor(initial_tokens, device=image_embeds.device).unsqueeze(-1)  # (batch_size, 1)

        for _ in range(max_length):
            token_embeds = self.word_embedder(tokens)[0]  # Đảm bảo kích thước: (batch_size, seq_len, embed_dim)

            # Sinh từ tiếp theo
            output = self.text_decoder(token_embeds, image_embeds)
            output = self.fc(output[:, -1, :])  # Chỉ lấy embedding của từ cuối cùng

            if temperature == 0:
                next_token = torch.argmax(output, dim=-1).unsqueeze(-1)
            else:
                output = output / temperature
                next_token = torch.multinomial(F.softmax(output, dim=-1), num_samples=1)

            tokens = torch.cat([tokens, next_token], dim=1)

            # Kiểm tra token '[SEP]'
            if torch.all(next_token == self.tokenizer.sep_token_id):
                break

        # Decode token IDs sang văn bản
        results = [self.tokenizer.decode(t, skip_special_tokens=True) for t in tokens]
        return results

