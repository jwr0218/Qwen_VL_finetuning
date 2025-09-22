import torch
import torch.nn as nn
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image
import requests
import warnings

# 경고 메시지 무시
warnings.filterwarnings("ignore")

class OCRandDescriptionQwenVL(nn.Module):
    def __init__(self, model_name="Qwen/Qwen2.5-VL-3B-Instruct"):
        super().__init__()
        
        self.qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, 
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        qwen_config = self.qwen_model.config
        
        self.token_embedder = self.qwen_model.get_input_embeddings()

        ocr_decoder_layer = nn.TransformerDecoderLayer(
            d_model=qwen_config.hidden_size,
            nhead=qwen_config.num_attention_heads,
            dim_feedforward=qwen_config.intermediate_size,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.ocr_head = nn.TransformerDecoder(
            decoder_layer=ocr_decoder_layer,
            num_layers=2,
            
        )
        self.ocr_output_layer = nn.Linear(qwen_config.hidden_size, qwen_config.vocab_size )

        desc_decoder_layer = nn.TransformerDecoderLayer(
            d_model=qwen_config.hidden_size,
            nhead=qwen_config.num_attention_heads,
            dim_feedforward=qwen_config.intermediate_size,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.desc_head = nn.TransformerDecoder(
            decoder_layer=desc_decoder_layer,
            num_layers=2,
            
        )
        self.desc_output_layer = nn.Linear(qwen_config.hidden_size, qwen_config.vocab_size )

    def forward(self, image, ocr_labels, desc_labels, processor, tokenizer):
        if not self.training:
            raise ValueError("forward() is only for training. Use generate() for inference.")
            
        device = next(self.parameters()).device

        
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": 'DES & OCR'}]}
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(device, torch.float16)

        outputs = self.qwen_model(**inputs, output_hidden_states=True, return_dict=True)
        encoder_output = outputs.hidden_states[-1]

        # OCR Head
        shifted_ocr_labels = ocr_labels.new_zeros(ocr_labels.shape)
        shifted_ocr_labels[:, 1:] = ocr_labels[:, :-1].clone()
        # FIX: bos_token_id -> eos_token_id
        shifted_ocr_labels[:, 0] = self.qwen_model.config.eos_token_id 
        ocr_decoder_input = self.token_embedder(shifted_ocr_labels)
        ocr_tgt_mask = nn.Transformer.generate_square_subsequent_mask(ocr_labels.size(1)).to(device)
        ocr_decoded_output = self.ocr_head(tgt=ocr_decoder_input, memory=encoder_output, tgt_mask=ocr_tgt_mask)
        ocr_logits = self.ocr_output_layer(ocr_decoded_output)

        # Description Head
        shifted_desc_labels = desc_labels.new_zeros(desc_labels.shape)
        shifted_desc_labels[:, 1:] = desc_labels[:, :-1].clone()
        # FIX: bos_token_id -> eos_token_id
        shifted_desc_labels[:, 0] = self.qwen_model.config.eos_token_id
        desc_decoder_input = self.token_embedder(shifted_desc_labels)
        desc_tgt_mask = nn.Transformer.generate_square_subsequent_mask(desc_labels.size(1)).to(device)
        desc_decoded_output = self.desc_head(tgt=desc_decoder_input, memory=encoder_output, tgt_mask=desc_tgt_mask)
        desc_logits = self.desc_output_layer(desc_decoded_output)

        # Loss Calculation
        loss_fct = nn.CrossEntropyLoss(ignore_index=self.qwen_model.config.pad_token_id if self.qwen_model.config.pad_token_id is not None else -100)
        ocr_loss = loss_fct(ocr_logits.view(-1, ocr_logits.size(-1)).float(), ocr_labels.view(-1))
        desc_loss = loss_fct(desc_logits.view(-1, desc_logits.size(-1)).float(), desc_labels.view(-1))
        
        total_loss = 0.5 * ocr_loss + 0.5 * desc_loss
        return total_loss

    def _generate_for_head(self, head_name, encoder_output, tokenizer, max_length):
        device = encoder_output.device
        # FIX: bos_token_id -> eos_token_id
        generated_ids = torch.tensor([[tokenizer.eos_token_id]], dtype=torch.long, device=device)

        for _ in range(max_length):
            decoder_input_emb = self.token_embedder(generated_ids)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(generated_ids.size(1)).to(device)

            if head_name == "ocr":
                decoded_output = self.ocr_head(decoder_input_emb, encoder_output, tgt_mask)
                logits = self.ocr_output_layer(decoded_output)
            else: 
                decoded_output = self.desc_head(decoder_input_emb, encoder_output, tgt_mask)
                logits = self.desc_output_layer(decoded_output)
            
            next_token_logits = logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
            
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
            
            if next_token_id.item() == tokenizer.eos_token_id:
                break
                
        return tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    @torch.no_grad()
    def generate(self, image, tokenizer, processor, max_length=50):
        """
        Instruction-Tuned 모델을 위한 추론(generate) 메서드.
        각 태스크에 맞는 명확한 instruction을 주입하여 결과를 생성합니다.
        """
        self.eval()
        device = next(self.parameters()).device

        # 1. OCR 태스크를 위한 명확한 Instruction 주입
        ocr_instruction = "What text is written in the image? Transcribe it accurately."
        ocr_messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": ocr_instruction}]}]
        ocr_prompt = processor.apply_chat_template(ocr_messages, tokenize=False, add_generation_prompt=True)
        ocr_inputs = processor(text=[ocr_prompt], images=[image], return_tensors="pt").to(device, torch.float16)
        ocr_encoder_output = self.qwen_model(**ocr_inputs, output_hidden_states=True).hidden_states[-1]
        
        # 2. Description 태스크를 위한 명확한 Instruction 주입
        desc_instruction = "Describe the contents of this image in detail."
        desc_messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": desc_instruction}]}]
        desc_prompt = processor.apply_chat_template(desc_messages, tokenize=False, add_generation_prompt=True)
        desc_inputs = processor(text=[desc_prompt], images=[image], return_tensors="pt").to(device, torch.float16)
        desc_encoder_output = self.qwen_model(**desc_inputs, output_hidden_states=True).hidden_states[-1]
        
        # 3. 각 헤드에서 Instruction 기반의 결과 생성
        ocr_text = self._generate_for_head("ocr", ocr_encoder_output, tokenizer, max_length)
        desc_text = self._generate_for_head("desc", desc_encoder_output, tokenizer, max_length)
        
        return ocr_text, desc_text


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "huihui-ai/Qwen2.5-VL-3B-Instruct-abliterated"
    
    print(f"디바이스: {device} | 모델: {model_name}\n")
    
    # 모델과 프로세서를 float16 타입으로 일관성 있게 로드
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16)
    tokenizer = processor.tokenizer
    model = OCRandDescriptionQwenVL(model_name).to(device).half()

    # --- 데이터 준비 ---
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    
    gt_ocr_text = "A cat is sitting on a couch." # 예시 OCR 정답
    gt_desc_text = "A cat is resting on a blue and white striped couch." # 예시 설명 정답

    # 정답 레이블 토크나이징
    ocr_labels = tokenizer(gt_ocr_text, return_tensors='pt', padding='max_length', max_length=30, truncation=True).input_ids.to(device)
    desc_labels = tokenizer(gt_desc_text, return_tensors='pt', padding='max_length', max_length=30, truncation=True).input_ids.to(device)

    # --- 훈련 (1 스텝) ---
    print("--- 훈련 (1 스텝) ---")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    optimizer.zero_grad()
    
    # forward 메서드는 내부적으로 instruction 프롬프트를 동적으로 생성합니다.
    total_loss = model(
        image=image,
        ocr_labels=ocr_labels,
        desc_labels=desc_labels,
        processor=processor,
        tokenizer=tokenizer
    )
    
    print(f"계산된 총 손실 (Total Loss): {total_loss.item():.4f}")
    
    total_loss.backward()
    optimizer.step()
    
    print("손실을 기반으로 역전파 및 옵티마이저 스텝 완료.\n")
    
    # --- 추론 (생성) ---
    print("--- 추론 (생성) ---")
    
    # generate 메서드는 각 태스크에 맞는 명확한 instruction을 주입하여 결과를 생성합니다.
    generated_ocr, generated_desc = model.generate(
        image=image,
        tokenizer=tokenizer,
        processor=processor,
        max_length=30
    )
    
    print(f"OCR 헤드 생성 결과 (Instruction-based): '{generated_ocr}'")
    print(f"설명 헤드 생성 결과 (Instruction-based): '{generated_desc}'")


if __name__ == "__main__":
    main()