# The main J.O.S.I.E.v4o model file. It's heavely barrowed from NeXT-GPT, big thanks.
import os
from typing import List

from ImageBind.imagebind import *
from ImageBind.imagebind import data

import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer


class JOSIE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.max_length = args["max_length"]
        self.stage = args["stage"]

        ##### ENCODER STUFF
        print(f"Initializing ImageBind encoder ...")
        # imagebind_encoder_path = os.path.join(self.args["imagebind_encoder_path"])
        self.imagebind_encoder, self.imagebind_encoder_output_dim = imagebind_model.imagebind_huge(pretrained=True) #, store_path=imagebind_encoder_path)

        for name, param in self.imagebind_encoder.named_parameters():
            param.requires_grad = False
        self.imagebind_encoder.eval()
        print(f"... ImageBind encoder initialized.")


        ##### REASONER STUFF
        print(f"Initializing Reasoner LLM model and tokenizer ...")
        reasoner_path = os.path.join(self.args['reasoner_path'])
        self.reasoner = AutoModelForCausalLM.from_pretrained(reasoner_path)
        self.tokenizer = AutoTokenizer.from_pretrained(reasoner_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        print(f"... tokenizer initialized.")

        if self.args.get('freeze_lm'):
            print("Freezing the Reasoner ...")
            for param in self.reasoner.parameters():
                param.requires_grad = False
            self.reasoner.eval()
        else:
            print("Instruct tuning the Reasoner ...") # TODO implement the LoraConfig stuff
        self.reasoner.print_trainable_parameters()
        print(f"... Reasoner LLM model initialized.")


        ##### TOKENIZER SETTING STUFF
        if self.args.get('add_spetial_tokens'):
            print("Adding Spetial Tokens to vocabulary ...")
            self._add_image_token()
            self._add_video_token()
            self._add_audio_token()
            self.reasoner.resize_token_embeddings(len(self.tokenizer))
            print("...  Spetial Tokens added to vocabulary")


        ##### INPUUT PROJECTOR STUFF
        print("Initializing input ImageBind Projection ...")
        self.input_projetor = nn.Linear(self.imagebind_encoder_output_dim, self.reasoner.config.hidden_size)
        if self.args.get('freeze_input_proj'):
            for param in self.input_projetor.parameters():
                param.requires_grad = False

        # self.output_projetor = nn.Linear(self.reasoner.config.hidden_size, audio_model_input_dim)

    def _add_image_token(self):
        self.tokenizer.add_tokens(["<|image_start|>"])
        self.tokenizer.add_tokens(["<|image_end|>"])

        self.args['gen_img_token_idx'] = []
        for i in range(self.args['num_gen_img_tokens']):
            print(f'Adding <|image_{i}|> token to vocabulary.')
            print(f'Before adding new token, tokenizer("<|image_{i}|>") =', self.tokenizer(f'<|image_{i}|>', add_special_tokens=False))
            num_added_tokens = self.tokenizer.add_tokens(f'<|image_{i}|>')
            print(f'After adding {num_added_tokens} new tokens, tokenizer("<|image_{i}|>") =', self.tokenizer(f'<|image_{i}|>', add_special_tokens=False))
            gen_token_idx = self.tokenizer(f'<|image_{i}|>', add_special_tokens=False).input_ids
            assert len(gen_token_idx) == 1, gen_token_idx
            self.args['gen_img_token_idx'].append(gen_token_idx[0])

    def _add_video_token(self):
    # self.tokenizer.add_tokens({"<|video_start|>"})
    # self.tokenizer.add_tokens({"<|video_end|>"})

        self.args['gen_video_token_idx'] = []
        for i in range(self.args['num_gen_video_tokens']):
            print(f'Adding <|video_{i}|> token to vocabulary.')
            print(f'Before adding new token, tokenizer("<|video_{i}|>") =', self.tokenizer(f'<|video_{i}|>', add_special_tokens=False))
            num_added_tokens = self.tokenizer.add_tokens(f'<|video_{i}|>')
            print(f'After adding {num_added_tokens} new tokens, tokenizer("<|video_{i}|>") =', self.tokenizer(f'<|video_{i}|>', add_special_tokens=False))
            gen_token_idx = self.tokenizer(f'<|video_{i}|>', add_special_tokens=False).input_ids
            assert len(gen_token_idx) == 1, gen_token_idx
            self.args['gen_video_token_idx'].append(gen_token_idx[0])

    def _add_audio_token(self):
        # self.tokenizer.add_tokens({"<|audio_start|>"})
        # self.tokenizer.add_tokens({"<|audio_end|>"})

        self.args['gen_audio_token_idx'] = []
        for i in range(self.args['num_gen_audio_tokens']):
            print(f'Adding <|audio_{i}|> token to vocabulary.')
            print(f'Before adding new token, tokenizer("<|audio_{i}|>") =', self.tokenizer(f'<|audio_{i}|>', add_special_tokens=False))
            num_added_tokens = self.tokenizer.add_tokens(f'<|audio_{i}|>')
            print(f'After adding {num_added_tokens} new tokens, tokenizer("<|audio_{i}|>") =', self.tokenizer(f'<|audio_{i}|>', add_special_tokens=False))
            gen_token_idx = self.tokenizer(f'<|audio_{i}|>', add_special_tokens=False).input_ids
            assert len(gen_token_idx) == 1, gen_token_idx
            self.args['gen_audio_token_idx'].append(gen_token_idx[0])

    def encode_video(self, video_paths):
        inputs = {ModalityType.VISION: data.load_and_transform_video_data(video_paths, self.device)}
        inputs = {key: inputs[key].to(self.reasoner.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            video_embeds = embeddings[ModalityType.VISION]
        inputs_reasoner = self.input_projetor(video_embeds).unsqueeze(1)
        atts_llama = torch.ones(inputs_reasoner.size()[:-1], dtype=torch.long).to(self.device)
        return inputs_reasoner, atts_llama

    def encode_audio(self, audio_paths):
        inputs = {ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, self.device)}
        inputs = {key: inputs[key].to(self.reasoner.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            audio_embeds = embeddings[ModalityType.AUDIO]
        inputs_reasoner = self.input_projetor(audio_embeds).unsqueeze(1)
        atts_llama = torch.ones(inputs_reasoner.size()[:-1], dtype=torch.long).to(self.device)
        return inputs_reasoner, atts_llama

    def encode_image(self, image_paths):
        inputs = {ModalityType.VISION: data.load_and_transform_vision_data(image_paths, self.device)}
        inputs = {key: inputs[key].to(self.reasoner.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            image_embeds = embeddings['vision']
        inputs_reasoner = self.input_projetor(image_embeds).unsqueeze(1)
        atts_llama = torch.ones(inputs_reasoner.size()[:-1], dtype=torch.long).to(self.device)
        return inputs_reasoner, atts_llama


    def prompt_wrap_old(self, img_embeds, input_ids, target_ids, attention_mask):
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        batch_size = input_ids.shape[0]
        bos = torch.ones([batch_size, 1], dtype=input_ids.dtype, device=input_ids.device) * self.tokenizer.bos_token_id  # bsz x 1
        if self.args['freeze_lm']:
            p_after_embeds = self.reasoner.model.embed_tokens(input_ids).expand(batch_size, -1, -1)
            bos_embeds = self.reasoner.model.embed_tokens(bos)
        else:
            p_after_embeds = self.reasoner.model.model.embed_tokens(input_ids).expand(batch_size, -1, -1)
            bos_embeds = self.reasoner.model.model.embed_tokens(bos)
        if img_embeds is not None:
            p_before = "<|im_start|>main user 'Gökdeniz Gülemz'\n<|image_start|>"
            p_before_tokens = self.tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(self.device)
            if self.args['freeze_lm']:
                p_before_embeds = self.reasoner.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            else:
                p_before_embeds = self.reasoner.model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            inputs_embeds = torch.cat([bos_embeds, p_before_embeds, img_embeds, p_after_embeds], dim=1).to(self.device)

            # create targets
            empty_targets = (torch.ones([batch_size, 1 + p_before_embeds.size()[1] + 1], dtype=torch.long).to(self.device).fill_(-100))
            targets = torch.cat([empty_targets, target_ids], dim=1).to(self.device)  # bsz x (1 + s1 + 1 + s2)
            assert inputs_embeds.size()[1] == targets.size()[1]

            atts_prefix = torch.ones([batch_size, 1 + p_before_embeds.size()[1] + 1], dtype=torch.long).to(self.device)
            attention_mask = torch.cat([atts_prefix, attention_mask], dim=1).to(self.device)
            assert attention_mask.size() == targets.size()  # bsz x (1 + s1 + 1 + s2)
        else:
            p_before = "<|im_start|>main user 'Gökdeniz Gülemz'\n"
            p_before_tokens = self.tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(self.device)
            if self.args['freeze_lm']:
                p_before_embeds = self.reasoner.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            else:
                p_before_embeds = self.reasoner.model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            inputs_embeds = torch.cat([bos_embeds, p_before_embeds, p_after_embeds], dim=1).to(self.device)

            empty_targets = (torch.ones([batch_size, 1 + p_before_embeds.size()[1]], dtype=torch.long).to(self.device).fill_(-100))
            targets = torch.cat([empty_targets, target_ids], dim=1).to(self.device)
            assert inputs_embeds.size()[1] == targets.size()[1]

            atts_prefix = torch.ones([batch_size, 1 + p_before_embeds.size()[1]], dtype=torch.long).to(self.device)
            attention_mask = torch.cat([atts_prefix, attention_mask], dim=1).to(self.device)
            assert attention_mask.size() == targets.size()
        return inputs_embeds, targets, attention_mask


    def prompt_wrap_new(self, img_embeds, input_ids, target_ids, attention_mask):
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        batch_size = input_ids.shape[0]

        # 1. Precompute common elements
        bos = torch.full((batch_size, 1), self.tokenizer.bos_token_id, dtype=input_ids.dtype, device=self.device)
        embed_fn = self.reasoner.model.embed_tokens if self.args['freeze_lm'] else self.reasoner.model.model.embed_tokens

        # 2. Use torch.cat for concatenation instead of multiple .expand() calls
        p_before = "<|im_start|>main user 'Gökdeniz Gülemz'\n"
        p_before += "<|image_start|>" if img_embeds is not None else ""
        p_before_tokens = self.tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(self.device)

        # 3. Combine embeddings in a single operation
        embeds_list = [
            embed_fn(bos),
            embed_fn(p_before_tokens.input_ids).expand(batch_size, -1, -1),
            img_embeds if img_embeds is not None else torch.tensor([]),
            embed_fn(input_ids)
        ]
        inputs_embeds = torch.cat([emb for emb in embeds_list if emb.numel() > 0], dim=1)

        # 4. Simplify target and attention mask creation
        prefix_length = inputs_embeds.size(1) - input_ids.size(1)
        empty_targets = torch.full((batch_size, prefix_length), -100, dtype=torch.long, device=self.device)
        targets = torch.cat([empty_targets, target_ids], dim=1)

        atts_prefix = torch.ones((batch_size, prefix_length), dtype=torch.long, device=self.device)
        attention_mask = torch.cat([atts_prefix, attention_mask], dim=1)

        return inputs_embeds, targets, attention_mask


    def _encoder_alignment_training_stage_1(self, inputs):
        """
        In this stage: training the encoding-side alignment via image/video/audio caption tasks
        modality: the input modality for each caption task, it could be 'image', 'video' or 'audio'.
        """
        dataset_type = inputs['dataset_types'][0]
        if dataset_type == 'ImageToEmbeddings':
            image_paths = inputs['mm_paths']
            mm_embeds, _ = self.encode_image(image_paths)
        elif dataset_type == 'VideoToEmbeddings':
            video_paths = inputs['mm_paths']
            mm_embeds, _ = self.encode_video(video_paths)
        elif dataset_type == 'AudioToEmbeddings':
            audio_paths = inputs['mm_paths']
            mm_embeds, _ = self.encode_audio(audio_paths)
        else:
            raise NotImplementedError
        input_ids, target_ids, attention_mask = process_batch_stage_1(self.llama_tokenizer, inputs['output_texts'], self.max_length, self.args['prompt'])
        # print(input_ids)
        inputs_embeds, targets, attention_mask = self.prompt_wrap(mm_embeds, input_ids, target_ids, attention_mask)
        outputs = self.reasoner(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
            labels=targets,
        )

        loss = outputs.loss
        # calculate the token accuracy
        chosen_tokens = torch.max(outputs.logits, dim=-1)[1][:, 1:-1]  # [B, S-1]
        labels = targets[:, 2:]
        gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)  # [B*S]
        valid_mask = (labels != -100).reshape(-1)
        valid_tokens = gen_acc & valid_mask  # [B*S]
        gen_acc = valid_tokens.sum().item() / (valid_mask.sum().item() + 1.0)
        return loss, gen_acc
