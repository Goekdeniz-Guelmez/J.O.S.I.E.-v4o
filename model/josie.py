# The main J.O.S.I.E.v4o model file. It's heavely barrowed from NeXT-GPT, big thanks.
import os, re
from typing import List

from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from imagebind import data

import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops: List = None, encounters: int = 1):
        super().__init__()
        self.stops = stops
        self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        stop_count = 0
        for stop in self.stops:
            _stop = torch.tensor(stop).to(input_ids[0].device)
            indices = torch.where(_stop[0] == input_ids)
            for i in indices:
                if len(i) > 0:
                    if torch.all(input_ids[0][i:i + len(_stop)] == _stop):
                        stop_count += 1
        if stop_count >= self.ENCOUNTERS:
            return True
        return False

class JOSIE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.max_length = args.max_length
        self.stage = args.stage

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
        reasoner_path = os.path.join(self.args.reasoner_path)
        self.reasoner = AutoModelForCausalLM.from_pretrained(reasoner_path)
        self.tokenizer = AutoTokenizer.from_pretrained(reasoner_path)
        self.bos_token_id = self.tokenizer.encode("<|im_start|>", add_special_tokens=False)[0] # Manual Implementaiton beacause Qwen2 has it not set in the Tokenizer config
        print(f"... tokenizer initialized.")

        if self.args.freeze_lm:
            print("Freezing the Reasoner ...")
            for param in self.reasoner.parameters():
                param.requires_grad = False
            self.reasoner.eval()
        else:
            print("Instruct tuning the Reasoner ...") # TODO implement the LoraConfig stuff
        print(f"... Reasoner LLM model initialized.")


        ##### TOKENIZER SETTING STUFF
        if self.args.add_spetial_tokens:
            print("Adding Spetial Tokens to vocabulary ...")
            self._add_image_token()
            self._add_video_token()
            self._add_audio_token()
            self._add_thermal_token()
            self._add_depth_token()
            self._add_imu_token()
            self.reasoner.resize_token_embeddings(len(self.tokenizer))
            print("...  Spetial Tokens added to vocabulary")


        ##### INPUUT PROJECTOR STUFF
        print("Initializing input ImageBind Projection ...")
        self.input_projetor = nn.Linear(self.imagebind_encoder_output_dim, self.reasoner.config.hidden_size)
        if self.args.freeze_input_proj:
            for param in self.input_projetor.parameters():
                param.requires_grad = False
        print("...  Input ImageBind Projection initialized")

        # self.output_projetor = nn.Linear(self.reasoner.config.hidden_size, audio_model_input_dim)

    def _add_image_token(self):
        self.tokenizer.add_tokens(["<|image_start|>"])
        self.tokenizer.add_tokens(["<|image_end|>"])

        # self.args['gen_img_token_idx'] = []
        # for i in range(self.args['num_gen_img_tokens']):
        #     print(f'Adding <|image_{i}|> token to vocabulary.')
        #     print(f'Before adding new token, tokenizer("<|image_{i}|>") =', self.tokenizer(f'<|image_{i}|>', add_special_tokens=False))
        #     num_added_tokens = self.tokenizer.add_tokens(f'<|image_{i}|>')
        #     print(f'After adding {num_added_tokens} new tokens, tokenizer("<|image_{i}|>") =', self.tokenizer(f'<|image_{i}|>', add_special_tokens=False))
        #     gen_token_idx = self.tokenizer(f'<|image_{i}|>', add_special_tokens=False).input_ids
        #     assert len(gen_token_idx) == 1, gen_token_idx
        #     self.args['gen_img_token_idx'].append(gen_token_idx[0])

    def _add_video_token(self):
        self.tokenizer.add_tokens(["<|video_start|>"])
        self.tokenizer.add_tokens(["<|video_end|>"])

        # self.args['gen_video_token_idx'] = []
        # for i in range(self.args['num_gen_video_tokens']):
        #     print(f'Adding <|video_{i}|> token to vocabulary.')
        #     print(f'Before adding new token, tokenizer("<|video_{i}|>") =', self.tokenizer(f'<|video_{i}|>', add_special_tokens=False))
        #     num_added_tokens = self.tokenizer.add_tokens(f'<|video_{i}|>')
        #     print(f'After adding {num_added_tokens} new tokens, tokenizer("<|video_{i}|>") =', self.tokenizer(f'<|video_{i}|>', add_special_tokens=False))
        #     gen_token_idx = self.tokenizer(f'<|video_{i}|>', add_special_tokens=False).input_ids
        #     assert len(gen_token_idx) == 1, gen_token_idx
        #     self.args['gen_video_token_idx'].append(gen_token_idx[0])

    def _add_audio_token(self):
        self.tokenizer.add_tokens(["<|audio_start|>"])
        self.tokenizer.add_tokens(["<|audio_end|>"])

        # self.args['gen_audio_token_idx'] = []
        # for i in range(self.args['num_gen_audio_tokens']):
        #     print(f'Adding <|audio_{i}|> token to vocabulary.')
        #     print(f'Before adding new token, tokenizer("<|audio_{i}|>") =', self.tokenizer(f'<|audio_{i}|>', add_special_tokens=False))
        #     num_added_tokens = self.tokenizer.add_tokens(f'<|audio_{i}|>')
        #     print(f'After adding {num_added_tokens} new tokens, tokenizer("<|audio_{i}|>") =', self.tokenizer(f'<|audio_{i}|>', add_special_tokens=False))
        #     gen_token_idx = self.tokenizer(f'<|audio_{i}|>', add_special_tokens=False).input_ids
        #     assert len(gen_token_idx) == 1, gen_token_idx
        #     self.args['gen_audio_token_idx'].append(gen_token_idx[0])

    def _add_thermal_token(self): # TODO Continue
        self.tokenizer.add_tokens(["<|thermal_start|>"])
        self.tokenizer.add_tokens(["<|thermal_end|>"])

    def _add_depth_token(self): # TODO Continue
        self.tokenizer.add_tokens(["<|depth_start|>"])
        self.tokenizer.add_tokens(["<|depth_end|>"])

    def _add_imu_token(self): # TODO Continue
        self.tokenizer.add_tokens(["<|imu_start|>"])
        self.tokenizer.add_tokens(["<|imu_end|>"])

    def encode_video(self, video_paths):
        print("Encoding Video")
        inputs = {ModalityType.VISION: data.load_and_transform_video_data(video_paths, device)}
        inputs = {key: inputs[key].to(self.reasoner.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = self.imagebind_encoder(inputs)
            video_embeds = embeddings[ModalityType.VISION]
        outputs_input_projetor = self.input_projetor(video_embeds).unsqueeze(1)
        atts_reasoner = torch.ones(outputs_input_projetor.size()[:-1], dtype=torch.long).to(device)
        return outputs_input_projetor, atts_reasoner

    def encode_audio(self, audio_paths):
        print("Encoding Audio")
        inputs = {ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, device)}
        inputs = {key: inputs[key].to(self.reasoner.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = self.imagebind_encoder(inputs)
            audio_embeds = embeddings[ModalityType.AUDIO]
        outputs_input_projetor = self.input_projetor(audio_embeds).unsqueeze(1)
        atts_reasoner = torch.ones(outputs_input_projetor.size()[:-1], dtype=torch.long).to(device)
        return outputs_input_projetor, atts_reasoner

    def encode_image(self, image_paths):
        print("Encoding Image")
        inputs = {ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device)}
        inputs = {key: inputs[key].to(self.reasoner.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = self.imagebind_encoder(inputs)
            image_embeds = embeddings[ModalityType.VISION]
        outputs_input_projetor = self.input_projetor(image_embeds).unsqueeze(1)
        atts_reasoner = torch.ones(outputs_input_projetor.size()[:-1], dtype=torch.long).to(device)
        return outputs_input_projetor, atts_reasoner
    
    def encode_themal_image(self, thermal_paths):
        print("Encoding thermal Image")
        inputs = {ModalityType.THERMAL: data.load_and_transform_vision_data(thermal_paths, device)}
        inputs = {key: inputs[key].to(self.reasoner.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = self.imagebind_encoder(inputs)
            image_embeds = embeddings[ModalityType.VITHERMALSION]
        outputs_input_projetor = self.input_projetor(image_embeds).unsqueeze(1)
        atts_reasoner = torch.ones(outputs_input_projetor.size()[:-1], dtype=torch.long).to(device)
        return outputs_input_projetor, atts_reasoner
    
    def encode_depth_image(self, depth_paths):
        print("Encoding depth Image")
        inputs = {ModalityType.DEPTH: data.load_and_transform_vision_data(depth_paths, device)}
        inputs = {key: inputs[key].to(self.reasoner.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = self.imagebind_encoder(inputs)
            image_embeds = embeddings[ModalityType.DEPTH]
        outputs_input_projetor = self.input_projetor(image_embeds).unsqueeze(1)
        atts_reasoner = torch.ones(outputs_input_projetor.size()[:-1], dtype=torch.long).to(device)
        return outputs_input_projetor, atts_reasoner
    
    def encode_imu(self, imu_paths):
        print("Encoding Imu")
        inputs = {ModalityType.IMU: data.load_and_transform_vision_data(imu_paths, device)}
        inputs = {key: inputs[key].to(self.reasoner.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = self.imagebind_encoder(inputs)
            image_embeds = embeddings[ModalityType.IMU]
        outputs_input_projetor = self.input_projetor(image_embeds).unsqueeze(1)
        atts_reasoner = torch.ones(outputs_input_projetor.size()[:-1], dtype=torch.long).to(device)
        return outputs_input_projetor, atts_reasoner


    def prompt_wrap_old(self, img_embeds, input_ids, target_ids, attention_mask):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        attention_mask = attention_mask.to(device)
        batch_size = input_ids.shape[0]
        bos = torch.ones([batch_size, 1], dtype=input_ids.dtype, device=input_ids.device) * self.bos_token_id
        if self.args['freeze_lm']:
            p_after_embeds = self.reasoner.model.embed_tokens(input_ids).expand(batch_size, -1, -1)
            bos_embeds = self.reasoner.model.embed_tokens(bos)
        else:
            p_after_embeds = self.reasoner.model.model.embed_tokens(input_ids).expand(batch_size, -1, -1)
            bos_embeds = self.reasoner.model.model.embed_tokens(bos)
        if img_embeds is not None:
            print("Recieved Image")
            p_before = "<|im_end|>\n<|im_start|>user\n<|image_start|>"
            p_before_tokens = self.tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(device)
            if self.args['freeze_lm']:
                p_before_embeds = self.reasoner.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            else:
                p_before_embeds = self.reasoner.model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            inputs_embeds = torch.cat([bos_embeds, p_before_embeds, img_embeds, p_after_embeds], dim=1).to(device)

            # create targets
            empty_targets = (torch.ones([batch_size, 1 + p_before_embeds.size()[1] + 1], dtype=torch.long).to(device).fill_(-100))
            targets = torch.cat([empty_targets, target_ids], dim=1).to(device)
            assert inputs_embeds.size()[1] == targets.size()[1]

            atts_prefix = torch.ones([batch_size, 1 + p_before_embeds.size()[1] + 1], dtype=torch.long).to(device)
            attention_mask = torch.cat([atts_prefix, attention_mask], dim=1).to(device)
            assert attention_mask.size() == targets.size()
        else:
            p_before = "<|im_end|>\n<|im_start|>user\n"
            p_before_tokens = self.tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(device)
            if self.args['freeze_lm']:
                p_before_embeds = self.reasoner.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            else:
                p_before_embeds = self.reasoner.model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            inputs_embeds = torch.cat([bos_embeds, p_before_embeds, p_after_embeds], dim=1).to(device)
            empty_targets = (torch.ones([batch_size, 1 + p_before_embeds.size()[1]], dtype=torch.long).to(device).fill_(-100))
            targets = torch.cat([empty_targets, target_ids], dim=1).to(device)
            assert inputs_embeds.size()[1] == targets.size()[1]
            atts_prefix = torch.ones([batch_size, 1 + p_before_embeds.size()[1]], dtype=torch.long).to(device)
            attention_mask = torch.cat([atts_prefix, attention_mask], dim=1).to(device)
            assert attention_mask.size() == targets.size()
        return inputs_embeds, targets, attention_mask


    def prompt_wrap_new(self, img_embeds, input_ids, target_ids, attention_mask):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        attention_mask = attention_mask.to(device)
        batch_size = input_ids.shape[0]

        # 1. Precompute common elements
        bos = torch.full((batch_size, 1), self.bos_token_id, dtype=input_ids.dtype, device=device)
        embed_fn = self.reasoner.model.embed_tokens if self.args['freeze_lm'] else self.reasoner.model.model.embed_tokens

        # 2. Use torch.cat for concatenation instead of multiple .expand() calls
        p_before = "<|im_end|>\n<|im_start|>user\n"
        p_before += "<|image_start|>" if img_embeds is not None else ""
        p_before_tokens = self.tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(device)

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
        empty_targets = torch.full((batch_size, prefix_length), -100, dtype=torch.long, device=device)
        targets = torch.cat([empty_targets, target_ids], dim=1)

        atts_prefix = torch.ones((batch_size, prefix_length), dtype=torch.long, device=device)
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
        input_ids, target_ids, attention_mask = process_batch_stage_1(self.tokenizer, inputs['output_texts'], self.max_length, self.args['prompt'])
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
    

    def _prepare_image_embed(self, inputs, batch_size):
        features = []
        p_before_token = self.tokenizer('<|image_start|>', add_special_tokens=False, return_tensors='pt').to(device)
        p_after_token = self.tokenizer('<|image_end|>', add_special_tokens=False, return_tensors='pt').to(device)
        
        p_before_embeds = self.reasoner.model.embed_tokens(p_before_token.input_ids).expand(batch_size, -1, -1)
        p_after_embeds = self.reasoner.model.embed_tokens(p_after_token.input_ids).expand(batch_size, -1, -1)

        _temp_embedding, _ = self.encode_image(inputs['image_paths'])
        features.append(_temp_embedding)

        feature_embeds = torch.cat(features).sum(dim=0).unsqueeze(0)
        return torch.cat([p_before_embeds, feature_embeds, p_after_embeds], dim=1)

    def _prepare_video_embed(self, inputs, batch_size):
        features = []
        p_before_token = self.tokenizer('<|video_start|>', add_special_tokens=False, return_tensors='pt').to(device)
        p_after_token = self.tokenizer('<|video_end|>', add_special_tokens=False, return_tensors='pt').to(device)

        p_before_embeds = self.reasoner.model.embed_tokens(p_before_token.input_ids).expand(batch_size, -1, -1)
        p_after_embeds = self.reasoner.model.embed_tokens(p_after_token.input_ids).expand(batch_size, -1, -1)

        _temp_embedding, _ = self.encode_video(inputs['video_paths'])
        features.append(_temp_embedding)

        feature_embeds = torch.cat(features).sum(dim=0).unsqueeze(0)
        return torch.cat([p_before_embeds, feature_embeds, p_after_embeds], dim=1)

    def _prepare_audio_embed(self, inputs, batch_size):
        features = []
        p_before_token = self.tokenizer('<|audio_start|>', add_special_tokens=False, return_tensors='pt').to(device)
        p_after_token = self.tokenizer('<|audio_end|>', add_special_tokens=False, return_tensors='pt').to(device)

        p_before_embeds = self.reasoner.model.embed_tokens(p_before_token.input_ids).expand(batch_size, -1, -1)
        p_after_embeds = self.reasoner.model.embed_tokens(p_after_token.input_ids).expand(batch_size, -1, -1)

        _temp_embedding, _ = self.encode_audio(inputs['audio_paths'])
        features.append(_temp_embedding)

        feature_embeds = torch.cat(features).sum(dim=0).unsqueeze(0)
        return torch.cat([p_before_embeds, feature_embeds, p_after_embeds], dim=1)
    
    def _prepare_thermal_embed(self, inputs, batch_size):
        features = []
        p_before_token = self.tokenizer('<|thermal_start|>', add_special_tokens=False, return_tensors='pt').to(device)
        p_after_token = self.tokenizer('<|thermal_end|>', add_special_tokens=False, return_tensors='pt').to(device)

        p_before_embeds = self.reasoner.model.embed_tokens(p_before_token.input_ids).expand(batch_size, -1, -1)
        p_after_embeds = self.reasoner.model.embed_tokens(p_after_token.input_ids).expand(batch_size, -1, -1)

        _temp_embedding, _ = self.encode_themal_image(inputs['thermal_paths'])
        features.append(_temp_embedding)

        feature_embeds = torch.cat(features).sum(dim=0).unsqueeze(0)
        return torch.cat([p_before_embeds, feature_embeds, p_after_embeds], dim=1)
    
    def _prepare_depth_embed(self, inputs, batch_size):
        features = []
        p_before_token = self.tokenizer('<|depth_start|>', add_special_tokens=False, return_tensors='pt').to(device)
        p_after_token = self.tokenizer('<|depth_end|>', add_special_tokens=False, return_tensors='pt').to(device)

        p_before_embeds = self.reasoner.model.embed_tokens(p_before_token.input_ids).expand(batch_size, -1, -1)
        p_after_embeds = self.reasoner.model.embed_tokens(p_after_token.input_ids).expand(batch_size, -1, -1)

        _temp_embedding, _ = self.encode_depth_image(inputs['depth_paths'])
        features.append(_temp_embedding)

        feature_embeds = torch.cat(features).sum(dim=0).unsqueeze(0)
        return torch.cat([p_before_embeds, feature_embeds, p_after_embeds], dim=1)
    
    def _prepare_imu_embed(self, inputs, batch_size):
        features = []
        p_before_token = self.tokenizer('<|imu_start|>', add_special_tokens=False, return_tensors='pt').to(device)
        p_after_token = self.tokenizer('<|imu_end|>', add_special_tokens=False, return_tensors='pt').to(device)

        p_before_embeds = self.reasoner.model.embed_tokens(p_before_token.input_ids).expand(batch_size, -1, -1)
        p_after_embeds = self.reasoner.model.embed_tokens(p_after_token.input_ids).expand(batch_size, -1, -1)

        _temp_embedding, _ = self.encode_imu(inputs['imu_paths'])
        features.append(_temp_embedding)

        feature_embeds = torch.cat(features).sum(dim=0).unsqueeze(0)
        return torch.cat([p_before_embeds, feature_embeds, p_after_embeds], dim=1)
    

    # def extract_multimodal_feature(self, inputs):
    #     features = []
    #     if inputs['image_paths']:
    #         image_embeds, _ = self.encode_image(inputs['image_paths'])
    #         features.append(image_embeds)
    #     if inputs['audio_paths']:
    #         audio_embeds, _ = self.encode_audio(inputs['audio_paths'])
    #         features.append(audio_embeds)
    #     if inputs['video_paths']:
    #         video_embeds, _ = self.encode_video(inputs['video_paths'])
    #         features.append(video_embeds)
    #     if inputs[thermal_paths']:
    #         thermal_embeds, _ = self.encode_themal_image(inputs['thermal_paths'])
    #         features.append(thermal_embeds)
    #     if inputs['depth_paths']:
    #         depth_embeds, _ = self.encode_depth_image(inputs['depth_paths'])
    #         features.append(depth_embeds)
    #     if inputs['imu_paths']:
    #         imu_embeds, _ = self.encode_imu(inputs['imu_paths'])
    #         features.append(imu_embeds)

    #     feature_embeds = torch.cat(features).sum(dim=0).unsqueeze(0)
    #     return feature_embeds

    def prepare_generation_embedding(self, inputs):
        prompt = inputs['prompt']
        text = f"<|im_start|>system\nYou are a assistant<|im_end|>\n<|im_start|>user\n{prompt}"
        print("text prompt: ", text)
        input()
        batch_size = 1
        input_embeds = []
        if 'image_paths' in inputs:
            text += ' <|image_start|> '
        if 'audio_paths' in inputs:
            text += ' <|audio_start|> '
        if 'video_paths' in inputs:
            text += ' <|video_start|> '
        if 'thermal_paths' in inputs:
            text += ' <|thermal_start|> '
        if 'depth_paths' in inputs:
            text += ' <|depth_start|> '
        if 'imu_paths' in inputs:
            text += ' <imu_start|> '
        print(text)
        input()

        split_text = re.split(r'(<\|image_start\|>|<\|audio_start\|>|<\|video_start\|>)', text)
        for st in split_text:
            if st.startswith('<|image_start|>'):
                print("recieved image")
                input_embeds.append(self._prepare_image_embed(inputs, batch_size))
            elif st.startswith('<|audio_start|>'):
                print("recieved audio")
                input_embeds.append(self._prepare_audio_embed(inputs, batch_size))
            elif st.startswith('<|video_start|>'):
                print("recieved video")
                input_embeds.append(self._prepare_video_embed(inputs, batch_size))
            elif st.startswith('<|thermal_start|>'):
                print("recieved thermal")
                input_embeds.append(self._prepare_thermal_embed(inputs, batch_size))
            elif st.startswith('<|depth_start|>'):
                print("recieved depth")
                input_embeds.append(self._prepare_depth_embed(inputs, batch_size))
            elif st.startswith('<|imu_start|>'):
                print("recieved Imu")
                input_embeds.append(self._prepare_imu_embed(inputs, batch_size))
            else:
                text_tokens = self.tokenizer(st, add_special_tokens=False, return_tensors='pt').to(device)
                bos = torch.ones([batch_size, 1], dtype=text_tokens.input_ids.dtype, device=text_tokens.input_ids.device) * self.bos_token_id
                text_embeds = self.reasoner.model.embed_tokens(text_tokens.input_ids).expand(batch_size, -1, -1)
                bos_embeds = self.reasoner.model.embed_tokens(bos)
                input_embeds.append(bos_embeds)
                input_embeds.append(text_embeds)
        inputs_embeds = torch.cat(input_embeds, dim=1)
        return inputs_embeds
    
    def generate_tokens_embeddings(self, inputs, input_embeds, temperature: float = 0.0, top_p: float = 1.0):
        """
        This function is used to generate the tokens
        inputs: dict
        input_embeds: tensor
        return:
            out: the output tokens index
        """
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=inputs['stops_id'], encounters=1)])

        outputs = self.reasoner.generate(
            inputs_embeds=input_embeds,
            max_new_tokens=inputs['max_tgt_len'],
            top_p=inputs['top_p'],
            temperature=inputs['temperature'],
            # repeat_pen,
            do_sample=True,
            use_cache=True,
            stopping_criteria=stopping_criteria,
            output_hidden_states=True,
            return_dict_in_generate=True,
            output_attentions=True
        )

        return outputs.sequences

    def generate(self, inputs):
        """
        inputs = {
            'image_paths': optional,
            'audio_paths': optional,
            'video_paths': optional,
            'thermal_paths': optional,
            'depth_paths': optional,
            'imu_paths': optional,
            'mode': generation mode,
            'prompt': human input prompt,
            'max_tgt_len': generation length,
            'top_p': top_p,
            'temperature': temperature,
            'modality_embeds': None or torch.tensor,
            'modality_cache': save the image cache,
            'filter_value': Value to assign to tokens that should never be generated,
            'min_word_tokens': Minimum number of words to generate before allowing a [IMG] output,
            'gen_scale_factor': float = 1.0,
            'stops_id': the default value is [[835], [2277, 29937]] the stop token is '###', which has two types of tokenization ways, [835] and [2277, 29937],
            'ENCOUNTERS': the times that the generated sentence will be ended,
            'load_sd': whether use SD for image generation,
            'max_num_imgs': Maximum number of images to return in one generation pass,
            'guidance_scale_for_img': the guidance ratio of conditioner, if it is None, the default value will be applied in SD,
            'num_inference_steps_for_img': the number of inference step for image generation in the stable diffusion model,
            'load_vd': whether use VD for video generation,
            'max_num_vids': Maximum number of videos to return in one generation pass,
            'guidance_scale_for_vid': the guidance ratio of conditioner, if it is None, the default value will be applied in VD,
            'num_inference_steps_for_vid': the number of inference step for video generation in the stable diffusion model,
            'height': (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor),
            'width': (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor),
            'num_frames': (`int`, *optional*, defaults to 16),
            'load_ad': whether use AD for audio generation,
            'max_num_auds': Maximum number of audios to return in one generation pass,
            'guidance_scale_for_aud': the guidance ratio of conditioner, if it is None, the default value will be applied in AD,
            'num_inference_steps_for_aud': the number of inference step for audio generation in the stable diffusion model,
            'audio_length_in_s': the seconds for generated audio length
        }
        """

        input_embeds = self.prepare_generation_embedding(inputs)
        print("Generaded and Prepared the inputs:")
        print(input_embeds)

        generated_ids = self.generate_tokens_embeddings(inputs, input_embeds)
        print("Generates IDs from the Reasoner:")
        print(generated_ids)

        output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print("Tokenized the Outputs")
        return output

