from utils import process_batch_stage_2
import torch


def _encoder_alignment_training_stage_2(self, model, inputs):
        """
        In this stage: training the encoding-side alignment via image/video/audio caption tasks
        modality: the input modality for each caption task, it could be 'image', 'video' or 'audio'.
        """
        dataset_type = inputs['dataset_types'][0]
        if dataset_type == 'ImageToEmbeddings':
            image_paths = inputs['mm_paths']
            mm_embeds, _ = self.model.encode_image(image_paths)
        elif dataset_type == 'VideoToEmbeddings':
            video_paths = inputs['mm_paths']
            mm_embeds, _ = self.model.encode_video(video_paths)
        elif dataset_type == 'AudioToEmbeddings':
            audio_paths = inputs['mm_paths']
            mm_embeds, _ = self.model.encode_audio(audio_paths)
        else:
            raise NotImplementedError
        input_ids, target_ids, attention_mask = process_batch_stage_2(self.tokenizer, inputs['output_texts'], self.max_length, self.args['prompt'])
        print(input_ids)
        inputs_embeds, targets, attention_mask = self.model.prompt_wrap(mm_embeds, input_ids, target_ids, attention_mask)
        outputs = self.model.reasoner(
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