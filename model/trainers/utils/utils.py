import torch
from torch.nn.utils import rnn

def build_one_instance_stage_2(tokenizer, captions, prompt=''):
    input_ids, target_ids = [], []
    texts = ''
    text = "<|image_end|> " + prompt + "<|im_start|>assistant \n"
    texts += text
    one_input_id = tokenizer(text, add_special_tokens=False).input_ids
    input_ids += one_input_id
    target_ids += [-100] * len(one_input_id)  # do not perform loss regression on human prompt

    text = captions + '\n###'
    texts += text
    one_input_id = tokenizer(text, add_special_tokens=False).input_ids
    input_ids += one_input_id
    target_ids += one_input_id
    return input_ids, target_ids

def process_batch_stage_2(tokenizer, batch_of_captions, max_tgt_len, prompt=''):
    batch_input_ids, batch_target_ids = [], []
    for caption in batch_of_captions:
        one_input_ids, one_target_ids = build_one_instance_stage_2(tokenizer, caption, prompt)
        batch_input_ids.append(torch.LongTensor(one_input_ids))
        batch_target_ids.append(torch.LongTensor(one_target_ids))
    input_ids = rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    target_ids = rnn.pad_sequence(batch_target_ids, batch_first=True, padding_value=-100)
    assert input_ids.size() == target_ids.size()
    input_ids = input_ids[:, :max_tgt_len]
    target_ids = target_ids[:, :max_tgt_len]
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    assert attention_mask.size() == input_ids.size()
    return input_ids, target_ids, attention_mask.long()