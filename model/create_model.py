from josie import JOSIE

class Args():
    imagebind_encoder_path = "/Users/gokdenizgulmez/Desktop/J.O.S.I.E.-v4o/models/imagebind_huge.pth"
    reasoner_path = "Qwen/Qwen2-0.5B-Instruct"
    freeze_lm = True
    freeze_input_proj = False
    add_spetial_tokens = True
    max_length = 512

model = JOSIE(Args())

print(model)
