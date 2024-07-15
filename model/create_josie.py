import torch

from josie import JOSIE

class Args():
    imagebind_encoder_path = "/Users/gokdenizgulmez/Desktop/J.O.S.I.E.-v4o/models/imagebind_huge.pth"
    reasoner_path = "Qwen/Qwen2-0.5B-Instruct"
    freeze_lm = True
    freeze_input_proj = False
    add_spetial_tokens = False
    max_length = 512

    text_emb_to_img_layers = [-1]
    text_emb_to_video_layers = [-1]
    text_emb_to_audio_layers = [-1]
    stage = 1

model = JOSIE(Args())

print("Loading model...")
# model.load_state_dict(torch.load("/Users/gokdenizgulmez/Desktop/first_working_creation.pth"))
model.eval()
print("... Model loaded")

# print(model)

# with open("/Users/gokdenizgulmez/Desktop/J.O.S.I.E.-v4o/model_architecture.txt", "w") as file:
#     file.write(str(model))


# Define inputs
inputs = {
    'image_paths': ['/Users/gokdenizgulmez/Desktop/J.O.S.I.E.-v4o/.assets/bird_image.jpg'],  # Path to your image file
    'audio_paths': ['/Users/gokdenizgulmez/Desktop/J.O.S.I.E.-v4o/.assets/bird_audio.wav'],  # Path to your audio file
    'prompt': 'Describe the image and audio:',
    'max_tgt_len': 50,
    'top_p': 0.9,
    'temperature': 0.7,
    'stops_id': [[151645]],  # Adjust this if necessary
    'ENCOUNTERS': 1,
    'max_num_imgs': 1,
    'max_num_vids': 0,
    'max_num_auds': 1,
    'guidance_scale_for_img': 7.5,
    'num_inference_steps_for_img': 40,
    'guidance_scale_for_vid': 7.5,
    'num_inference_steps_for_vid': 40,
    'height': 320,
    'width': 576,
    'num_frames': 16,
    'guidance_scale_for_aud': 7.5,
    'num_inference_steps_for_aud': 40,
    'audio_length_in_s': 5.0
}


# Generate tokens
output_tokens = model.generate(inputs)

# Print the generated tokens
print(output_tokens)


# torch.save(model.state_dict(), "/Users/gokdenizgulmez/Desktop/first_working_creation.pth")

# print("Model's state_dict:")
# for param_tensor in model.state_dict():
#     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
