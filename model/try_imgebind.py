from encoder import data
from encoder.models import imagebind_model
from encoder.models.imagebind_model import ModalityType

import torch

text_list=["A dog.", "A car", "A bird"]
image_paths=[".assets/dog_image.jpg", ".assets/car_image.jpg", ".assets/bird_image.jpg"]
audio_paths=[".assets/dog_audio.wav", ".assets/car_audio.wav", ".assets/bird_audio.wav"]

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Instantiate model
model, dim = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

# Load data
inputs = {
    ModalityType.TEXT: data.load_and_transform_text(text_list, device),
    ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device),
    ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, device),
}

with torch.no_grad():
    embeddings = model(inputs)

print("Vision x Text: ", torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T, dim=-1))
print("Audio x Text: ", torch.softmax(embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.TEXT].T, dim=-1))
print("Vision x Audio: ", torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.AUDIO].T, dim=-1))

"""
Vision x Text:  tensor([
        [9.9684e-01, 3.1310e-03, 2.5928e-05],
        [5.4496e-05, 9.9993e-01, 2.0353e-05],
        [4.4848e-05, 1.3246e-02, 9.8671e-01]
    ])
Audio x Text:  tensor([
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]
    ])
Vision x Audio:  tensor([
        [8.2460e-01, 8.0997e-02, 9.4405e-02],
        [1.4594e-01, 6.6145e-01, 1.9261e-01],
        [1.1730e-03, 7.4875e-04, 9.9808e-01]
    ])
"""