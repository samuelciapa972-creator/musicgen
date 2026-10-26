import torch
from scipy.io.wavfile import write
from transformers import AutoProcessor, MusicgenForConditionalGeneration

Model_ID = "facebook/musicgen-small"
DEVICE = "cpu"

PROMPT="Aggressive modern trap banger, hard-hitting 808 bass, rapid hi-hat rolls, dark ominous cinematic synth melodies. A confident male rapper with a fast, choppy, and aggressive flow, auto-tuned chorus, high-energy club anthem, modern pristine mix, 140bpm"

print("Cargando musicgen...")

processor= AutoProcessor.from_pretrained(Model_ID)

model= MusicgenForConditionalGeneration.from_pretrained(Model_ID)

model= model.to(DEVICE)
model.eval()

print("Preparando prompt...")

inputs=processor(text=[PROMPT], padding=True, return_tensors="pt").to(DEVICE)

print("Generando musica ...")

with torch.inference_mode():
    audio = model.generate(**inputs, do_sample=True, guidance_scale=3.0, max_new_tokens=2500) # 5 segundos - 750: 15s

sample_rate=model.config.audio_encoder.sampling_rate
waveform = audio[0,0].cpu().float().numpy()

write("musica.wav", sample_rate, waveform)
print("Musica generada")
