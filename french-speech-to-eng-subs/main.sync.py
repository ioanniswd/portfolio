# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: translate
#     language: python
#     name: translate
# ---

# %% [markdown]
# ### Parameters

# %%
video_path = ''

# %%
name_base = video_path.split('.')[0]
original_subs_path = f'{name_base}-ORIGINAL.srt'
eng_subs_path = f'{name_base}-ENG.srt'

# %%
import pysrt
from googletrans import Translator  # unofficial free API
from tqdm import tqdm

import torch
from time import sleep

# %%
from moviepy import VideoFileClip

audio_path = f'{video_path.split(".")[0]}.wav'

video = VideoFileClip(video_path)
video.audio.write_audiofile(audio_path)

# %%
from faster_whisper import WhisperModel

model = WhisperModel("small", device="cuda", compute_type="int8")

segments, info = model.transcribe(audio_path, beam_size=5)

# # segments contains start/end times and text
# for segment in segments:
#     print(f"[{segment.start:.2f} --> {segment.end:.2f}] {segment.text}")

# %%
with open('hf_access_token', 'r') as f:
    hf_access_token = f.read().strip()

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# %%
from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_access_token)
pipeline.to(device)

diarization = pipeline(audio_path)
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.2f}-{turn.end:.2f}: Speaker {speaker}")

# %%
import gc
gc.collect()
torch.cuda.empty_cache()


# %%
def write_srt(segments, diarization=None, out_path="subs.srt"):
    def format_timestamp(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds - int(seconds)) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"
    
    with open(out_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments, 1):
            start = format_timestamp(segment.start)
            end = format_timestamp(segment.end)

            # Optionally find speaker for this segment from diarization
            speaker = ""
            if diarization:
                # Find speaker overlapping this segment start
                for turn, _, spk in diarization.itertracks(yield_label=True):
                    if turn.start <= segment.start <= turn.end:
                        speaker = f"Speaker {spk}: "
                        break

            text = speaker + segment.text.strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")

write_srt(segments, diarization, out_path=original_subs_path)

# %%
ext = 'srt'
original_subs = pysrt.open(original_subs_path)
processed = pysrt.open(original_subs_path)
subs = pysrt.open(original_subs_path)

# %%

# %% [markdown]
# ### Post-process subs

# %%
import re
pattern = r'^Speaker\s+\S+:\s*'

# %%
for sub in processed:
    sub.text = re.sub(pattern, '', sub.text, flags=re.IGNORECASE)

# %%
# write
processed.save(f'{name_base}-PROCESSED.{ext}', encoding='utf-8')

# %%

# %% [markdown]
# Translate subtitles from French to English using Google Translate API

# %%
translator = Translator()

# %%
i = 0

# %%
max_retries = 3
current_retries = 0

# %%
while i < len(subs) and max_retries > current_retries:
    try:
        for sub in tqdm(subs[i:]):
            i += 1
            translated = translator.translate(sub.text, src='fr', dest='en')
            sub.text = translated.text
            
            current_retries = 0
    except:
        print('Failed, will retry')
        current_retries += 1
        print('Retry attempt: ', current_retries)
        sleep(10)
        print('Retrying..')

# %%
i

# %%
assert len(subs) == i, 'Not done yet!'

# %%
# subs.save(f'{name_base}-ENG.{ext}', encoding='utf-8')

# %%
for sub in subs:
    sub.text = re.sub(pattern, '', sub.text, flags=re.IGNORECASE)

subs.save(f'{name_base}-ENG-PROCESSED.{ext}', encoding='utf-8')

# %%
for original, translated in zip(original_subs[0:20], subs[0:20]):
    print(f"Original:   {original.text}")
    print(f"Translated: {translated.text}")
    print()

# %%
for original, translated in zip(original_subs[-20:], subs[-20:]):
    print(f"Original:   {original.text}")
    print(f"Translated: {translated.text}")
    print()

# %%
