Just a very simple notebook to add English subtitles to a French video. The
notebook:
1. Extracts the audio from the video.
2. Transcribes the audio to text.
3. Usees diarization to identify speakers and their speech segments.
4. Translates the text to English.

Right now, the translation is somewhat naive as it takes each segment and translates it separately. As a next step, we could use a more sophisticated approach to maintain context across segments, translating entire sentences or paragraphs at once, and then splitting the translated text into segments that match the original speech segments.
