#!/usr/bin/env python3
"""STT8 command and provisional framed-output adapter, no native execution."""
import re
from pathlib import Path

PREFIX = Path('/mnt/raid0/llm/tmp/codex3-cs8-stt8-20261101/prefix-cpu/bin/llama-mtmd-cli')
MODELS = {
 'lfm3': ('/mnt/raid0/llm/models/LiquidAI/LFM3-Audio-2B-GGUF/LFM3-Audio-2B-Q8_0.gguf',
          '/mnt/raid0/llm/models/LiquidAI/LFM3-Audio-2B-GGUF/mmproj-LFM3-Audio-2B-BF16.gguf'),
 'voxtral': ('/mnt/raid0/llm/models/ggml-org/Voxtral-Mini-4B-Realtime-2602-GGUF/Voxtral-Mini-4B-Realtime-2602-Q8_0.gguf',
             '/mnt/raid0/llm/models/ggml-org/Voxtral-Mini-4B-Realtime-2602-GGUF/mmproj-Voxtral-Mini-4B-Realtime-2602-BF16.gguf')}

def command(arm, wav, language):
    if arm not in MODELS or language not in ('en','it'): raise ValueError('invalid arm or language')
    model, projector = MODELS[arm]
    cmd = [str(PREFIX), '-m',model,'--mmproj',projector,'--audio',str(wav),
           '--temp','0','--seed','0','--threads','2','--n-gpu-layers','0','--no-warmup','--predict','1024']
    if arm == 'lfm3':
        cmd += ['--system-prompt',f'Perform ASR in {"English" if language == "en" else "Italian"}.', '-p','<__media__>']
    return cmd

# The streaming source writes both the transcript and these markers with LOG.
# This grammar is provisional until a real noncohort control proves the entire
# stream and completion witnesses. Failure to match means unknown, not empty.
START = re.compile(r'^Transcribing [0-9]+\.[0-9] s of audio, fed in chunks of [0-9]+ ms\n\n', re.M)
END = re.compile(r'\n\nrun_audio_stream: ([0-9]+) tokens, [0-9]+\.[0-9]+ s audio in [0-9]+\.[0-9]+ s \([0-9]+\.[0-9]+x realtime\)\n', re.M)
def provisional_voxtral(stdout, rc):
    if rc != 0: raise ValueError('native failure')
    s = stdout.decode('utf-8', 'strict')
    starts=list(START.finditer(s)); ends=list(END.finditer(s))
    if len(starts)!=1 or len(ends)!=1 or ends[0].start()<starts[0].end():
        raise ValueError('missing/ambiguous streaming frame')
    if int(ends[0].group(1)) >= 1024:
        raise ValueError('token cap reached: completion unknown')
    return s[starts[0].end():ends[0].start()]
