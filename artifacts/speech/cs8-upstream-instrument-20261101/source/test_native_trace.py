#!/usr/bin/env python3
"""Compile the exact source trace class with a model-free driver."""
import os, subprocess, tempfile
from pathlib import Path
from sidecar_reader import read
D=Path(__file__).resolve().parent
src=(D/'tools/mtmd/mtmd-cli.cpp').read_text()
start=src.index('struct cs8_trace {'); end=src.index('\n};',start)+3
harness='#include <cstdio>\n#include <cstdlib>\n#include <string>\n#include <unistd.h>\nusing llama_token=int;\n'+src[start:end]+r'''
int main(int argc,char** argv) {
  cs8_trace t("stream"); if (t.failed) return 2;
  if (argc>1 && argv[1][0]=='x') return t.finish("unknown","token_limit") ? 0 : 3;
  if (argc>1 && argv[1][0]=='t') { t.token(7,"hi"); t.token(8,""); }
  return t.finish("complete","stream_end") ? 0 : 3;
}
'''
with tempfile.TemporaryDirectory(dir=D) as td:
    p=Path(td); (p/'test.cpp').write_text(harness)
    subprocess.run(['g++','-std=c++17','-O0',str(p/'test.cpp'),'-o',str(p/'test')],check=True)
    def run(name,arg):
        env={**os.environ,'CS8_TRANSCRIPT_PATH':str(p/name)}
        return subprocess.run([str(p/'test'),arg],env=env,check=False).returncode
    assert run('text','t')==0
    result=read((p/'text').read_bytes()); assert result['text']=='hi' and result['token_ids']==[7,8]
    assert run('empty','e')==0 and read((p/'empty').read_bytes())['text']==''
    assert run('unknown','x')==0
    try: read((p/'unknown').read_bytes())
    except ValueError: pass
    else: raise AssertionError('unknown accepted')
    assert run('text','t')==2
    raw=(p/'text').read_bytes()
    try: read(raw[:-3])
    except ValueError: pass
    else: raise AssertionError('truncated accepted')
print('native trace writer: text/control IDs, known empty, unknown footer, exclusive create, truncation PASS')
