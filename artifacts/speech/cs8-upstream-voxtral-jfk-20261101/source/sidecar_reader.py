#!/usr/bin/env python3
"""Read only fully terminated native CS8TRACE sidecars; unknown never scores."""
import re

HEADER=re.compile(rb'CS8TRACE 1 (chat|stream)\n')
TOKEN=re.compile(rb'T (-?[0-9]+) ([0-9]+)\n')
END=re.compile(rb'E (complete|unknown) ([a-z_]+)\n')

def read(data):
    if len(data)>64*1024*1024: raise ValueError('sidecar oversized')
    m=HEADER.match(data)
    if not m: raise ValueError('missing header')
    mode=m.group(1).decode(); pos=m.end(); pieces=[]; tokens=[]
    while True:
        m=TOKEN.match(data,pos)
        if not m: break
        token=int(m.group(1)); size=int(m.group(2)); start=m.end(); end=start+size
        if size>1024*1024 or end>=len(data) or data[end:end+1]!=b'\n': raise ValueError('truncated piece')
        tokens.append(token); pieces.append(data[start:end]); pos=end+1
    m=END.match(data,pos)
    if not m or m.end()!=len(data): raise ValueError('missing or trailing end record')
    status,reason=(x.decode() for x in m.groups())
    if status != 'complete' or reason not in ('eog','stream_end'): raise ValueError('native completion unknown')
    if mode=='chat' and reason!='eog': raise ValueError('invalid mode/reason')
    text=b''.join(pieces).decode('utf-8','strict')
    return {'mode':mode,'status':status,'reason':reason,'token_ids':tokens,'text':text}
