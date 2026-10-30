#!/usr/bin/env python3
"""Capture and safely signal only the exact process instance we spawned."""
import argparse, os, select, signal, sys, time
from pathlib import Path

def identity(pid):
 try:
  proc=Path('/proc')/str(pid); uid=proc.stat().st_uid; text=(proc/'stat').read_text(); fields=text[text.rfind(')')+2:].split()
  if fields[0]=='Z':return None
  return uid,int(fields[19])
 except (FileNotFoundError,ProcessLookupError):return None

def parent(pid):
 try:
  text=(Path('/proc')/str(pid)/'stat').read_text(); return int(text[text.rfind(')')+2:].split()[1])
 except (FileNotFoundError,ProcessLookupError):return None

def stop(pid, expected, grace=5):
 if identity(pid)!=expected:return 'absent-or-reused'
 try: fd=os.pidfd_open(pid)
 except ProcessLookupError:return 'absent'
 try:
  if identity(pid)!=expected:return 'absent-or-reused'
  signal.pidfd_send_signal(fd,signal.SIGTERM)
  if select.select([fd],[],[],grace)[0]:return 'terminated'
  if identity(pid)!=expected:return 'absent-or-reused'
  signal.pidfd_send_signal(fd,signal.SIGKILL)
  if select.select([fd],[],[],grace)[0]:return 'killed'
  raise RuntimeError('captured process did not exit')
 except ProcessLookupError:return 'absent'
 finally:os.close(fd)

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['capture','stop','check']);p.add_argument('pid',type=int);p.add_argument('uid',type=int,nargs='?');p.add_argument('birth',type=int,nargs='?');p.add_argument('--parent',type=int);a=p.parse_args()
 if a.pid<=0:raise ValueError('invalid pid')
 if a.action=='capture':
  v=identity(a.pid)
  if v is None or v[0]!=os.getuid() or a.parent is None or parent(a.pid)!=a.parent or identity(a.pid)!=v:raise SystemExit('process absent or not our captured child')
  print(*v);return
 if a.uid is None or a.birth is None or a.uid!=os.getuid():raise SystemExit('invalid expected identity')
 if a.action=='check':raise SystemExit(0 if identity(a.pid)==(a.uid,a.birth) else 1)
 print(stop(a.pid,(a.uid,a.birth)))
if __name__=='__main__':main()
