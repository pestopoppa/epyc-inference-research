#!/usr/bin/python3
"""Bounded in-memory proc controls; run only inside own official host grant."""
import errno, pathlib, sys
from unittest.mock import patch
source=pathlib.Path(sys.argv[1]).read_text()
namespace={'__name__':'sampler_fixture'}
exec(compile(source,sys.argv[1],'exec'),namespace)
if '--isolated' in sys.argv:
 import datetime, os, tempfile
 with tempfile.TemporaryDirectory(prefix='fake-guards-',dir=str(pathlib.Path(sys.argv[1]).parent)) as fixture_dir:
  fixture_root=pathlib.Path(fixture_dir);flags=fixture_root/'flags';(flags/'grants').mkdir(parents=True)
  namespace['ROOT']=fixture_root;namespace['FLAGS']=flags
  grant=flags/'grants'/(namespace['ID']+'.grant')
  current=datetime.datetime.now(datetime.timezone.utc)
  valid='regions=mi210_0,184-191 start='+ (current-datetime.timedelta(seconds=1)).isoformat()+' end='+(current+datetime.timedelta(seconds=600)).isoformat()
  with patch.dict(os.environ,{'FILLER_DEADLINE':str(current.timestamp()+500)}),patch.object(os,'sched_getaffinity',return_value={184,185}):
   try:namespace['safe']()
   except FileNotFoundError:pass
   else:raise AssertionError('missing fake grant accepted')
   grant.write_text(valid);namespace['safe']()
   for flag in [flags/'HOLD_fake',fixture_root/'STOP']:
    flag.touch()
    try:namespace['safe']()
    except AssertionError:pass
    else:raise AssertionError('fake abort flag accepted')
    flag.unlink()
   grant.write_text(valid.replace('mi210_0,184-191','mi210_0,0-23'))
   try:namespace['safe']()
   except AssertionError:pass
   else:raise AssertionError('fake wrong grant shape accepted')
   grant.write_text(valid)
  print('isolated fake missing-grant/HOLD/STOP/shape refusal controls passed; not real scheduling authority')
else:
 namespace['safe']()
real_path=pathlib.PurePosixPath
base=str(namespace['BUILD']/'bin')
resident_expression=next(line.strip().split('=',1)[1] for line in source.splitlines() if line.strip().startswith('resident=['))
def qualifies(record):return bool(eval(resident_expression,namespace,{'local':[record]}))
class Child:
 alive=True
 def poll(self):return None if self.alive else 0
child=Child();namespace['child']=child
class ProcPath:
 fault=None;target='task';mask='184-191';foreign=False;changed=False;stat_reads=0;empty=False
 def __init__(self,path):self.path=str(path)
 @property
 def name(self):return self.path.rsplit('/',1)[-1]
 @property
 def parent(self):return ProcPath(self.path.rsplit('/',1)[0])
 def glob(self,pattern):
  if self.path.endswith('/task'):return [] if self.empty else [ProcPath('/proc/42/task/42/status'),ProcPath('/proc/42/task/43/status')]
  if self.path.endswith('/42'):return [ProcPath(self.path+'/vram_0')]
  raise AssertionError(self.path)
 def read_text(self):
  if self.path.endswith('/status'):
   if '/43/' in self.path and self.target=='task' and self.fault:raise self.fault
   return 'Cpus_allowed_list:\t'+self.mask+'\n'
  if self.path.endswith('/stat'):
   ProcPath.stat_reads+=1
   if self.fault and ((self.target=='stat_first' and self.stat_reads==1) or (self.target=='stat_last' and self.stat_reads==2)):raise self.fault
   return '42 (fixture) '+' '.join(['S']+['0']*18+[str(999 if self.changed and self.stat_reads>1 else 123)])
  if self.path.endswith('/maps'):
   if self.target=='maps' and self.fault:raise self.fault
   directory='/foreign' if self.foreign else base
   return '\n'.join('000-fff r-xp 0 0:0 1 '+directory+'/'+lib+'.so' for lib in ['libggml-base','libggml-hip','libggml-cpu'])
  if '/vram_' in self.path:return '4096'
  if self.name in ('pasid','queues'):return 'fixture'
  raise AssertionError(self.path)
 def exists(self):return True
 def resolve(self):return real_path(self.path)
 def __truediv__(self,name):return ProcPath(self.path+'/'+name)
 def __str__(self):return self.path
count=0
for target in ['task','stat_first','stat_last','maps']:
 for fault in [None,FileNotFoundError(errno.ENOENT,'vanished'),ProcessLookupError(errno.ESRCH,'vanished'),PermissionError(errno.EPERM,'denied'),OSError(errno.EIO,'io')]:
  ProcPath.target=target;ProcPath.fault=fault;ProcPath.stat_reads=0
  with patch.object(namespace['pathlib'],'Path',ProcPath):
   try:record=namespace['sample'](42)
   except (PermissionError,OSError) as exc:
    assert fault is exc and exc.errno in (errno.EPERM,errno.EIO);count+=1;continue
  assert fault is None or isinstance(fault,(FileNotFoundError,ProcessLookupError))
  assert qualifies(record)==(fault is None)
  assert record['partial']==(fault is not None)
  if fault is None:assert record['overlap'] and record['tasks']==2 and record['start_ticks']=='123' and record['kfd_detail']=={'pasid':'fixture','queues':'fixture'}
  elif target=='task':assert record['tasks']==1 and record['skipped_tasks'][0]['errno']==fault.errno
  else:assert record['missing_observations'][0]['errno']==fault.errno
  count+=1
for mode in ['affinity','maps','pid_changed','exited','empty_tasks']:
 ProcPath.fault=None;ProcPath.stat_reads=0;ProcPath.mask='0' if mode=='affinity' else '184-191';ProcPath.foreign=mode=='maps';ProcPath.changed=mode=='pid_changed';ProcPath.empty=mode=='empty_tasks';child.alive=mode!='exited'
 with patch.object(namespace['pathlib'],'Path',ProcPath):
  try:record=namespace['sample'](42)
  except AssertionError:
   assert mode in ('affinity','maps');count+=1;continue
 assert mode in ('pid_changed','exited','empty_tasks') and record['partial'] and not qualifies(record);count+=1
assert count==25
print('25 sampler controls passed; fixture data are not native numerical/residency evidence')
