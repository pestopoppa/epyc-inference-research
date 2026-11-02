#!/usr/bin/env python3
"""Fail-closed binding of the intended MI210 PCI/DRM/KFD identities.

The KFD GPU id and DRM card number are discovered, never assumed stable. The
specific production host device is pinned by PCI BDF, AMD device id and unique
id. A changed topology refuses rather than sampling an unrelated GPU.
"""
from pathlib import Path
import re

DRM=Path('/sys/class/drm')
KFD=Path('/sys/class/kfd/kfd/topology/nodes')
BDF='0000:43:00.0'
VENDOR=0x1002
DEVICE=0x740f
UNIQUE=0xc6cb1cf088bd97ec
LOCATION=0x4300
DOMAIN=0


def _properties(path):
    out={}
    for line in path.read_text().splitlines():
        parts=line.split()
        if len(parts)!=2 or parts[0] in out:
            raise ValueError('invalid KFD properties')
        out[parts[0]]=int(parts[1],0)
    return out


def discover(drm_root=DRM,kfd_root=KFD):
    drm_root,kfd_root=Path(drm_root),Path(kfd_root)
    cards=[]
    for card in drm_root.iterdir():
        if not re.fullmatch(r'card[0-9]+',card.name):continue
        dev=card/'device'
        if dev.resolve(strict=True).name!=BDF:continue
        if int((dev/'vendor').read_text().strip(),0)!=VENDOR or int((dev/'device').read_text().strip(),0)!=DEVICE:
            raise RuntimeError('matching PCI BDF has wrong vendor/device')
        if int((dev/'unique_id').read_text().strip(),16)!=UNIQUE:
            raise RuntimeError('matching PCI BDF has wrong GPU unique id')
        usage=dev/'mem_info_vram_used'
        if not usage.is_file():raise RuntimeError('matching GPU has no whole-device VRAM counter')
        cards.append((card,dev.resolve(strict=True),usage))
    if len(cards)!=1:raise RuntimeError('expected exactly one matching MI210 DRM card')
    card,device_path,usage=cards[0];nodes=[]
    for node in kfd_root.iterdir():
        if not node.name.isdigit():continue
        props=_properties(node/'properties')
        if props.get('unique_id')!=UNIQUE:continue
        expected={'vendor_id':VENDOR,'device_id':DEVICE,'location_id':LOCATION,'domain':DOMAIN}
        if any(props.get(k)!=v for k,v in expected.items()):raise RuntimeError('KFD GPU topology disagrees with pinned PCI identity')
        gpu_id=int((node/'gpu_id').read_text().strip())
        minor=props.get('drm_render_minor')
        if gpu_id<=0 or not isinstance(minor,int) or minor<128:raise RuntimeError('invalid KFD GPU/render id')
        render=drm_root/f'renderD{minor}'/'device'
        if render.resolve(strict=True)!=device_path:raise RuntimeError('KFD render node does not bind the same PCI device')
        nodes.append((node,gpu_id,minor))
    if len(nodes)!=1:raise RuntimeError('expected exactly one matching MI210 KFD node')
    node,gpu_id,minor=nodes[0]
    return {'bdf':BDF,'vendor':VENDOR,'device':DEVICE,'unique_id':UNIQUE,
            'gpu_id':gpu_id,'render_minor':minor,'card':card.name,'card_vram_path':str(usage),
            'kfd_node':str(node),'drm_root':str(drm_root),'kfd_root':str(kfd_root)}


def revalidate(binding):
    current=discover(binding['drm_root'],binding['kfd_root'])
    if current!=binding:raise RuntimeError('MI210 DRM/KFD device binding changed during run')
    return current
