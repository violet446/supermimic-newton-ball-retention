"""Read-only Windows resource admission, deliberately independent of Newton/Warp."""
import ctypes, shutil, subprocess
from pathlib import Path

class MemoryStatus(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(name,ctypes.c_ulonglong) for name in ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','avail_extended')]

def snapshot(root):
    m=MemoryStatus(); m.length=ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
        raise OSError('GlobalMemoryStatusEx failed')
    output=subprocess.check_output(['nvidia-smi','--query-gpu=memory.total,memory.used','--format=csv,noheader,nounits'],text=True,timeout=10)
    total,used=[int(x.strip()) for x in output.splitlines()[0].split(',')]
    return {'available_ram_bytes':m.avail_phys,'total_ram_bytes':m.total_phys,
            'gpu_total_bytes':total*1024**2,'gpu_used_bytes':used*1024**2,
            'disk_free_bytes':shutil.disk_usage(root).free}

def admission(sample,budget):
    checks={'ram':sample['available_ram_bytes']>=budget['min_start_ram_gib']*1024**3,
            'gpu_free':sample['gpu_total_bytes']-sample['gpu_used_bytes']>=budget['min_start_gpu_free_gib']*1024**3,
            'gpu_used':sample['gpu_used_bytes']<=budget['max_gpu_used_gib']*1024**3,
            'disk':sample['disk_free_bytes']>=budget['min_disk_free_gib']*1024**3}
    return {'passed':all(checks.values()),'checks':checks}

def directory_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file())
