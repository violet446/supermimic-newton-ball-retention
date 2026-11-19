"""Resource decisions independent of simulator imports; polled by parent process."""
import ctypes

class ProcessMemory(ctypes.Structure):
    _fields_=[('cb',ctypes.c_ulong),('faults',ctypes.c_ulong)]+[(n,ctypes.c_size_t) for n in ('peak_ws','ws','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile')]

def process_memory(pid):
    kernel=ctypes.windll.kernel32; kernel.OpenProcess.restype=ctypes.c_void_p
    kernel.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_bool,ctypes.c_ulong]
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=kernel.OpenProcess(0x0400|0x0010,False,pid)
    if not handle: raise OSError('cannot open owned worker for memory monitoring')
    try:
        m=ProcessMemory(); m.cb=ctypes.sizeof(m)
        fn=ctypes.windll.psapi.GetProcessMemoryInfo; fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_ulong]
        if not fn(handle,ctypes.byref(m),m.cb): raise OSError('cannot read worker memory')
        return {'rss_bytes':m.ws,'peak_rss_bytes':m.peak_ws,'private_commit_bytes':m.pagefile,'peak_private_commit_bytes':m.peak_pagefile}
    finally: kernel.CloseHandle(handle)

def process_rss(pid):
    return process_memory(pid)['rss_bytes']

def stop_reason(sample,budget,elapsed_s,rss_bytes,output_cache_bytes,low_ram_s,private_commit_bytes=0):
    if elapsed_s>=budget['wall_s']: return 'wall_budget'
    if rss_bytes>budget['max_process_rss_gib']*1024**3: return 'process_memory_budget'
    if private_commit_bytes>budget.get('max_process_commit_gib',float('inf'))*1024**3: return 'process_commit_budget'
    if sample.get('available_ram_bytes',float('inf'))<budget.get('critical_ram_gib',0)*1024**3: return 'critical_system_ram'
    if sample['gpu_used_bytes']>budget['max_gpu_used_gib']*1024**3: return 'gpu_memory_budget'
    if sample['disk_free_bytes']<budget['min_disk_free_gib']*1024**3: return 'disk_free_floor'
    if output_cache_bytes>budget['max_output_cache_gib']*1024**3: return 'output_cache_budget'
    if low_ram_s>=budget.get('low_ram_grace_s',5): return 'system_available_ram_below_2gib_for_5s' if 'min_running_ram_gib' not in budget else 'system_available_ram_floor'
    return None
