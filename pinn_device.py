"""Inspect this machine and select a bounded training configuration."""
import ctypes
import json
import os
import platform
import subprocess

import torch


def inspect_device():
    report = {"os": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
              "logical_processors": os.cpu_count(), "cuda_available": torch.cuda.is_available()}
    if os.name == "nt":
        script = "$c=Get-CimInstance Win32_Processor; $m=Get-CimInstance Win32_OperatingSystem; $g=Get-CimInstance Win32_VideoController; [pscustomobject]@{cpu=$c.Name;physical_cores=$c.NumberOfCores;total_memory_gib=$m.TotalVisibleMemorySize/1MB;free_memory_gib=$m.FreePhysicalMemory/1MB;gpus=@($g|Select-Object Name,DriverVersion)}|ConvertTo-Json -Depth 4"
        completed = subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True,
                                   text=True, timeout=30, creationflags=subprocess.CREATE_NO_WINDOW)
        if completed.returncode == 0:
            report.update(json.loads(completed.stdout))
    if report["cuda_available"]:
        properties = torch.cuda.get_device_properties(0)
        report.update(selected_device="cuda", accelerator=properties.name,
                      gpu_memory_gib=properties.total_memory/1024**3, batch_size=1024, width=64)
    else:
        report.update(selected_device="cpu", accelerator="CPU", batch_size=256, width=48,
                      reason="No CUDA device available; use tested float64 CPU autograd. Intel integrated graphics is not used.")
    report["torch_threads"] = max(1, min(2, (report["logical_processors"] or 2)//2))
    report["memory_policy"] = "Small batches, 1-hour initial screening horizon; future extension requires validation."
    if report.get("free_memory_gib", 4) < 1:
        report["batch_size"] = 64
        report["width"] = 32
    return report


def resident_memory_mb():
    if os.name != "nt":
        return None
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    process = ctypes.c_void_p(-1)  # Current-process pseudo handle, pointer-sized on Win64.
    if ctypes.windll.psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
        return counters.WorkingSetSize/1024**2
    return None
