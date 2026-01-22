"""
导出 timeline 数据为 Perfetto Chrome Trace Event Format
使用 ui.perfetto.dev 查看
"""

import json
import torch
from typing import Dict, List, Tuple


def detect_clock_rate_mhz(gpu_index: int = 0) -> float:
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        mhz = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
        if mhz and mhz > 0:
            return float(mhz)
        mhz = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
        if mhz and mhz > 0:
            return float(mhz)
    except Exception:
        pass

    # Fallback: nvidia-smi
    import subprocess
    for field in ["clocks.current.sm", "clocks.current.graphics"]:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=" + field, "--format=csv,noheader,nounits"]
            ).decode().strip().splitlines()
            if out:
                val = out[gpu_index].strip()
                mhz = float(val)
                if mhz > 0:
                    return mhz
        except Exception:
            continue

    raise RuntimeError("Cannot detect MHz。")




# Event ID definitions (from util.cuh)
TEVENT_CONTROLLER_START = 0
TEVENT_IFETCH_DONE = 1
TEVENT_PAGE_ALLOC_DONE = 2
TEVENT_SEMS_SETUP = 3
TEVENT_CONTROLLER_END = 4
TEVENT_LOADER_START = 5
TEVENT_IFETCH_DONE_LOADER = 6
TEVENT_LAUNCHER_START = 7
TEVENT_LAUNCHER_END = 8
TEVENT_STORER_START = 9
TEVENT_STORER_END = 10
TEVENT_CONSUMER_START = 11

# Event ranges corresponding to each worker
WORKER_EVENTS = {
    "controller": (TEVENT_CONTROLLER_START, TEVENT_CONTROLLER_END),
    "loader": (TEVENT_LOADER_START, TEVENT_IFETCH_DONE_LOADER),
    "launcher": (TEVENT_LAUNCHER_START, TEVENT_LAUNCHER_END),
    "storer": (TEVENT_STORER_START, TEVENT_STORER_END),
    "consumer": (TEVENT_CONSUMER_START, TEVENT_CONSUMER_START + 1),  # consumer 每个 warp 2个槽
}

# Opcode to operation name mapping
OPCODE_NAMES = {
    1: "RMS_QKV_MatVecRopeAppend",
    2: "PartialAttention",
    3: "AttentionReduction",
    4: "O_ProjResidual",
    5: "RMS_DoubleMatVecSiLU",
    6: "DownProjResidual",
    7: "RMS_LM_Head"
}



def export_to_perfetto(
    schedule,
    worker_name: str = "consumer",
    output_file: str = "timeline.json",
    clock_rate_mhz: float = 1800.0,
) -> str:
    """
    Export timeline data to Perfetto Chrome Trace Event Format
    
    Args:
        schedule: Schedule object containing globs (with timings and instructions)
        worker_name: Name of the worker to export ("consumer", "loader", "launcher", "storer", "controller")
        output_file: Output file name
        clock_rate_mhz: GPU clock frequency (MHz)
    
    Returns:
        Output file path
    """
    
    timings = schedule.globs.timings.cpu()  # [num_sms, max_queue_len, 128]
    instructions = schedule.globs.instructions.cpu()  # [num_sms, max_queue_len, 32]
    
    if worker_name not in WORKER_EVENTS:
        raise ValueError(f"Unknown worker: {worker_name}. Available: {list(WORKER_EVENTS.keys())}")
    
    start_event_id, end_event_id = WORKER_EVENTS[worker_name]
    
    events = []
    
    # Iterate over all SMs and instructions
    for sm_id in range(timings.shape[0]):
        for instr_idx in range(timings.shape[1]):
            start_cycles = timings[sm_id, instr_idx, start_event_id].item()
            end_cycles = timings[sm_id, instr_idx, end_event_id].item()
            
            if start_cycles <= 0 or end_cycles <= 0:
                continue
            
            # convert to microseconds
            start_us = start_cycles / clock_rate_mhz
            end_us = end_cycles / clock_rate_mhz
            duration_us = end_us - start_us
            
            # Skip events with zero or negative duration
            if duration_us <= 0:
                continue
            
            # get opcode and operation name
            opcode = instructions[sm_id, instr_idx, 0].item()
            op_name = OPCODE_NAMES.get(opcode, f"Op{opcode}")
            
            # create event name
            event_name = f"{op_name}:{worker_name}"
            
            # create Perfetto event (Complete event)
            event = {
                "name": event_name,
                "ph": "X",  # Complete event
                "ts": start_us,  # timestamp in microseconds
                "dur": duration_us,  # duration in microseconds
                "pid": sm_id,  # SM ID as process
                "tid": worker_name,  # Worker name as thread
                "args": {
                    "sm_id": sm_id,
                    "instruction_index": instr_idx,
                    "opcode": opcode,
                    "operation": op_name,
                    "start_cycles": start_cycles,
                    "end_cycles": end_cycles,
                    "start_us": start_us,
                    "end_us": end_us,
                },
            }
            
            events.append(event)
    
    # Write JSON file
    with open(output_file, 'w') as f:
        json.dump(events, f, indent=2)
    
    print(f"✓ Exported {len(events)} events to {output_file}")
    
    return output_file


def export_all_workers(
    schedule,
    output_dir: str = ".",
    clock_rate_mhz: float = 1800.0,
) -> Dict[str, str]:
    """
    Export timelines for all workers
    
    Args:
        schedule: Schedule object
        output_dir: Output directory
        clock_rate_mhz: GPU clock rate in MHz
    
    Returns:
        {worker_name: output_file} dictionary
    """
    
    results = {}
    
    for worker_name in WORKER_EVENTS.keys():
        try:
            output_file = f"{output_dir}/timeline_{worker_name}.json"
            export_to_perfetto(schedule, worker_name, output_file, clock_rate_mhz)
            results[worker_name] = output_file
        except Exception as e:
            print(f"✗ Failed to export {worker_name}: {e}")
    
    print(f"\nExported {len(results)} workers")
    return results


def merge_traces(*trace_files: str, output_file: str = "merged_trace.json") -> str:
    """
    Merge multiple trace files
    
    Args:
        *trace_files: JSON files to merge
        output_file: Output file name
    
    Returns:
        Output file path
    """
    
    all_events = []
    
    for trace_file in trace_files:
        with open(trace_file, 'r') as f:
            events = json.load(f)
            all_events.extend(events)
    
    with open(output_file, 'w') as f:
        json.dump(all_events, f, indent=2)
    
    print(f"✓ Merged {len(trace_files)} traces into {output_file}")
    return output_file


# Helper function: List available workers
def list_available_workers() -> List[str]:
    """List all available workers"""
    return list(WORKER_EVENTS.keys())


if __name__ == "__main__":
    print("Perfetto exporter ready!")
    print(f"Available workers: {list_available_workers()}")
