import time
import threading
import subprocess
import statistics
from datetime import timedelta
import torch

class ResourceMonitor:
    def __init__(self, interval=1.0):
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = None
        self.gpu_utilization = []
        self.memory_usage = []
        self.start_time = None
        self.end_time = None
        self.peak_memory = 0
        self.gpu_name = "Unknown"
        self.num_gpus = 0

        if torch.cuda.is_available():
            self.num_gpus = torch.cuda.device_count()
            self.gpu_name = torch.cuda.get_device_name(0)

    def _monitor(self):
        while not self.stop_event.is_set():
            try:
                # Get GPU stats via nvidia-smi
                # query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits
                result = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                    encoding='utf-8'
                )
                lines = result.strip().split('\n')
                
                batch_util = []
                batch_mem = []
                
                for line in lines:
                    util, mem = map(int, line.split(','))
                    batch_util.append(util)
                    batch_mem.append(mem)

                # Store averages across all GPUs for this tick
                if batch_util:
                    self.gpu_utilization.append(statistics.mean(batch_util))
                if batch_mem:
                    current_total_mem = sum(batch_mem)
                    self.memory_usage.append(current_total_mem)
                    self.peak_memory = max(self.peak_memory, current_total_mem)

            except Exception:
                # If nvidia-smi fails, we just skip this tick
                pass
            
            time.sleep(self.interval)

    def start(self):
        if self.num_gpus == 0:
            print("No GPUs detected. Resource monitoring disabled.")
            return

        self.start_time = time.time()
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def stop(self):
        if self.thread and self.thread.is_alive():
            self.stop_event.set()
            self.thread.join()
        self.end_time = time.time()

    def generate_report(self):
        if not self.start_time:
            return "Monitoring not started."
            
        end = self.end_time if self.end_time else time.time()
        duration_seconds = end - self.start_time
        duration_str = str(timedelta(seconds=int(duration_seconds)))
        
        # Calculate stats
        avg_util = statistics.mean(self.gpu_utilization) if self.gpu_utilization else 0
        peak_util = max(self.gpu_utilization) if self.gpu_utilization else 0
        
        # Estimate power/cost (simple approximation: hours * num_gpus)
        gpu_hours = (duration_seconds / 3600.0) * self.num_gpus
        
        report = []
        report.append("=" * 60)
        report.append("       TRAINING RESOURCE USAGE REPORT")
        report.append("=" * 60)
        report.append(f"Total Training Time:    {duration_str}")
        report.append(f"Device:                 {self.gpu_name} (x{self.num_gpus})")
        report.append("-" * 60)
        report.append("GPU MEMORY:")
        report.append(f"  Peak Usage:           {self.peak_memory} MB")
        if self.memory_usage:
            report.append(f"  Average Usage:        {int(statistics.mean(self.memory_usage))} MB")
        
        report.append("-" * 60)
        report.append("GPU UTILIZATION:")
        report.append(f"  Average:              {avg_util:.1f} %")
        report.append(f"  Peak:                 {peak_util} %")
        report.append("-" * 60)
        report.append("COST ESTIMATION:")
        report.append(f"  Total GPU Hours:      {gpu_hours:.4f} hours")
        report.append("=" * 60)
        
        return "\n".join(report)
