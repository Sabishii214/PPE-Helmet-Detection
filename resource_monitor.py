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
        self.gpu_watts = []
        self.epoch_times = []
        self.start_time = None
        self.end_time = None
        self.epoch_start_time = None
        self.peak_memory = 0
        self.gpu_name = "Unknown"
        self.num_gpus = 0

        if torch.cuda.is_available():
            self.num_gpus = torch.cuda.device_count()
            self.gpu_name = torch.cuda.get_device_name(0)

    def start_epoch(self):
        """Mark the start of an epoch"""
        self.epoch_start_time = time.time()

    def end_epoch(self):
        """Mark the end of an epoch and record duration"""
        if self.epoch_start_time:
            duration = time.time() - self.epoch_start_time
            self.epoch_times.append(duration)
            self.epoch_start_time = None

    def _monitor(self):
        while not self.stop_event.is_set():
            try:
                # Get GPU stats via nvidia-smi
                # query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits
                result = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,power.draw', '--format=csv,noheader,nounits'],
                    encoding='utf-8'
                )
                lines = result.strip().split('\n')
                
                batch_util = []
                batch_mem = []
                batch_watts = []
                
                for line in lines:
                    try:
                        util, mem, power = map(float, line.split(','))
                        batch_util.append(util)
                        batch_mem.append(mem)
                        batch_watts.append(power)
                    except ValueError:
                        continue

                # Store averages across all GPUs for this tick
                if batch_util:
                    self.gpu_utilization.append(statistics.mean(batch_util))
                if batch_mem:
                    current_total_mem = sum(batch_mem)
                    self.memory_usage.append(current_total_mem)
                    self.peak_memory = max(self.peak_memory, current_total_mem)
                if batch_watts:
                    self.gpu_watts.append(statistics.mean(batch_watts))

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
        
        avg_watts = statistics.mean(self.gpu_watts) if self.gpu_watts else 0
        min_watts = min(self.gpu_watts) if self.gpu_watts else 0
        max_watts = max(self.gpu_watts) if self.gpu_watts else 0
        
        # Epoch stats
        if self.epoch_times:
            avg_epoch = statistics.mean(self.epoch_times)
            min_epoch = min(self.epoch_times)
            max_epoch = max(self.epoch_times)
            epoch_count = len(self.epoch_times)
        else:
            avg_epoch = min_epoch = max_epoch = 0
            epoch_count = 0
        
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
        report.append("GPU POWER USAGE (Watts):")
        report.append(f"  Average:              {avg_watts:.1f} W")
        report.append(f"  Min:                  {min_watts:.1f} W")
        report.append(f"  Max:                  {max_watts:.1f} W")

        if epoch_count > 0:
            report.append("-" * 60)
            report.append(f"EPOCH TIMING ({epoch_count} epochs recorded):")
            report.append(f"  Average Time:         {avg_epoch:.2f} s")
            report.append(f"  Min Time:             {min_epoch:.2f} s")
            report.append(f"  Max Time:             {max_epoch:.2f} s")
            
        report.append("-" * 60)
        report.append("COST ESTIMATION:")
        report.append(f"  Total GPU Hours:      {gpu_hours:.4f} hours")
        report.append("=" * 60)
        
        return "\n".join(report)
