"""
Real NVIDIA DGX Spark Integration.

Provides actual GPU metrics using pynvml or nvidia-smi fallback.
Falls back to simulation mode when hardware is unavailable.
"""

import asyncio
import subprocess
import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Try to import pynvml
try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False
    pynvml = None
    logger.info("pynvml not available, will try nvidia-smi or simulation mode")


@dataclass
class GPUMetrics:
    """Container for GPU metrics."""
    index: int
    name: str
    uuid: str
    
    # Utilization
    gpu_utilization: float  # Percentage 0-100
    memory_utilization: float  # Percentage 0-100
    
    # Memory
    memory_total: int  # MB
    memory_used: int  # MB
    memory_free: int  # MB
    
    # Temperature
    temperature: int  # Celsius
    
    # Power
    power_draw: float  # Watts
    power_limit: float  # Watts
    
    # Clock speeds
    sm_clock: int  # MHz
    memory_clock: int  # MHz
    
    # Bandwidth (estimated)
    memory_bandwidth: float  # GB/s
    
    # Compute
    compute_mode: str
    
    # Processes
    processes: List[Dict[str, Any]] = field(default_factory=list)
    
    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def memory_used_percent(self) -> float:
        """Memory usage as percentage."""
        if self.memory_total == 0:
            return 0.0
        return (self.memory_used / self.memory_total) * 100


@dataclass
class SystemMetrics:
    """Container for system-wide GPU metrics."""
    driver_version: str
    cuda_version: str
    gpu_count: int
    gpus: List[GPUMetrics] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    simulation_mode: bool = False
    
    @property
    def total_memory(self) -> int:
        """Total GPU memory across all GPUs (MB)."""
        return sum(gpu.memory_total for gpu in self.gpus)
    
    @property
    def total_memory_used(self) -> int:
        """Total used GPU memory across all GPUs (MB)."""
        return sum(gpu.memory_used for gpu in self.gpus)
    
    @property
    def average_utilization(self) -> float:
        """Average GPU utilization across all GPUs."""
        if not self.gpus:
            return 0.0
        return sum(gpu.gpu_utilization for gpu in self.gpus) / len(self.gpus)
    
    @property
    def average_temperature(self) -> float:
        """Average temperature across all GPUs."""
        if not self.gpus:
            return 0.0
        return sum(gpu.temperature for gpu in self.gpus) / len(self.gpus)


class DGXSparkAPI:
    """
    NVIDIA DGX Spark API for real GPU metrics.
    
    Supports:
    - pynvml (preferred, direct NVML access)
    - nvidia-smi (fallback, subprocess)
    - Simulation mode (when no GPU available)
    
    Usage:
        api = DGXSparkAPI()
        metrics = await api.get_metrics()
        
        for gpu in metrics.gpus:
            print(f"GPU {gpu.index}: {gpu.gpu_utilization}% util")
    """
    
    _instance: Optional['DGXSparkAPI'] = None
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self._nvml_initialized = False
        self._simulation_mode = False
        self._last_metrics: Optional[SystemMetrics] = None
        self._metrics_history: List[SystemMetrics] = []
        self._history_limit = 60  # Keep 60 samples (1 minute at 1Hz)
        
        # Simulation state
        self._sim_base_util = 45.0
        self._sim_last_time = time.time()
        
        # Try to initialize NVML
        self._init_nvml()
    
    def _init_nvml(self) -> bool:
        """Initialize NVML library."""
        if not HAS_PYNVML or pynvml is None:
            logger.info("pynvml not available, using nvidia-smi or simulation mode")
            return False

        try:
            pynvml.nvmlInit()
            self._nvml_initialized = True
            logger.info("NVML initialized successfully")
            return True
        except Exception as e:
            logger.warning(f"Failed to initialize NVML: {e}. Falling back to nvidia-smi or simulation.")
            return False
    
    def _shutdown_nvml(self):
        """Shutdown NVML library."""
        if self._nvml_initialized and pynvml is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_initialized = False
    
    def __del__(self):
        """Cleanup on destruction."""
        self._shutdown_nvml()
    
    @property
    def is_simulation_mode(self) -> bool:
        """Check if running in simulation mode."""
        return self._simulation_mode
    
    def set_simulation_mode(self, enabled: bool):
        """Enable or disable simulation mode."""
        self._simulation_mode = enabled
    
    async def get_metrics(self, force_refresh: bool = False) -> SystemMetrics:
        """
        Get current GPU metrics.
        
        Args:
            force_refresh: Force refresh even if recent metrics exist
            
        Returns:
            SystemMetrics object with GPU data
        """
        # Try pynvml first
        if self._nvml_initialized and not self._simulation_mode:
            try:
                metrics = self._get_metrics_pynvml()
                self._add_to_history(metrics)
                return metrics
            except Exception as e:
                logger.warning(f"pynvml metrics failed: {e}, trying nvidia-smi")
        
        # Try nvidia-smi
        if not self._simulation_mode:
            try:
                metrics = await self._get_metrics_nvidia_smi()
                if metrics:
                    self._add_to_history(metrics)
                    return metrics
            except Exception as e:
                logger.warning(f"nvidia-smi failed: {e}, falling back to simulation")
        
        # Fall back to simulation
        metrics = self._get_metrics_simulation()
        self._add_to_history(metrics)
        return metrics
    
    def _get_metrics_pynvml(self) -> SystemMetrics:
        """Get metrics using pynvml."""
        if pynvml is None:
            raise RuntimeError("pynvml not available")

        driver_version = pynvml.nvmlSystemGetDriverVersion()
        cuda_version = "N/A"
        try:
            cuda_version = pynvml.nvmlSystemGetCudaDriverVersion_v2()
            cuda_version = f"{cuda_version // 1000}.{(cuda_version % 1000) // 10}"
        except Exception:
            pass
        
        device_count = pynvml.nvmlDeviceGetCount()
        gpus = []
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            
            # Basic info
            name = pynvml.nvmlDeviceGetName(handle)
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            
            # Utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            # Memory
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            # Temperature
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            # Power
            try:
                power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
            except Exception:
                power_draw = 0.0
            
            try:
                power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
            except Exception:
                power_limit = 0.0
            
            # Clocks
            try:
                sm_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
            except Exception:
                sm_clock = 0
            
            try:
                mem_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            except Exception:
                mem_clock = 0
            
            # Compute mode
            try:
                compute_mode = pynvml.nvmlDeviceGetComputeMode(handle)
                compute_modes = {0: "Default", 1: "Exclusive Thread", 2: "Prohibited", 3: "Exclusive Process"}
                compute_mode_str = compute_modes.get(compute_mode, "Unknown")
            except Exception:
                compute_mode_str = "Unknown"
            
            # Processes
            processes = []
            try:
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                for proc in procs:
                    processes.append({
                        "pid": proc.pid,
                        "memory": proc.usedGpuMemory // (1024 * 1024),  # MB
                    })
            except Exception:
                pass
            
            # Estimate memory bandwidth (based on memory clock and bus width)
            # This is a rough estimate - actual bandwidth depends on memory type
            memory_bandwidth = mem_clock * 0.256  # Rough estimate in GB/s
            
            gpu = GPUMetrics(
                index=i,
                name=name,
                uuid=uuid,
                gpu_utilization=util.gpu,
                memory_utilization=util.memory,
                memory_total=mem_info.total // (1024 * 1024),
                memory_used=mem_info.used // (1024 * 1024),
                memory_free=mem_info.free // (1024 * 1024),
                temperature=temp,
                power_draw=power_draw,
                power_limit=power_limit,
                sm_clock=sm_clock,
                memory_clock=mem_clock,
                memory_bandwidth=memory_bandwidth,
                compute_mode=compute_mode_str,
                processes=processes,
            )
            gpus.append(gpu)
        
        return SystemMetrics(
            driver_version=driver_version,
            cuda_version=cuda_version,
            gpu_count=device_count,
            gpus=gpus,
            simulation_mode=False,
        )
    
    async def _get_metrics_nvidia_smi(self) -> Optional[SystemMetrics]:
        """Get metrics using nvidia-smi subprocess."""
        try:
            # Run nvidia-smi with JSON output
            cmd = [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,utilization.gpu,utilization.memory,"
                "memory.total,memory.used,memory.free,temperature.gpu,"
                "power.draw,power.limit,clocks.sm,clocks.mem,compute_mode",
                "--format=csv,noheader,nounits"
            ]
            
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("nvidia-smi GPU metrics query timed out after 5 seconds")
                proc.kill()
                await proc.wait()
                return None
            
            if proc.returncode != 0:
                return None
            
            # Parse output
            lines = stdout.decode().strip().split('\n')
            gpus = []
            
            for line in lines:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) < 14:
                    continue
                
                gpu = GPUMetrics(
                    index=int(parts[0]),
                    name=parts[1],
                    uuid=parts[2],
                    gpu_utilization=float(parts[3]) if parts[3] != '[N/A]' else 0.0,
                    memory_utilization=float(parts[4]) if parts[4] != '[N/A]' else 0.0,
                    memory_total=int(parts[5]) if parts[5] != '[N/A]' else 0,
                    memory_used=int(parts[6]) if parts[6] != '[N/A]' else 0,
                    memory_free=int(parts[7]) if parts[7] != '[N/A]' else 0,
                    temperature=int(parts[8]) if parts[8] != '[N/A]' else 0,
                    power_draw=float(parts[9]) if parts[9] != '[N/A]' else 0.0,
                    power_limit=float(parts[10]) if parts[10] != '[N/A]' else 0.0,
                    sm_clock=int(parts[11]) if parts[11] != '[N/A]' else 0,
                    memory_clock=int(parts[12]) if parts[12] != '[N/A]' else 0,
                    memory_bandwidth=float(parts[12]) * 0.256 if parts[12] != '[N/A]' else 0.0,
                    compute_mode=parts[13],
                    processes=[],
                )
                gpus.append(gpu)
            
            # Get driver version
            driver_cmd = ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
            driver_proc = await asyncio.create_subprocess_exec(
                *driver_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            try:
                driver_stdout, _ = await asyncio.wait_for(driver_proc.communicate(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("nvidia-smi driver version query timed out after 5 seconds")
                driver_proc.kill()
                await driver_proc.wait()
                return None
            driver_version = driver_stdout.decode().strip().split('\n')[0] if driver_proc.returncode == 0 else "Unknown"
            
            return SystemMetrics(
                driver_version=driver_version,
                cuda_version="N/A",
                gpu_count=len(gpus),
                gpus=gpus,
                simulation_mode=False,
            )
            
        except FileNotFoundError:
            logger.warning("nvidia-smi not found")
            return None
        except Exception as e:
            logger.error(f"nvidia-smi error: {e}")
            return None
    
    def _get_metrics_simulation(self) -> SystemMetrics:
        """Generate simulated GPU metrics."""
        current_time = time.time()
        dt = current_time - self._sim_last_time
        self._sim_last_time = current_time
        
        # Simulate realistic GPU usage patterns
        # Base utilization varies over time with some noise
        self._sim_base_util += random.uniform(-2, 2)
        self._sim_base_util = max(30, min(85, self._sim_base_util))
        
        # Add periodic spikes to simulate workload
        spike = 15 * abs(((current_time % 30) / 30 - 0.5) * 2)
        
        gpus = []
        for i in range(1):  # Simulate 1 GPU (DGX Spark)
            noise = random.uniform(-5, 5)
            gpu_util = max(0, min(100, self._sim_base_util + spike + noise))
            
            # Memory correlates somewhat with utilization
            mem_total = 128 * 1024  # 128 GB (H200 spec)
            mem_base = int(mem_total * 0.3)  # Base memory usage
            mem_workload = int(mem_total * (gpu_util / 100) * 0.4)
            mem_used = mem_base + mem_workload + random.randint(-1024, 1024)
            mem_used = max(0, min(mem_total, mem_used))
            
            # Temperature correlates with utilization
            temp_base = 35
            temp_load = int(gpu_util * 0.4)
            temp = temp_base + temp_load + random.randint(-2, 2)
            
            # Power correlates with utilization
            power_idle = 50
            power_max = 700  # H200 TDP
            power_draw = power_idle + (power_max - power_idle) * (gpu_util / 100) + random.uniform(-10, 10)
            
            gpu = GPUMetrics(
                index=i,
                name="NVIDIA H200 NVL (Simulated)",
                uuid=f"GPU-SIM-{i:04d}-0000-0000-000000000000",
                gpu_utilization=gpu_util,
                memory_utilization=(mem_used / mem_total) * 100,
                memory_total=mem_total,
                memory_used=mem_used,
                memory_free=mem_total - mem_used,
                temperature=temp,
                power_draw=power_draw,
                power_limit=700.0,
                sm_clock=1800 + random.randint(-50, 50),
                memory_clock=2619,  # LPDDR5x spec
                memory_bandwidth=273.0 + random.uniform(-5, 5),  # GB/s
                compute_mode="Default",
                processes=[
                    {"pid": 1234, "memory": random.randint(1000, 5000)},
                    {"pid": 5678, "memory": random.randint(500, 2000)},
                ] if gpu_util > 50 else [],
            )
            gpus.append(gpu)
        
        return SystemMetrics(
            driver_version="550.54.15 (Simulated)",
            cuda_version="12.4",
            gpu_count=1,
            gpus=gpus,
            simulation_mode=True,
        )
    
    def _add_to_history(self, metrics: SystemMetrics):
        """Add metrics to history."""
        self._last_metrics = metrics
        self._metrics_history.append(metrics)
        if len(self._metrics_history) > self._history_limit:
            self._metrics_history.pop(0)
    
    def get_history(self, limit: int = 60) -> List[SystemMetrics]:
        """Get metrics history."""
        return self._metrics_history[-limit:]
    
    def get_utilization_history(self, gpu_index: int = 0, limit: int = 60) -> List[float]:
        """Get GPU utilization history for sparkline display."""
        history = self.get_history(limit)
        result = []
        for metrics in history:
            if gpu_index < len(metrics.gpus):
                result.append(metrics.gpus[gpu_index].gpu_utilization)
        return result
    
    def get_memory_history(self, gpu_index: int = 0, limit: int = 60) -> List[float]:
        """Get memory usage history for sparkline display."""
        history = self.get_history(limit)
        result = []
        for metrics in history:
            if gpu_index < len(metrics.gpus):
                result.append(metrics.gpus[gpu_index].memory_used_percent)
        return result
    
    def get_temperature_history(self, gpu_index: int = 0, limit: int = 60) -> List[int]:
        """Get temperature history."""
        history = self.get_history(limit)
        result = []
        for metrics in history:
            if gpu_index < len(metrics.gpus):
                result.append(metrics.gpus[gpu_index].temperature)
        return result


# Global singleton accessor
def get_dgx_api() -> DGXSparkAPI:
    """Get the global DGXSparkAPI instance."""
    return DGXSparkAPI()

