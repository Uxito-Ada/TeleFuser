#!/usr/bin/env python3
"""
Multi-GPU communication and memory-to-GPU transfer speed test script
Test content includes:
1. Multi-GPU communication method detection
2. NCCL communication speed test
3. Memory to GPU transfer speed test
4. GPU-to-GPU data transfer speed test
"""

import time

import numpy as np
import torch
import torch.distributed as dist


class MultiGPUCommunicationTester:
    def __init__(self):
        self.device_count = torch.cuda.device_count()
        self.results = {}

    def detect_cuda_environment(self) -> dict:
        """Detect CUDA environment and multi-GPU configuration"""
        env_info = {
            "cuda_available": torch.cuda.is_available(),
            "device_count": self.device_count,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "torch_version": torch.__version__,
            "devices": [],
        }

        if env_info["cuda_available"]:
            for i in range(self.device_count):
                device_props = torch.cuda.get_device_properties(i)
                env_info["devices"].append(
                    {
                        "device_id": i,
                        "name": device_props.name,
                        "total_memory_gb": round(device_props.total_memory / (1024**3), 2),
                        "compute_capability": f"{device_props.major}.{device_props.minor}",
                        "multi_processor_count": device_props.multi_processor_count,
                    }
                )

        return env_info

    def detect_communication_backends(self) -> dict:
        """Detect available communication backends"""
        backends = {
            "nccl": dist.is_nccl_available(),
            "gloo": dist.is_gloo_available(),
            "mpi": dist.is_mpi_available(),
        }
        return backends

    def test_host_to_device_speed(self, data_size_mb: int = 100, num_runs: int = 10) -> dict:
        """Test host memory to GPU transfer speed"""
        results = {}

        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}

        # Prepare test data
        data_size_bytes = data_size_mb * 1024 * 1024
        host_data = torch.randn(data_size_bytes // 4, dtype=torch.float32)  # float32 occupies 4 bytes

        for device_id in range(self.device_count):
            device = torch.device(f"cuda:{device_id}")

            # 预热
            _ = host_data.to(device)
            torch.cuda.synchronize(device)

            # Formal test - run multiple times
            transfer_times = []
            speeds_mbps = []
            for _ in range(num_runs):
                start_time = time.time()
                device_data = host_data.to(device)
                torch.cuda.synchronize(device)
                end_time = time.time()
                transfer_time = end_time - start_time
                transfer_times.append(transfer_time)
                speeds_mbps.append(data_size_mb / transfer_time)

            # Calculate statistics
            transfer_times_array = np.array(transfer_times)
            speeds_array = np.array(speeds_mbps)
            avg_time = np.mean(transfer_times_array)
            avg_speed_mbps = np.mean(speeds_array)
            std_speed_mbps = np.std(speeds_array)

            results[f"device_{device_id}"] = {
                "transfer_time_avg_seconds": round(avg_time, 4),
                "speed_mbps_avg": round(avg_speed_mbps, 2),
                "speed_mbps_std": round(std_speed_mbps, 2),
                "num_runs": num_runs,
                "data_size_mb": data_size_mb,
            }

        return results

    def test_device_to_host_speed(self, data_size_mb: int = 100, num_runs: int = 10) -> dict:
        """Test GPU to host memory transfer speed"""
        results = {}

        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}

        # Prepare test data size
        data_size_bytes = data_size_mb * 1024 * 1024

        for device_id in range(self.device_count):
            device = torch.device(f"cuda:{device_id}")

            # Create data on GPU
            device_data = torch.randn(data_size_bytes // 4, dtype=torch.float32, device=device)

            # Warm up
            _ = device_data.cpu()
            torch.cuda.synchronize(device)

            # Formal test - run multiple times
            transfer_times = []
            speeds_mbps = []
            for _ in range(num_runs):
                start_time = time.time()
                host_data = device_data.cpu()
                torch.cuda.synchronize(device)
                end_time = time.time()
                transfer_time = end_time - start_time
                transfer_times.append(transfer_time)
                speeds_mbps.append(data_size_mb / transfer_time)

            # Calculate statistics
            transfer_times_array = np.array(transfer_times)
            speeds_array = np.array(speeds_mbps)
            avg_time = np.mean(transfer_times_array)
            avg_speed_mbps = np.mean(speeds_array)
            std_speed_mbps = np.std(speeds_array)

            results[f"device_{device_id}_to_host"] = {
                "transfer_time_avg_seconds": round(avg_time, 4),
                "speed_mbps_avg": round(avg_speed_mbps, 2),
                "speed_mbps_std": round(std_speed_mbps, 2),
                "num_runs": num_runs,
                "data_size_mb": data_size_mb,
            }

        return results

    def test_device_to_device_speed(self, data_size_mb: int = 100, num_runs: int = 10) -> dict:
        """Test data transfer speed between GPUs"""
        results = {}

        if self.device_count < 2:
            return {"error": "Need at least 2 GPUs for device-to-device test"}

        data_size_bytes = data_size_mb * 1024 * 1024

        for src_device in range(self.device_count):
            for dst_device in range(self.device_count):
                if src_device == dst_device:
                    continue

                src_dev = torch.device(f"cuda:{src_device}")
                dst_dev = torch.device(f"cuda:{dst_device}")

                # Create data on source device
                src_data = torch.randn(data_size_bytes // 4, dtype=torch.float32, device=src_dev)

                # Warm up
                _ = src_data.to(dst_dev)
                torch.cuda.synchronize(src_dev)
                torch.cuda.synchronize(dst_dev)

                # Formal test - run multiple times
                transfer_times = []
                speeds_mbps = []
                for _ in range(num_runs):
                    start_time = time.time()
                    dst_data = src_data.to(dst_dev)
                    torch.cuda.synchronize(src_dev)
                    torch.cuda.synchronize(dst_dev)
                    end_time = time.time()
                    transfer_time = end_time - start_time
                    transfer_times.append(transfer_time)
                    speeds_mbps.append(data_size_mb / transfer_time)

                # Calculate statistics
                transfer_times_array = np.array(transfer_times)
                speeds_array = np.array(speeds_mbps)
                avg_time = np.mean(transfer_times_array)
                avg_speed_mbps = np.mean(speeds_array)
                std_speed_mbps = np.std(speeds_array)

                key = f"device_{src_device}_to_device_{dst_device}"
                results[key] = {
                    "transfer_time_avg_seconds": round(avg_time, 4),
                    "speed_mbps_avg": round(avg_speed_mbps, 2),
                    "speed_mbps_std": round(std_speed_mbps, 2),
                    "num_runs": num_runs,
                    "data_size_mb": data_size_mb,
                }

        return results

    def test_nccl_collective_operations(self, data_size_mb: int = 50, num_runs: int = 10) -> dict:
        """Test NCCL collective operations speed"""
        results = {}

        if not dist.is_nccl_available() or self.device_count < 2:
            return {"error": "NCCL not available or insufficient GPUs"}

        # Initialize process group (single machine multiple GPUs)
        try:
            dist.init_process_group(
                backend="nccl",
                init_method="tcp://localhost:23456",
                rank=0,
                world_size=1,
            )
        except Exception as e:
            return {"error": f"Failed to initialize process group for {e}"}

        data_size_bytes = data_size_mb * 1024 * 1024
        tensor_size = data_size_bytes // 4  # float32

        # Test all_reduce operation
        for device_id in range(min(2, self.device_count)):  # Test first two devices
            device = torch.device(f"cuda:{device_id}")

            # Create test tensor
            tensor = torch.randn(tensor_size, dtype=torch.float32, device=device)

            # Warm up
            dist.all_reduce(tensor)
            torch.cuda.synchronize(device)

            # Formal test - run multiple times
            operation_times = []
            for _ in range(num_runs):
                # Recreate tensor each time to ensure consistency
                tensor = torch.randn(tensor_size, dtype=torch.float32, device=device)
                start_time = time.time()
                dist.all_reduce(tensor)
                torch.cuda.synchronize(device)
                end_time = time.time()
                operation_times.append(end_time - start_time)

            # Calculate statistics
            operation_times_array = np.array(operation_times)
            avg_time = np.mean(operation_times_array)
            std_time = np.std(operation_times_array)

            results[f"all_reduce_device_{device_id}"] = {
                "operation_time_avg_seconds": round(avg_time, 4),
                "operation_time_std_seconds": round(std_time, 4),
                "num_runs": num_runs,
                "data_size_mb": data_size_mb,
            }

        # Clean up process group
        dist.destroy_process_group()

        return results

    def run_comprehensive_test(self, data_sizes: list[int] = [10, 100, 500]) -> dict:
        """Run comprehensive performance test"""
        print("Starting multi-GPU communication performance test...")

        # 1. Environment detection
        print("\n1. Detecting CUDA environment and multi-GPU configuration...")
        env_info = self.detect_cuda_environment()
        print(f"CUDA available: {env_info['cuda_available']}")
        print(f"GPU count: {env_info['device_count']}")

        if env_info["cuda_available"]:
            for device in env_info["devices"]:
                print(f"  GPU {device['device_id']}: {device['name']} ({device['total_memory_gb']}GB)")

        # 2. Communication backend detection
        print("\n2. Detecting communication backends...")
        backends = self.detect_communication_backends()
        for backend, available in backends.items():
            print(f"  {backend.upper()}: {'Available' if available else 'Not available'}")

        self.results["environment"] = env_info
        self.results["backends"] = backends

        # 3. Transfer speed tests with different data sizes
        print("\n3. Running transfer speed tests...")

        for size in data_sizes:
            print(f"\nTesting data size: {size}MB")

            # Host to device transfer test
            print("  Host memory → GPU transfer test...")
            host_to_device = self.test_host_to_device_speed(size)
            self.results[f"host_to_device_{size}mb"] = host_to_device

            if "error" not in host_to_device:
                for device, result in host_to_device.items():
                    if "error" not in result:
                        print(f"    {device}: {result['speed_mbps_avg']} MB/s")

            # Device to host transfer test
            print("  GPU → Host memory transfer test...")
            device_to_host = self.test_device_to_host_speed(size)
            self.results[f"device_to_host_{size}mb"] = device_to_host

            if "error" not in device_to_host:
                for transfer, result in device_to_host.items():
                    if "error" not in result:
                        print(f"    {transfer}: {result['speed_mbps_avg']} MB/s")

            # Device-to-device transfer test
            if self.device_count >= 2:
                print("  GPU-to-GPU transfer test...")
                device_to_device = self.test_device_to_device_speed(size)
                self.results[f"device_to_device_{size}mb"] = device_to_device

                if "error" not in device_to_device:
                    for transfer, result in device_to_device.items():
                        if "error" not in result:
                            print(f"    {transfer}: {result['speed_mbps_avg']} MB/s")

            # NCCL collective operations test
            if backends["nccl"] and self.device_count >= 2:
                print("  NCCL collective operations test...")
                nccl_results = self.test_nccl_collective_operations(size)
                self.results[f"nccl_operations_{size}mb"] = nccl_results

                if "error" not in nccl_results:
                    for op, result in nccl_results.items():
                        if "error" not in result:
                            print(f"    {op}: {result['operation_time_avg_seconds']}s")

        return self.results

    def generate_report(self) -> str:
        """Generate test report"""
        report = []
        report.append("=" * 60)
        report.append("Multi-GPU Communication Performance Test Report")
        report.append("=" * 60)

        # Environment information
        env = self.results.get("environment", {})
        report.append("\nEnvironment Information:")
        report.append(f"  CUDA available: {env.get('cuda_available', False)}")
        report.append(f"  GPU count: {env.get('device_count', 0)}")

        if env.get("cuda_available"):
            for device in env.get("devices", []):
                report.append(f"  GPU {device['device_id']}: {device['name']} ({device['total_memory_gb']}GB)")

        # Communication backends
        backends = self.results.get("backends", {})
        report.append("\nCommunication Backends:")
        for backend, available in backends.items():
            report.append(f"  {backend.upper()}: {'Available' if available else 'Not available'}")

        # Performance results
        report.append("\nPerformance Test Results (average of 10 runs):")

        for key, result in self.results.items():
            if key in ["environment", "backends"]:
                continue

            if "host_to_device" in key:
                report.append(f"\n{key.replace('_', ' ').title()}:")
                for device, metrics in result.items():
                    if "error" not in metrics:
                        report.append(
                            f"  {device}: {metrics['speed_mbps_avg']} ± {metrics['speed_mbps_std']} MB/s (avg time: {metrics['transfer_time_avg_seconds']}s)"
                        )

            elif "device_to_host" in key:
                report.append(f"\n{key.replace('_', ' ').title()}:")
                for transfer, metrics in result.items():
                    if "error" not in metrics:
                        report.append(
                            f"  {transfer}: {metrics['speed_mbps_avg']} ± {metrics['speed_mbps_std']} MB/s (avg time: {metrics['transfer_time_avg_seconds']}s)"
                        )

            elif "device_to_device" in key:
                report.append(f"\n{key.replace('_', ' ').title()}:")
                for transfer, metrics in result.items():
                    if "error" not in metrics:
                        report.append(
                            f"  {transfer}: {metrics['speed_mbps_avg']} ± {metrics['speed_mbps_std']} MB/s (avg time: {metrics['transfer_time_avg_seconds']}s)"
                        )

            elif "nccl_operations" in key:
                report.append(f"\n{key.replace('_', ' ').title()}:")
                for op, metrics in result.items():
                    if "error" not in metrics:
                        report.append(
                            f"  {op}: avg time: {metrics['operation_time_avg_seconds']}s ± {metrics['operation_time_std_seconds']}s"
                        )

        return "\n".join(report)


def main():
    """Main function"""
    print("Multi-GPU Communication Performance Test Script")
    print("=" * 40)

    tester = MultiGPUCommunicationTester()

    # Run tests
    results = tester.run_comprehensive_test(data_sizes=[10, 100, 500])

    # Generate and output report
    report = tester.generate_report()
    print("\n" + report)

    # Save report to file
    with open("multi_gpu_test_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print("\nTest report saved to: multi_gpu_test_report.txt")


if __name__ == "__main__":
    main()
