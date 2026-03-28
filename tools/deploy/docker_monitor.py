#!/usr/bin/env python3
"""
Docker容器监控脚本 - Python版本 (兼容Python 3.6+)
使用绝对内存单位(MB)进行监控
支持完整的命令行参数配置
"""

import argparse
import csv
import logging
import os
import statistics
import subprocess
import sys
import time
from collections import deque
from datetime import datetime


class DockerContainerMonitor:
    """Docker容器监控类"""

    def __init__(
        self,
        container_name: str,
        check_interval: int = 30,
        max_checks: int = 60,
        memory_threshold: float = 1024.0,  # 改为MB单位，默认1GB
        memory_leak_threshold: float = 100.0,  # 改为MB单位，默认100MB
        sliding_window_size: int = 10,
        log_file: str = "container_monitor.log",
        csv_file: str = "container_usage.csv",
    ):
        """初始化监控器

        Args:
            container_name: 容器名称或ID
            check_interval: 检查间隔(秒)
            max_checks: 最大检查次数
            memory_threshold: 内存阈值(MB)
            memory_leak_threshold: 内存泄漏检测阈值(MB)
            sliding_window_size: 滑动窗口大小
            log_file: 日志文件路径
            csv_file: CSV数据文件路径
        """
        self.container_name = container_name
        self.check_interval = check_interval
        self.max_checks = max_checks
        self.memory_threshold = memory_threshold
        self.memory_leak_threshold = memory_leak_threshold
        self.sliding_window_size = sliding_window_size
        self.log_file = log_file
        self.csv_file = csv_file

        # 初始化数据结构
        self.memory_history = []  # 存储内存使用量(MB)
        self.sliding_window = deque(maxlen=sliding_window_size)

        # 颜色代码
        self.colors = {
            "RED": "\033[0;31m",
            "GREEN": "\033[0;32m",
            "YELLOW": "\033[1;33m",
            "BLUE": "\033[0;34m",
            "PURPLE": "\033[0;35m",
            "CYAN": "\033[0;36m",
            "NC": "\033[0m",
        }

        # 初始化日志和文件
        self._init_logging()
        self._init_files()

    def _init_logging(self):
        """初始化日志配置"""
        # 确保日志目录存在
        log_dir = os.path.dirname(self.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def _init_files(self):
        """初始化输出文件"""
        # 确保CSV文件目录存在
        csv_dir = os.path.dirname(self.csv_file)
        if csv_dir and not os.path.exists(csv_dir):
            os.makedirs(csv_dir, exist_ok=True)

        # 初始化日志文件
        with open(self.log_file, "w") as f:
            f.write("=== Docker容器监控日志 ===\n")
            f.write(f"容器: {self.container_name}\n")
            f.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"检查间隔: {self.check_interval}秒\n")
            f.write(f"最大检查次数: {self.max_checks}\n")
            f.write(f"内存阈值: {self.memory_threshold}MB\n")
            f.write(f"内存泄漏阈值: {self.memory_leak_threshold}MB\n")
            f.write(f"滑动窗口大小: {self.sliding_window_size}\n")
            f.write("=================================\n")

        # 初始化CSV文件
        with open(self.csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "memory_usage_mb",
                    "sliding_avg_memory_mb",
                    "memory_usage",
                    "memory_limit",
                    "cpu_percent",
                    "gpu_memory_usage",
                    "gpu_memory_total",
                    "gpu_utilization",
                    "memory_leak_status",
                    "status",
                ]
            )

    def log_message(self, message: str, level: str = "INFO"):
        """记录日志消息

        Args:
            message: 日志消息
            level: 日志级别
        """
        color_map = {
            "ERROR": self.colors["RED"],
            "WARN": self.colors["YELLOW"],
            "INFO": self.colors["GREEN"],
            "DEBUG": self.colors["BLUE"],
        }

        color = color_map.get(level, self.colors["NC"])
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        log_entry = f"[{level}] {timestamp} - {message}"

        # 写入文件
        with open(self.log_file, "a") as f:
            f.write(log_entry + "\n")

        # 输出到控制台（带颜色）
        print(f"{color}[{level}]{self.colors['NC']} {timestamp} - {message}")

    def log_csv_data(
        self,
        timestamp: str,
        mem_usage_mb: float,
        sliding_avg_mb: float,
        mem_usage: str,
        mem_limit: str,
        cpu_percent: float,
        gpu_mem_usage: str,
        gpu_mem_total: str,
        gpu_util: str,
        leak_status: str,
        status: str,
    ):
        """记录CSV数据

        Args:
            timestamp: 时间戳
            mem_usage_mb: 内存使用量(MB)
            sliding_avg_mb: 滑动平均内存(MB)
            mem_usage: 内存使用量字符串
            mem_limit: 内存限制字符串
            cpu_percent: CPU使用百分比
            gpu_mem_usage: GPU显存使用量
            gpu_mem_total: GPU显存总量
            gpu_util: GPU利用率
            leak_status: 内存泄漏状态
            status: 状态
        """
        with open(self.csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    timestamp,
                    mem_usage_mb,
                    sliding_avg_mb,
                    mem_usage,
                    mem_limit,
                    cpu_percent,
                    gpu_mem_usage,
                    gpu_mem_total,
                    gpu_util,
                    leak_status,
                    status,
                ]
            )

    def convert_to_mb(self, memory_str: str) -> float:
        """将内存字符串转换为MB单位

        Args:
            memory_str: 内存字符串 (如 "100MiB", "1.5GiB", "512KiB")

        Returns:
            内存大小(MB)
        """
        if memory_str == "N/A":
            return 0.0

        try:
            # 移除空格并转换为小写
            memory_str = memory_str.strip().lower()

            # 分离数字和单位
            num_str = ""
            unit = ""
            for char in memory_str:
                if char.isdigit() or char == ".":
                    num_str += char
                else:
                    unit = memory_str[len(num_str) :]
                    break

            if not num_str:
                return 0.0

            value = float(num_str)

            # 根据单位转换
            if "kib" in unit or "kb" in unit:
                return value / 1024.0
            elif "mib" in unit or "mb" in unit:
                return value
            elif "gib" in unit or "gb" in unit:
                return value * 1024.0
            elif "tib" in unit or "tb" in unit:
                return value * 1024.0 * 1024.0
            elif "b" in unit:  # 字节
                return value / (1024.0 * 1024.0)
            else:
                # 如果没有明确单位，假设是字节
                return value / (1024.0 * 1024.0)

        except (ValueError, AttributeError, IndexError) as e:
            self.log_message(f"转换内存字符串失败: {memory_str}, 错误: {e}", "DEBUG")
            return 0.0

    def calculate_sliding_average(self, current_memory_mb: float) -> float:
        """计算滑动窗口平均值(MB)

        Args:
            current_memory_mb: 当前内存使用量(MB)

        Returns:
            滑动平均值(MB)
        """
        # 添加到滑动窗口
        self.sliding_window.append(current_memory_mb)

        # 计算平均值
        if len(self.sliding_window) == 0:
            return 0.0

        return statistics.mean(self.sliding_window)

    def detect_memory_leak(self, current_memory_mb: float, sliding_avg_mb: float) -> str:
        """检测内存泄漏

        Args:
            current_memory_mb: 当前内存使用量(MB)
            sliding_avg_mb: 滑动平均值(MB)

        Returns:
            泄漏状态字符串
        """
        # 如果数据不足，返回正常
        if len(self.sliding_window) < self.sliding_window_size:
            return "正常"

        # 计算当前内存与滑动平均的差异(MB)
        diff = current_memory_mb - sliding_avg_mb

        # 如果当前内存比滑动平均高出阈值，可能发生内存泄漏
        if diff > self.memory_leak_threshold:
            return f"可能泄漏(+{diff:.1f}MB)"
        else:
            return "正常"

    def check_nvidia_gpu(self) -> str:
        """检查NVIDIA GPU可用性

        Returns:
            GPU访问类型: 'container', 'host', 或 'none'
        """
        try:
            # 检查nvidia-smi是否可用
            result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                return "none"

            # 检查容器内是否有GPU访问权限
            result = subprocess.run(
                [
                    "docker",
                    "exec",
                    self.container_name,
                    "sh",
                    "-c",
                    "command -v nvidia-smi",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            if result.returncode == 0:
                return "container"
            else:
                return "host"

        except (subprocess.CalledProcessError, FileNotFoundError):
            return "none"

    def get_gpu_usage(self, gpu_access: str) -> tuple[str, str, str]:
        """获取GPU使用情况

        Args:
            gpu_access: GPU访问类型

        Returns:
            (gpu_memory_usage, gpu_memory_total, gpu_utilization)
        """
        gpu_memory_usage = "N/A"
        gpu_memory_total = "N/A"
        gpu_utilization = "N/A"

        if gpu_access == "none":
            return gpu_memory_usage, gpu_memory_total, gpu_utilization

        try:
            if gpu_access == "container":
                # 在容器内执行nvidia-smi
                cmd = [
                    "docker",
                    "exec",
                    self.container_name,
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ]
            else:
                # 在宿主机上执行nvidia-smi
                cmd = [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ]

            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                return gpu_memory_usage, gpu_memory_total, gpu_utilization

            gpu_info = result.stdout.decode("utf-8").strip().split("\n")[0]

            if gpu_info:
                parts = [part.strip() for part in gpu_info.split(",")]
                if len(parts) >= 3:
                    gpu_memory_usage, gpu_memory_total, gpu_utilization = parts[:3]

        except (
            subprocess.CalledProcessError,
            IndexError,
            ValueError,
            AttributeError,
        ) as e:
            self.log_message(f"获取GPU使用信息失败: {e}", "DEBUG")

        return gpu_memory_usage, gpu_memory_total, gpu_utilization

    def get_container_usage(self) -> tuple[float, str, str, float, float]:
        """获取容器内存和CPU使用情况

        Returns:
            (mem_usage_mb, mem_usage_str, mem_limit_str, cpu_percent, mem_limit_mb)
        """
        try:
            # 使用docker stats获取容器统计信息
            result = subprocess.run(
                [
                    "docker",
                    "stats",
                    "--no-stream",
                    "--format",
                    "{{.MemPerc}}|{{.MemUsage}}|{{.CPUPerc}}",
                    self.container_name,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            if result.returncode != 0 or not result.stdout:
                self.log_message("无法获取容器统计信息", "ERROR")
                return 0.0, "N/A", "N/A", 0.0, 0.0

            # 解析输出
            output = result.stdout.decode("utf-8").strip()
            if not output:
                self.log_message("无法获取容器统计信息", "ERROR")
                return 0.0, "N/A", "N/A", 0.0, 0.0

            parts = output.split("|")

            if len(parts) < 3:
                self.log_message("解析容器统计信息失败", "ERROR")
                return 0.0, "N/A", "N/A", 0.0, 0.0

            mem_percent_str, mem_usage_str, cpu_percent_str = parts[:3]

            # 清理和转换CPU数据
            try:
                cpu_percent = (
                    float(cpu_percent_str.replace("%", "").strip())
                    if cpu_percent_str.replace("%", "").replace(".", "").strip().isdigit()
                    else 0.0
                )
            except (ValueError, AttributeError):
                cpu_percent = 0.0

            # 解析内存使用量和限制
            mem_parts = mem_usage_str.split("/")
            mem_usage_str_clean = mem_parts[0].strip() if len(mem_parts) > 0 else "N/A"
            mem_limit_str_clean = mem_parts[1].strip() if len(mem_parts) > 1 else "N/A"

            # 转换为MB单位
            mem_usage_mb = self.convert_to_mb(mem_usage_str_clean)
            mem_limit_mb = self.convert_to_mb(mem_limit_str_clean)

            return (
                mem_usage_mb,
                mem_usage_str_clean,
                mem_limit_str_clean,
                cpu_percent,
                mem_limit_mb,
            )

        except (
            subprocess.CalledProcessError,
            ValueError,
            IndexError,
            AttributeError,
        ) as e:
            self.log_message(f"获取容器使用数据失败: {e}", "ERROR")
            return 0.0, "N/A", "N/A", 0.0, 0.0

    def format_gpu_display(self, gpu_memory_usage: str, gpu_memory_total: str, gpu_utilization: str) -> tuple[str, str]:
        """格式化GPU显示信息

        Args:
            gpu_memory_usage: GPU显存使用量
            gpu_memory_total: GPU显存总量
            gpu_utilization: GPU利用率

        Returns:
            (gpu_mem_display, gpu_util_display)
        """
        gpu_mem_display = "N/A"
        gpu_util_display = "N/A"

        try:
            if gpu_memory_usage.isdigit() and gpu_memory_total.isdigit():
                usage = int(gpu_memory_usage)
                total = int(gpu_memory_total)
                if total > 0:
                    percent = (usage / total) * 100
                    gpu_mem_display = f"{usage}MB/{total}MB ({percent:.1f}%)"
                else:
                    gpu_mem_display = f"{gpu_memory_usage}MB/{gpu_memory_total}MB"
            elif gpu_memory_usage.isdigit():
                gpu_mem_display = f"{gpu_memory_usage}MB/N/A"
        except (ValueError, ZeroDivisionError, AttributeError):
            pass

        try:
            if gpu_utilization.isdigit():
                gpu_util_display = f"{gpu_utilization}%"
        except AttributeError:
            pass

        return gpu_mem_display, gpu_util_display

    def print_header(self, gpu_access: str):
        """打印表头

        Args:
            gpu_access: GPU访问类型
        """
        if gpu_access != "none":
            print(
                "时间戳                | 内存使用(MB) | 滑动平均(MB) | 内存使用量 | 内存限制 | CPU使用率 | GPU显存使用/总量 | GPU利用率 | 泄漏检测 | 状态"
            )
            print(
                "--------------------------------------------------------------------------------------------------------------------------------"
            )
        else:
            print(
                "时间戳                | 内存使用(MB) | 滑动平均(MB) | 内存使用量 | 内存限制 | CPU使用率 | 泄漏检测 | 状态"
            )
            print(
                "------------------------------------------------------------------------------------------------------"
            )

    def print_data_row(
        self,
        timestamp: str,
        mem_usage_mb: float,
        sliding_avg_mb: float,
        mem_usage: str,
        mem_limit: str,
        cpu_percent: float,
        gpu_mem_display: str,
        gpu_util_display: str,
        leak_status: str,
        status: str,
        gpu_access: str,
    ):
        """打印数据行

        Args:
            timestamp: 时间戳
            mem_usage_mb: 内存使用量(MB)
            sliding_avg_mb: 滑动平均内存(MB)
            mem_usage: 内存使用量字符串
            mem_limit: 内存限制字符串
            cpu_percent: CPU使用百分比
            gpu_mem_display: GPU显存显示信息
            gpu_util_display: GPU利用率显示信息
            leak_status: 内存泄漏状态
            status: 状态
            gpu_access: GPU访问类型
        """
        # 状态颜色
        status_color = self.colors["GREEN"] if status == "正常" else self.colors["YELLOW"]
        leak_color = self.colors["GREEN"] if "正常" in leak_status else self.colors["RED"]

        if gpu_access != "none":
            print(
                f"{timestamp} | {mem_usage_mb:>11.1f} | {sliding_avg_mb:>11.1f} | {mem_usage:>10} | {mem_limit:>8} | {cpu_percent:>8.1f}% | {gpu_mem_display:>16} | {gpu_util_display:>9} | {leak_color}{leak_status:>12}{self.colors['NC']} | {status_color}{status:>6}{self.colors['NC']}"
            )
        else:
            print(
                f"{timestamp} | {mem_usage_mb:>11.1f} | {sliding_avg_mb:>11.1f} | {mem_usage:>10} | {mem_limit:>8} | {cpu_percent:>8.1f}% | {leak_color}{leak_status:>12}{self.colors['NC']} | {status_color}{status:>6}{self.colors['NC']}"
            )

    def monitor_container(self):
        """主监控函数"""
        check_count = 0
        alert_triggered = False
        leak_alert_triggered = False

        # 检查GPU可用性
        gpu_access = self.check_nvidia_gpu()
        if gpu_access == "container":
            self.log_message("检测到容器内可访问NVIDIA GPU", "INFO")
        elif gpu_access == "host":
            self.log_message("检测到宿主机有NVIDIA GPU，但容器内可能无法直接访问", "INFO")
        else:
            self.log_message("未检测到NVIDIA GPU或nvidia-smi不可用", "INFO")

        self.log_message(f"开始监控容器: {self.container_name}", "INFO")
        self.log_message(
            f"检查间隔: {self.check_interval}秒, 最��检查次数: {self.max_checks}",
            "INFO",
        )
        self.log_message(f"内存阈值: {self.memory_threshold}MB", "INFO")
        self.log_message(f"内存泄漏检测阈值: {self.memory_leak_threshold}MB", "INFO")
        self.log_message(f"滑动窗口大小: {self.sliding_window_size}", "INFO")
        self.log_message(f"日志文件: {self.log_file}", "INFO")
        self.log_message(f"数据文件: {self.csv_file}", "INFO")

        # 打印表头
        self.print_header(gpu_access)

        while check_count < self.max_checks:
            # 获取容器使用情况
            mem_usage_mb, mem_usage_str, mem_limit_str, cpu_percent, mem_limit_mb = self.get_container_usage()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 添加到内存历史记录
            self.memory_history.append(mem_usage_mb)

            # 计算滑动平均内存(MB)
            sliding_avg_mb = self.calculate_sliding_average(mem_usage_mb)

            # 检测内存泄漏
            leak_status = self.detect_memory_leak(mem_usage_mb, sliding_avg_mb)

            # 获取GPU使用情况
            gpu_memory_usage, gpu_memory_total, gpu_utilization = self.get_gpu_usage(gpu_access)
            gpu_mem_display, gpu_util_display = self.format_gpu_display(
                gpu_memory_usage, gpu_memory_total, gpu_utilization
            )

            # 状态指示
            status = "正常"
            if mem_usage_mb > self.memory_threshold:
                status = "警告"
                if not alert_triggered:
                    self.log_message(
                        f"内存使用量超过阈值: {mem_usage_mb:.1f}MB > {self.memory_threshold}MB",
                        "WARN",
                    )
                    alert_triggered = True

            # 内存泄漏检查
            if "可能泄漏" in leak_status and not leak_alert_triggered:
                self.log_message(
                    f"检测到可能的内存泄漏: 当前内存 {mem_usage_mb:.1f}MB, 滑动平均 {sliding_avg_mb:.1f}MB",
                    "WARN",
                )
                leak_alert_triggered = True

            # 记录到日志和CSV
            if gpu_access != "none":
                self.log_message(
                    f"内存: {mem_usage_mb:.1f}MB, 滑动平均: {sliding_avg_mb:.1f}MB, 使用量: {mem_usage_str}, 限制: {mem_limit_str}, CPU: {cpu_percent:.1f}%, GPU显存: {gpu_mem_display}, GPU利用率: {gpu_util_display}, 泄漏检测: {leak_status}, 状态: {status}",
                    "DATA",
                )
            else:
                self.log_message(
                    f"内存: {mem_usage_mb:.1f}MB, 滑动平均: {sliding_avg_mb:.1f}MB, 使用量: {mem_usage_str}, 限制: {mem_limit_str}, CPU: {cpu_percent:.1f}%, 泄漏检测: {leak_status}, 状态: {status}",
                    "DATA",
                )

            self.log_csv_data(
                timestamp,
                mem_usage_mb,
                sliding_avg_mb,
                mem_usage_str,
                mem_limit_str,
                cpu_percent,
                gpu_memory_usage,
                gpu_memory_total,
                gpu_utilization,
                leak_status,
                status,
            )

            # 显示输出
            self.print_data_row(
                timestamp,
                mem_usage_mb,
                sliding_avg_mb,
                mem_usage_str,
                mem_limit_str,
                cpu_percent,
                gpu_mem_display,
                gpu_util_display,
                leak_status,
                status,
                gpu_access,
            )

            check_count += 1

            if check_count < self.max_checks:
                time.sleep(self.check_interval)

        self.log_message(f"监控完成，共进行 {check_count} 次检查", "INFO")

        # 显示统计信息和内存泄漏分析
        self._print_statistics()

    def _print_statistics(self):
        """打印统计信息和内存泄漏分析"""
        if not self.memory_history:
            self.log_message("警告: 没有收集到有效的内存使用数据", "WARN")
            return

        # 基本统计
        try:
            max_mem = max(self.memory_history)
            min_mem = min(self.memory_history)
            avg_mem = statistics.mean(self.memory_history)

            self.log_message("=== 统计摘要 ===", "INFO")
            self.log_message(f"有效数据点数: {len(self.memory_history)}", "INFO")
            self.log_message(f"最小内存使用量: {min_mem:.1f}MB", "INFO")
            self.log_message(f"最大内存使用量: {max_mem:.1f}MB", "INFO")
            self.log_message(f"平均内存使用量: {avg_mem:.1f}MB", "INFO")

            # 内存泄漏趋势分析
            if len(self.memory_history) >= self.sliding_window_size:
                first_avg = statistics.mean(self.memory_history[: self.sliding_window_size])
                last_avg = statistics.mean(self.memory_history[-self.sliding_window_size :])
                trend_diff = last_avg - first_avg

                self.log_message("=== 内存泄漏趋势分析 ===", "INFO")
                self.log_message(f"初始窗口平均内存: {first_avg:.1f}MB", "INFO")
                self.log_message(f"最终窗口平均内存: {last_avg:.1f}MB", "INFO")
                self.log_message(f"内存变化趋势: {trend_diff:+.1f}MB", "INFO")

                if trend_diff > self.memory_leak_threshold:
                    self.log_message("警告: 检测到持续内存增长趋势，可能存在内存泄漏!", "WARN")
                elif trend_diff > 0:
                    self.log_message("注意: 内存使用有轻微增长趋势", "INFO")
                else:
                    self.log_message("内存使用趋势稳定", "INFO")
        except (ValueError, statistics.StatisticsError) as e:
            self.log_message(f"计算统计信息时出错: {e}", "ERROR")


def validate_sliding_average_calculation():
    """验证滑动平均计算是否正确"""
    print("=== 滑动平均计算验证 ===")

    test_cases = [
        {
            "name": "递增序列",
            "data": [100, 200, 300, 400, 500, 600, 700],  # MB单位
            "expected": [100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0],
        },
        {
            "name": "递减序列",
            "data": [700, 600, 500, 400, 300, 200, 100],  # MB单位
            "expected": [700.0, 650.0, 600.0, 550.0, 500.0, 400.0, 300.0],
        },
        {
            "name": "小数序列",
            "data": [100.5, 200.3, 300.7, 400.1, 500.9, 600.2, 700.8],  # MB单位
            "expected": [100.5, 150.4, 200.5, 250.4, 300.5, 400.44, 500.58],
        },
    ]

    for test_case in test_cases:
        print(f"\n测试: {test_case['name']}")
        print("数据:", test_case["data"], "MB")
        print("预期:", test_case["expected"], "MB")

        # 创建临时监控器进行测试
        monitor = DockerContainerMonitor("test", sliding_window_size=5)
        monitor.sliding_window = deque(maxlen=5)

        computed = []
        for value in test_case["data"]:
            avg = monitor.calculate_sliding_average(value)
            computed.append(round(avg, 2))

        print("计算:", computed, "MB")

        # 验证结果
        all_correct = True
        for i, (comp, exp) in enumerate(zip(computed, test_case["expected"])):
            if abs(comp - exp) < 0.01:
                print(f"  步骤 {i+1}: ✓ 通过")
            else:
                print(f"  步骤 {i+1}: ✗ 失败 (计算: {comp}, 预期: {exp})")
                all_correct = False

        if all_correct:
            print("✓ 所有测试通过")
        else:
            print("✗ 部分测试失败")

    print("\n=== 验证完成 ===")


def check_docker_available():
    """检查Docker是否可用"""
    try:
        result = subprocess.run(["docker", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def check_container_exists(container_name):
    """检查容器是否存在"""
    try:
        result = subprocess.run(
            ["docker", "inspect", container_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.returncode == 0
    except subprocess.CalledProcessError:
        return False


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Docker容器监控脚本 - 监控容器资源使用情况和内存泄漏",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s my-container
  %(prog)s my-container -i 10 -m 120
  %(prog)s my-container -t 2048 -l 200  # 内存阈值2GB，泄漏阈值200MB
  %(prog)s my-container -f /var/log/container.log -c /var/data/usage.csv
  %(prog)s --validate
        """,
    )

    # 必需参数
    parser.add_argument("container_name", nargs="?", help="要监控的Docker容器名称或ID")

    # 监控参数
    parser.add_argument("-i", "--interval", type=int, default=30, help="检查间隔时间（秒），默认30秒")

    parser.add_argument("-m", "--max-checks", type=int, default=60, help="最大检查次数，默认60次")

    # 阈值参数 - 改为MB单位
    parser.add_argument(
        "-t",
        "--memory-threshold",
        type=float,
        default=1024.0,
        help="内存使用量阈值（MB），超过此值会发出警告，默认1024MB(1GB)",
    )

    parser.add_argument(
        "-l",
        "--memory-leak-threshold",
        type=float,
        default=100.0,
        help="内存泄漏检测阈值（MB），当前内存与滑动平均差值超过此值会发出警告，默认100MB",
    )

    parser.add_argument(
        "-w",
        "--window-size",
        type=int,
        default=10,
        help="滑动窗口大小，用于计算内存使用量的滑动平均值，默认10",
    )

    # 文件路径参数
    parser.add_argument(
        "-f",
        "--log-file",
        default="container_monitor.log",
        help="日志文件路径，默认container_monitor.log",
    )

    parser.add_argument(
        "-c",
        "--csv-file",
        default="container_usage.csv",
        help="CSV数据文件路径，默认container_usage.csv",
    )

    # 特殊模式
    parser.add_argument("--validate", action="store_true", help="验证滑动平均计算是否正确")

    parser.add_argument("--version", action="version", version="%(prog)s 2.0.0")

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()

    # 验证模式
    if args.validate:
        validate_sliding_average_calculation()
        sys.exit(0)

    # 检查必需参数
    if not args.container_name:
        print("错误: 必须指定要监控的容器名称")
        print("使用方法: python3 docker_monitor.py <容器名称> [选项]")
        print("使用 --help 查看完整选项")
        sys.exit(1)

    # 检查Docker是否可用
    if not check_docker_available():
        print("错误: Docker未安装或未在PATH中")
        sys.exit(1)

    # 检查容器是否存在
    if not check_container_exists(args.container_name):
        print(f"错误: 容器 '{args.container_name}' 不存在或无法访问")
        sys.exit(1)

    # 验证参数合理性
    if args.interval <= 0:
        print("错误: 检查间隔必须大于0")
        sys.exit(1)

    if args.max_checks <= 0:
        print("错误: 最大检查次数必须大于0")
        sys.exit(1)

    if args.memory_threshold <= 0:
        print("错误: 内存阈值必须大于0")
        sys.exit(1)

    if args.memory_leak_threshold <= 0:
        print("错误: 内存泄漏阈值必须大于0")
        sys.exit(1)

    if args.window_size <= 0:
        print("错误: 滑动窗口大小必须大于0")
        sys.exit(1)

    # 显示配置信息
    print("=== Docker容器监控配置 ===")
    print(f"容器名称: {args.container_name}")
    print(f"检查间隔: {args.interval}秒")
    print(f"最大检查次数: {args.max_checks}")
    print(f"内存阈值: {args.memory_threshold}MB")
    print(f"内存泄漏阈值: {args.memory_leak_threshold}MB")
    print(f"滑动窗口大小: {args.window_size}")
    print(f"日志文件: {args.log_file}")
    print(f"数据文件: {args.csv_file}")
    print("==========================")
    print()

    # 创建并启动监控器
    monitor = DockerContainerMonitor(
        container_name=args.container_name,
        check_interval=args.interval,
        max_checks=args.max_checks,
        memory_threshold=args.memory_threshold,
        memory_leak_threshold=args.memory_leak_threshold,
        sliding_window_size=args.window_size,
        log_file=args.log_file,
        csv_file=args.csv_file,
    )

    try:
        monitor.monitor_container()
    except KeyboardInterrupt:
        print("\n监控被用户中断")
    except Exception as e:
        print(f"监控过程中发生错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
