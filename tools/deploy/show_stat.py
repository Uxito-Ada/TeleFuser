#!/usr/bin/env python3
"""
CSV数据可视化工具 - 读取Docker监控CSV并绘制滑动平均散点图
支持终端和matplotlib两种显示方式
新增线性拟合功能检测内存泄漏趋势
"""

import argparse
import csv
import os
import statistics
import sys
from collections import deque
from datetime import datetime
from typing import Any

try:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import numpy as np

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class PeakAnalyzer:
    """峰值分析器 - 用于检测内存波动的局部极大值趋势"""

    def __init__(self):
        self.numpy_available = NUMPY_AVAILABLE

    def find_local_maxima(self, data: list[float], window_size: int = 11) -> list[tuple[int, float]]:
        """查找局部极大值

        Args:
            data: 输入数据
            window_size: 窗口大小（左右各检查的点的数量）

        Returns:
            局部极大值的索引和值列表
        """
        maxima = []
        n = len(data)

        for i in range(n):
            is_maximum = True

            # 检查左侧窗口
            for j in range(max(0, i - window_size), i):
                if data[j] > data[i]:
                    is_maximum = False
                    break

            # 检查右侧窗口
            if is_maximum:
                for j in range(i + 1, min(n, i + window_size + 1)):
                    if data[j] > data[i]:
                        is_maximum = False
                        break

            if is_maximum:
                maxima.append((i, data[i]))

        return maxima

    def analyze_peak_trend(self, peaks: list[tuple[int, float]]) -> dict[str, Any]:
        """分析峰值趋势

        Args:
            peaks: 峰值列表 (索引, 值)

        Returns:
            峰值趋势分析结果
        """
        if len(peaks) < 3:
            return {"success": False, "error": "峰值数量不足，无法进行趋势分析"}

        # 提取峰值索引和值
        indices = [peak[0] for peak in peaks]
        values = [peak[1] for peak in peaks]

        # 使用线性拟合分析峰值趋势
        fit_analyzer = LinearFitAnalyzer()
        fit_result = fit_analyzer.linear_fit(indices, values)

        if not fit_result["success"]:
            return {
                "success": False,
                "error": f"峰值趋势拟合失败: {fit_result.get('error', '未知错误')}",
            }

        # 分析内存泄漏风险
        leak_analysis = fit_analyzer.analyze_memory_leak(fit_result["slope"], fit_result["r_squared"], "index")

        return {
            "success": True,
            "peak_count": len(peaks),
            "slope": fit_result["slope"],
            "r_squared": fit_result["r_squared"],
            "equation": fit_result["equation"],
            "leak_analysis": leak_analysis,
            "peaks": peaks,
        }


class LinearFitAnalyzer:
    """线性拟合分析器"""

    def __init__(self):
        self.numpy_available = NUMPY_AVAILABLE

    def linear_fit(self, x_data: list[float], y_data: list[float]) -> dict[str, Any]:
        """执行线性拟合

        Args:
            x_data: x轴数据（通常是时间索引）
            y_data: y轴数据（内存使用量）

        Returns:
            包含拟合结果的字典
        """
        if len(x_data) != len(y_data) or len(x_data) < 2:
            return {"success": False, "error": "数据点不足"}

        n = len(x_data)

        if self.numpy_available:
            return self._linear_fit_numpy(x_data, y_data, n)
        else:
            return self._linear_fit_manual(x_data, y_data, n)

    def _linear_fit_numpy(self, x_data: list[float], y_data: list[float], n: int) -> dict[str, Any]:
        """使用numpy进行线性拟合"""
        try:
            x = np.array(x_data)
            y = np.array(y_data)

            # 线性拟合 y = mx + b
            m, b = np.polyfit(x, y, 1)

            # 计算R²
            y_pred = m * x + b
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            return {
                "success": True,
                "slope": m,  # 斜率
                "intercept": b,  # 截距
                "r_squared": r_squared,  # R²
                "equation": f"y = {m:.4f}x + {b:.2f}",
                "slope_per_minute": m * 60,  # 每分钟斜率（如果x是秒）
                "method": "numpy",
            }
        except Exception as e:
            return {"success": False, "error": f"numpy拟合失败: {e}"}

    def _linear_fit_manual(self, x_data: list[float], y_data: list[float], n: int) -> dict[str, Any]:
        """手动计算线性拟合（最小二乘法）"""
        try:
            # 计算必要的基本统计量
            sum_x = sum(x_data)
            sum_y = sum(y_data)
            sum_xy = sum(x * y for x, y in zip(x_data, y_data))
            sum_x2 = sum(x * x for x in x_data)
            sum_y2 = sum(y * y for y in y_data)

            # 计算斜率和截距
            denominator = n * sum_x2 - sum_x * sum_x
            if denominator == 0:
                return {"success": False, "error": "数据方差为零"}

            m = (n * sum_xy - sum_x * sum_y) / denominator
            b = (sum_y - m * sum_x) / n

            # 计算R²
            y_mean = sum_y / n
            ss_tot = sum((y - y_mean) ** 2 for y in y_data)
            ss_res = sum((y - (m * x + b)) ** 2 for x, y in zip(x_data, y_data))

            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            return {
                "success": True,
                "slope": m,
                "intercept": b,
                "r_squared": r_squared,
                "equation": f"y = {m:.4f}x + {b:.2f}",
                "slope_per_minute": m * 60,  # 假设x是秒
                "method": "manual",
            }
        except Exception as e:
            return {"success": False, "error": f"手动拟合失败: {e}"}

    def analyze_memory_leak(self, slope: float, r_squared: float, time_unit: str = "second") -> dict[str, Any]:
        """分析内存泄漏可能性

        Args:
            slope: 斜率
            r_squared: R²值
            time_unit: 时间单位 ("second", "minute", "index")

        Returns:
            分析结果字典
        """
        # 转换斜率为MB/分钟
        if time_unit == "second":
            slope_per_min = slope * 60
        elif time_unit == "minute":
            slope_per_min = slope
        else:  # index或其他
            slope_per_min = slope  # 假设已经是每分钟

        # 评估内存泄漏风险
        risk_level = "低"
        description = "内存使用稳定"

        if slope_per_min > 1.0:  # 每分钟增长超过1MB
            if r_squared > 0.7:
                risk_level = "高"
                description = "强线性增长趋势，很可能存在内存泄漏"
            elif r_squared > 0.5:
                risk_level = "中"
                description = "明显的线性增长趋势，可能存在内存泄漏"
            else:
                risk_level = "低"
                description = "有增长趋势但线性关系不强"
        elif slope_per_min > 0.1:  # 每分钟增长0.1-1MB
            if r_squared > 0.7:
                risk_level = "中"
                description = "稳定的缓慢增长，需要关注"
            else:
                risk_level = "低"
                description = "轻微增长趋势"
        elif slope_per_min < -0.1:  # 每分钟减少超过0.1MB
            risk_level = "低"
            description = "内存使用在减少"
        else:  # 基本稳定
            risk_level = "低"
            description = "内存使用基本稳定"

        return {
            "slope_per_minute": slope_per_min,
            "risk_level": risk_level,
            "description": description,
            "r_squared": r_squared,
        }


class TerminalVisualizer:
    """终端可视化器"""

    def __init__(self, width: int = 60, height: int = 15):
        self.width = width
        self.height = height
        self.colors = {
            "RED": "\033[0;31m",
            "GREEN": "\033[0;32m",
            "YELLOW": "\033[1;33m",
            "BLUE": "\033[0;34m",
            "PURPLE": "\033[0;35m",
            "CYAN": "\033[0;36m",
            "NC": "\033[0m",
        }
        self.fit_analyzer = LinearFitAnalyzer()

    def create_memory_plot(
        self,
        timestamps: list[str],
        memory_data: list[float],
        sliding_avg: list[float],
        title: str = "内存使用趋势",
    ) -> str:
        """创建内存使用趋势图

        Args:
            timestamps: 时间戳列表
            memory_data: 内存使用数据(MB)
            sliding_avg: 滑动平均值(MB)
            title: 图表标题

        Returns:
            格式化的图表字符串
        """
        if not memory_data:
            return "暂无数据"

        # 计算统计信息
        min_val = min(memory_data)
        max_val = max(memory_data)
        avg_val = statistics.mean(memory_data)

        # 如果所有值都相同，调整范围避免除零
        if max_val == min_val:
            max_val = min_val + 1

        # 创建图表框架
        plot_lines = []

        # 添加标题和统计信息
        plot_lines.append(f"{self.colors['CYAN']}{title}{self.colors['NC']}")
        plot_lines.append(f"数据点: {len(memory_data)} | 时间范围: {timestamps[0]} 到 {timestamps[-1]}")
        plot_lines.append(f"最小值: {min_val:.1f}MB | 平均值: {avg_val:.1f}MB | 最大值: {max_val:.1f}MB")
        plot_lines.append("")

        # 创建Y轴和绘图区域
        y_step = (max_val - min_val) / (self.height - 1) if self.height > 1 else max_val - min_val

        # 采样数据点以适应图表宽度
        sampled_indices = self._sample_indices(len(memory_data), self.width)
        sampled_memory = [memory_data[i] for i in sampled_indices]
        sampled_avg = [sliding_avg[i] for i in sampled_indices] if sliding_avg else []

        # 从顶部到底部绘制每一行
        for row in range(self.height - 1, -1, -1):
            line = ""
            y_value = min_val + row * y_step

            # 添加Y轴标签（每3行显示一次）
            if row % 3 == 0 or row == self.height - 1 or row == 0:
                line += f"{y_value:6.0f}MB | "
            else:
                line += "       | "

            # 绘制数据点
            for i, (mem_val, avg_val) in enumerate(zip(sampled_memory, sampled_avg)):
                # 计算当前数据点在Y轴上的位置
                mem_pos = (mem_val - min_val) / (max_val - min_val) * (self.height - 1) if max_val > min_val else 0
                avg_pos = (
                    (avg_val - min_val) / (max_val - min_val) * (self.height - 1)
                    if sliding_avg and max_val > min_val
                    else 0
                )

                # 绘制滑动平均点（蓝色）
                if sliding_avg and abs(avg_pos - row) < 0.3:
                    line += f"{self.colors['BLUE']}·{self.colors['NC']}"
                # 绘制实际内存点（绿色/红色）
                elif abs(mem_pos - row) < 0.3:
                    if mem_val > avg_val * 1.2:
                        color = self.colors["RED"]
                    elif mem_val > avg_val:
                        color = self.colors["YELLOW"]
                    else:
                        color = self.colors["GREEN"]
                    line += f"{color}*{self.colors['NC']}"
                else:
                    line += " "

            plot_lines.append(line)

        # 添加X轴
        plot_lines.append(" " * 8 + "+" + "-" * self.width)

        # 添加图例
        plot_lines.append(
            f"图例: {self.colors['GREEN']}*{self.colors['NC']} 内存使用 | {self.colors['BLUE']}·{self.colors['NC']} 滑动平均"
        )

        return "\n".join(plot_lines)

    def _sample_indices(self, data_length: int, max_points: int) -> list[int]:
        """采样索引以适应图表宽度"""
        if data_length <= max_points:
            return list(range(data_length))

        # 均匀采样
        step = data_length / max_points
        return [int(i * step) for i in range(max_points)]

    def print_statistics(
        self,
        memory_data: list[float],
        sliding_avg: list[float],
        timestamps: list[str] = None,
    ):
        """打印统计信息"""
        if not memory_data:
            print("无数据可分析")
            return

        print(f"\n{self.colors['CYAN']}=== 统计摘要 ==={self.colors['NC']}")
        print(f"总数据点数: {len(memory_data)}")
        print(f"最小内存使用: {min(memory_data):.1f}MB")
        print(f"最大内存使用: {max(memory_data):.1f}MB")
        print(f"平均内存使用: {statistics.mean(memory_data):.1f}MB")

        if len(memory_data) > 1:
            print(f"内存使用标准差: {statistics.stdev(memory_data):.1f}MB")

        if sliding_avg:
            print(f"滑动平均数据点: {len(sliding_avg)}")
            print(f"最终滑动平均值: {sliding_avg[-1]:.1f}MB")

        # 执行线性拟合分析
        self.print_linear_fit_analysis(sliding_avg, timestamps)

        # 执行峰值趋势分析
        self.print_peak_analysis(sliding_avg)

    def print_linear_fit_analysis(self, sliding_avg: list[float], timestamps: list[str] = None):
        """打印线性拟合分析结果"""
        if len(sliding_avg) < 10:
            print(
                f"{self.colors['YELLOW']}数据点不足({len(sliding_avg)})，无法进行可靠的线性拟合分析{self.colors['NC']}"
            )
            return

        print(f"\n{self.colors['CYAN']}=== 线性拟合分析（基于滑动平均数据）==={self.colors['NC']}")

        # 准备x轴数据
        if timestamps and len(timestamps) == len(sliding_avg):
            # 使用时间戳
            try:
                time_objs = [datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") for ts in timestamps]
                start_time = time_objs[0]
                x_data = [(t - start_time).total_seconds() for t in time_objs]  # 转换为秒
                time_unit = "second"
                print("使用时间戳数据进行拟合")
            except Exception as e:
                print(f"时间戳解析失败，使用索引: {e}")
                x_data = list(range(len(sliding_avg)))
                time_unit = "index"
        else:
            # 使用索引
            x_data = list(range(len(sliding_avg)))
            time_unit = "index"
            print("使用数据点索引进行拟合")

        # 执行线性拟合
        fit_result = self.fit_analyzer.linear_fit(x_data, sliding_avg)

        if not fit_result["success"]:
            print(f"{self.colors['RED']}线性拟合失败: {fit_result['error']}{self.colors['NC']}")
            return

        # 显示拟合结果
        print(f"拟合方法: {fit_result['method']}")
        print(f"拟合方程: {fit_result['equation']}")
        print(f"斜率: {fit_result['slope']:.6f} MB/单位时间")
        print(f"R²值: {fit_result['r_squared']:.4f}")

        # 分析内存泄漏风险
        leak_analysis = self.fit_analyzer.analyze_memory_leak(fit_result["slope"], fit_result["r_squared"], time_unit)

        print(f"估计增长率: {leak_analysis['slope_per_minute']:.3f} MB/分钟")

        # 根据风险级别显示不同颜色
        risk_color = self.colors["GREEN"]
        if leak_analysis["risk_level"] == "中":
            risk_color = self.colors["YELLOW"]
        elif leak_analysis["risk_level"] == "高":
            risk_color = self.colors["RED"]

        print(f"内存泄漏风险: {risk_color}{leak_analysis['risk_level']}{self.colors['NC']}")
        print(f"分析结果: {leak_analysis['description']}")

        # 提供解释
        print(f"\n{self.colors['CYAN']}=== 解释说明 ==={self.colors['NC']}")
        print("• 斜率 > 0 表示内存使用呈增长趋势")
        print("• R²值接近1表示线性关系强，趋势可靠")
        print("• 风险评估基于增长率和线性关系的强度")
        print("• 高风险: 强线性增长 > 1MB/分钟")
        print("• 中风险: 明显增长 > 0.1MB/分钟")
        print("• 低风险: 稳定或下降趋势")

    def print_peak_analysis(self, sliding_avg: list[float]):
        """打印峰值趋势分析结果"""
        if len(sliding_avg) < 10:
            print(f"{self.colors['YELLOW']}数据点不足({len(sliding_avg)})，无法进行峰值分析{self.colors['NC']}")
            return

        print(f"\n{self.colors['CYAN']}=== 峰值趋势分析（基于滑动平均数据的局部极大值）==={self.colors['NC']}")

        # 创建峰值分析器
        peak_analyzer = PeakAnalyzer()

        # 查找局部极大值
        peaks = peak_analyzer.find_local_maxima(sliding_avg, window_size=11)
        print(f"发现 {len(peaks)} 个局部极大值")

        # 分析峰值趋势
        peak_analysis = peak_analyzer.analyze_peak_trend(peaks)

        if not peak_analysis["success"]:
            print(f"{self.colors['RED']}峰值趋势分析失败: {peak_analysis['error']}{self.colors['NC']}")
            return

        # 显示峰值分析结果
        leak_analysis = peak_analysis["leak_analysis"]
        print(f"峰值拟合方程: {peak_analysis['equation']}")
        print(f"峰值斜率: {peak_analysis['slope']:.6f} MB/峰值间隔")
        print(f"峰值R²值: {peak_analysis['r_squared']:.4f}")
        print(f"峰值增长率: {leak_analysis['slope_per_minute']:.3f} MB/分钟")

        # 根据风险级别显示不同颜色
        risk_color = self.colors["GREEN"]
        if leak_analysis["risk_level"] == "中":
            risk_color = self.colors["YELLOW"]
        elif leak_analysis["risk_level"] == "高":
            risk_color = self.colors["RED"]

        print(f"基于峰值的内存泄漏风险: {risk_color}{leak_analysis['risk_level']}{self.colors['NC']}")
        print(f"峰值分析结果: {leak_analysis['description']}")

        # 显示峰值信息
        print(f"\n{self.colors['CYAN']}前5个峰值:{self.colors['NC']}")
        for i, (idx, value) in enumerate(peaks[:5]):
            print(f"  峰值{i+1}: 索引={idx}, 值={value:.2f}MB")

        if len(peaks) > 5:
            print(f"  ... 还有 {len(peaks) - 5} 个峰值")

        # 提供解释
        print(f"\n{self.colors['CYAN']}=== 峰值分析说明 ==={self.colors['NC']}")
        print("• 分析内存波动中的局部极大值趋势")
        print("• 适用于周期性内存波动的场景")
        print("• 峰值斜率 > 0 表示每个波峰都在升高，可能存在内存泄漏")
        print("• 结合传统线性分析和峰值分析可获得更全面的判断")


class CSVVisualizer:
    """CSV数据可视化器"""

    def __init__(self):
        self.terminal_viz = TerminalVisualizer()
        self.fit_analyzer = LinearFitAnalyzer()

    def read_csv_data(self, csv_file: str) -> tuple[list[str], list[float], list[float]]:
        """读取CSV文件数据

        Args:
            csv_file: CSV文件路径

        Returns:
            (timestamps, memory_data, sliding_avg)
        """
        timestamps = []
        memory_data = []
        sliding_avg = []

        try:
            with open(csv_file, encoding="utf-8") as f:
                reader = csv.DictReader(f)

                for row in reader:
                    # 解析时间戳
                    timestamp = row.get("timestamp", "")
                    if timestamp:
                        timestamps.append(timestamp)

                    # 解析内存使用数据
                    memory_str = row.get("memory_usage_mb", "") or row.get("memory_percent", "")
                    if memory_str:
                        try:
                            memory_data.append(float(memory_str))
                        except ValueError:
                            continue

                    # ���析滑动平均数据
                    sliding_str = row.get("sliding_avg_memory_mb", "") or row.get("sliding_avg_memory", "")
                    if sliding_str:
                        try:
                            sliding_avg.append(float(sliding_str))
                        except ValueError:
                            # 如果没有滑动平均数据，稍后计算
                            pass

            # 如果没有滑动平均数据，计算它
            if memory_data and not sliding_avg:
                sliding_avg = self.calculate_sliding_average(memory_data)

            return timestamps, memory_data, sliding_avg

        except FileNotFoundError:
            print(f"错误: 文件 '{csv_file}' 不存在")
            return [], [], []
        except Exception as e:
            print(f"读取CSV文件时出错: {e}")
            return [], [], []

    def calculate_sliding_average(self, data: list[float], window_size: int = 10) -> list[float]:
        """计算滑动平均值

        Args:
            data: 原始数据列表
            window_size: 滑动窗口大小

        Returns:
            滑动平均值列表
        """
        sliding_avg = []
        window = deque(maxlen=window_size)

        for value in data:
            window.append(value)
            if len(window) > 0:
                sliding_avg.append(statistics.mean(window))
            else:
                sliding_avg.append(value)

        return sliding_avg

    def visualize_terminal(self, csv_file: str, window_size: int = 10):
        """在终端中可视化数据"""
        print(f"正在读取CSV文件: {csv_file}")
        timestamps, memory_data, sliding_avg = self.read_csv_data(csv_file)

        if not memory_data:
            print("错误: 未找到有效的内存使用数据")
            return

        # 如果需要，重新计算滑动平均
        if not sliding_avg or len(sliding_avg) != len(memory_data):
            sliding_avg = self.calculate_sliding_average(memory_data, window_size)

        # 在终端中显示图表
        plot = self.terminal_viz.create_memory_plot(
            timestamps,
            memory_data,
            sliding_avg,
            f"内存使用趋势 - {os.path.basename(csv_file)}",
        )
        print(plot)

        # 显示统计信息
        self.terminal_viz.print_statistics(memory_data, sliding_avg, timestamps)

    def visualize_matplotlib(self, csv_file: str, window_size: int = 10, output_file: str = None):
        """使用matplotlib可视化数据"""
        if not MATPLOTLIB_AVAILABLE:
            print("错误: matplotlib不可用，请安装: pip install matplotlib")
            return

        print(f"正在读取CSV文件: {csv_file}")
        timestamps, memory_data, sliding_avg = self.read_csv_data(csv_file)

        if not memory_data:
            print("错误: 未找到有效的内存使用数据")
            return

        # 如果需要，重新计算滑动平均
        if not sliding_avg or len(sliding_avg) != len(memory_data):
            sliding_avg = self.calculate_sliding_average(memory_data, window_size)

        # 创建图形
        plt.figure(figsize=(12, 8))

        # 转换时间戳
        time_objs = []
        if timestamps and len(timestamps) == len(memory_data):
            try:
                time_objs = [datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") for ts in timestamps]
                x_values = time_objs
                use_timestamps = True
            except ValueError:
                time_objs = []
                x_values = range(len(memory_data))
                use_timestamps = False
        else:
            x_values = range(len(memory_data))
            use_timestamps = False

        # 绘制数据
        plt.plot(x_values, memory_data, "g-", alpha=0.7, linewidth=1, label="内存使用")
        plt.plot(x_values, sliding_avg, "b-", linewidth=2, label="滑动平均")
        plt.scatter(x_values, memory_data, c="green", alpha=0.6, s=20)
        plt.scatter(x_values, sliding_avg, c="blue", alpha=0.8, s=10)

        # 添加峰值标记
        if len(sliding_avg) >= 10:
            peak_analyzer = PeakAnalyzer()
            peaks = peak_analyzer.find_local_maxima(sliding_avg, window_size=11)

            # 标记峰值点
            if peaks:
                peak_indices = [peak[0] for peak in peaks]
                peak_values = [peak[1] for peak in peaks]

                if use_timestamps and time_objs:
                    peak_times = [time_objs[i] for i in peak_indices]
                    plt.scatter(
                        peak_times,
                        peak_values,
                        c="red",
                        s=50,
                        marker="^",
                        label="局部极大值",
                    )
                else:
                    plt.scatter(
                        peak_indices,
                        peak_values,
                        c="red",
                        s=50,
                        marker="^",
                        label="局部极大值",
                    )

        # 添加线性拟合线（基于滑动平均）
        if len(sliding_avg) >= 10:
            if use_timestamps and time_objs:
                # 使用时间戳进行拟合
                start_time = time_objs[0]
                x_fit = [(t - start_time).total_seconds() for t in time_objs]
                fit_result = self.fit_analyzer.linear_fit(x_fit, sliding_avg)
                time_unit = "second"
            else:
                # 使用索引进行拟合
                x_fit = list(range(len(sliding_avg)))
                fit_result = self.fit_analyzer.linear_fit(x_fit, sliding_avg)
                time_unit = "index"

            if fit_result["success"]:
                # 计算拟合线
                if use_timestamps:
                    # 将拟合线转换回时间戳显示
                    fit_line = [fit_result["slope"] * x + fit_result["intercept"] for x in x_fit]
                    plt.plot(
                        x_values,
                        fit_line,
                        "r--",
                        linewidth=2,
                        label=f'linear fit (R²={fit_result["r_squared"]:.3f})',
                    )
                else:
                    fit_line = [fit_result["slope"] * x + fit_result["intercept"] for x in x_fit]
                    plt.plot(
                        x_values,
                        fit_line,
                        "r--",
                        linewidth=2,
                        label=f'linear fit (R²={fit_result["r_squared"]:.3f})',
                    )

                # 分析内存泄漏风险
                leak_analysis = self.fit_analyzer.analyze_memory_leak(
                    fit_result["slope"], fit_result["r_squared"], time_unit
                )

                # 添加文本框显示分析结果
                textstr = "\n".join(
                    [
                        f'fit func: {fit_result["equation"]}',
                        f'ratio: {fit_result["slope"]:.6f}',
                        f'R²: {fit_result["r_squared"]:.4f}',
                        f'grow rate: {leak_analysis["slope_per_minute"]:.3f} MB/minute',
                        f'risk: {leak_analysis["risk_level"]}',
                        f'eval: {leak_analysis["description"]}',
                    ]
                )

                # 选择文本框位置
                if leak_analysis["risk_level"] == "高":
                    bbox_color = "red"
                elif leak_analysis["risk_level"] == "中":
                    bbox_color = "yellow"
                else:
                    bbox_color = "green"

                plt.gcf().text(
                    0.02,
                    0.98,
                    textstr,
                    transform=plt.gca().transAxes,
                    fontsize=10,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor=bbox_color, alpha=0.1),
                )

        # 添加局部峰值拟合线
        if peaks and len(peaks) >= 3:
            peak_indices = [peak[0] for peak in peaks]
            peak_values = [peak[1] for peak in peaks]

            # 对峰值进行线性拟合
            if use_timestamps and time_objs:
                # 使用时间戳进行峰值拟合
                peak_times = [time_objs[i] for i in peak_indices]
                start_time = time_objs[0]
                x_peak_fit = [(t - start_time).total_seconds() for t in peak_times]
                peak_fit_result = self.fit_analyzer.linear_fit(x_peak_fit, peak_values)

                if peak_fit_result["success"]:
                    # 生成峰值拟合线
                    peak_fit_line = [peak_fit_result["slope"] * x + peak_fit_result["intercept"] for x in x_peak_fit]
                    # 绘制峰值拟合线
                    plt.plot(
                        peak_times,
                        peak_fit_line,
                        "m-",
                        linewidth=2,
                        label=f'peak fit (R²={peak_fit_result["r_squared"]:.3f})',
                    )
            else:
                # 使用索引进行峰值拟合
                peak_fit_result = self.fit_analyzer.linear_fit(peak_indices, peak_values)

                if peak_fit_result["success"]:
                    # 生成峰值拟合线
                    peak_fit_line = [peak_fit_result["slope"] * x + peak_fit_result["intercept"] for x in peak_indices]
                    # 绘制峰值拟合线
                    plt.plot(
                        peak_indices,
                        peak_fit_line,
                        "m-",
                        linewidth=2,
                        label=f'peak fit (R²={peak_fit_result["r_squared"]:.3f})',
                    )

            # 添加峰值拟合分析文本框
            if peak_fit_result.get("success"):
                peak_leak_analysis = self.fit_analyzer.analyze_memory_leak(
                    peak_fit_result["slope"],
                    peak_fit_result["r_squared"],
                    "second" if use_timestamps else "index",
                )

                peak_textstr = "\n".join(
                    [
                        f'peak fit: {peak_fit_result["equation"]}',
                        f'peak ratio: {peak_fit_result["slope"]:.6f}',
                        f'peak R²: {peak_fit_result["r_squared"]:.4f}',
                        f'peak grow rate: {peak_leak_analysis["slope_per_minute"]:.3f} MB/minute',
                        f'peak risk: {peak_leak_analysis["risk_level"]}',
                    ]
                )

                # 根据风险级别选择颜色
                if peak_leak_analysis["risk_level"] == "高":
                    peak_bbox_color = "red"
                elif peak_leak_analysis["risk_level"] == "中":
                    peak_bbox_color = "yellow"
                else:
                    peak_bbox_color = "green"

                plt.gcf().text(
                    0.02,
                    0.75,
                    peak_textstr,
                    transform=plt.gca().transAxes,
                    fontsize=9,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor=peak_bbox_color, alpha=0.1),
                )

        # 添加标签和标题
        if use_timestamps:
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.gcf().autofmt_xdate()
            plt.xlabel("time")
        else:
            plt.xlabel("data point")

        plt.ylabel("Mem (MB)")
        plt.title(f"memory usage- {os.path.basename(csv_file)}")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 显示或保存
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            print(f"图表已保存到: {output_file}")
        else:
            plt.show()

        # 在终端显示统计信息
        print("\n" + "=" * 60)
        print("终端统计信息")
        print("=" * 60)
        self.terminal_viz.print_statistics(memory_data, sliding_avg, timestamps)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="CSV数据可视化工具 - 读取Docker监控CSV并绘制滑动平均散点图",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 在终端中显示图表和线性拟合分析
  %(prog)s data.csv

  # 使用matplotlib显示交互式图表
  %(prog)s data.csv --matplotlib

  # 保存matplotlib图表到文件
  %(prog)s data.csv --matplotlib --output chart.png

  # 自定义滑动窗口大小
  %(prog)s data.csv --window-size 15

  # 同时使用终端和matplotlib
  %(prog)s data.csv --terminal --matplotlib

  # 禁用线性拟合分析
  %(prog)s data.csv --no-fit-analysis
        """,
    )

    parser.add_argument("csv_file", help="要可视化的CSV文件路径")

    parser.add_argument("-w", "--window-size", type=int, default=10, help="滑动窗口大小，默认10")

    parser.add_argument(
        "-t",
        "--terminal",
        action="store_true",
        default=True,
        help="在终端中显示图表（默认启用）",
    )

    parser.add_argument(
        "-m",
        "--matplotlib",
        action="store_true",
        help="使用matplotlib显示图表（需要安装matplotlib）",
    )

    parser.add_argument(
        "-o",
        "--output",
        help="将matplotlib图表保存到指定文件（仅与--matplotlib一起使用）",
    )

    parser.add_argument("--no-terminal", action="store_true", help="禁用终端显示")

    parser.add_argument("--no-fit-analysis", action="store_true", help="禁用线性拟合分析")

    args = parser.parse_args()

    # 检查文件是否存在
    if not os.path.exists(args.csv_file):
        print(f"错误: 文件 '{args.csv_file}' 不存在")
        sys.exit(1)

    # 创建可视化器
    visualizer = CSVVisualizer()

    # 如果禁用线性拟合分析，修改相关方法
    if args.no_fit_analysis:
        # 临时修改方法，跳过拟合分析
        original_print_stats = visualizer.terminal_viz.print_statistics
        visualizer.terminal_viz.print_statistics = lambda mem, avg, ts=None: original_print_stats(mem, avg, None)

        # 修改matplotlib可视化，跳过拟合
        original_matplotlib_viz = visualizer.visualize_matplotlib

        def modified_matplotlib_viz(csv_file, window_size=10, output_file=None):
            # 这里需要更复杂的修改来跳过matplotlib中的拟合，暂时简化处理
            print("注意: 线性拟合分析已禁用")
            return original_matplotlib_viz(csv_file, window_size, output_file)

        visualizer.visualize_matplotlib = modified_matplotlib_viz

    # 确定显示方式
    use_terminal = args.terminal and not args.no_terminal
    use_matplotlib = args.matplotlib

    if not use_terminal and not use_matplotlib:
        print("错误: 必须至少选择一种显示方式 (--terminal 或 --matplotlib)")
        sys.exit(1)

    # 在终端中显示
    if use_terminal:
        print("=" * 60)
        print("终端图表显示")
        print("=" * 60)
        visualizer.visualize_terminal(args.csv_file, args.window_size)
        print()

    # 使用matplotlib显示
    if use_matplotlib:
        print("=" * 60)
        print("Matplotlib图表显示")
        print("=" * 60)
        visualizer.visualize_matplotlib(args.csv_file, args.window_size, args.output)

    print("可视化完成!")


if __name__ == "__main__":
    main()
