#!/usr/bin/env python3
"""
多线程处理进度监控脚本
实时监控处理进度和系统资源使用情况
"""

import os
import sys
import time
import json
import glob
import psutil
import threading
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

class ProgressMonitor:
    """进度监控类"""
    
    def __init__(self, output_root, data_root=None):
        self.output_root = output_root
        self.data_root = data_root
        self.start_time = time.time()
        self.last_update = time.time()
        self.monitoring = True
        
    def discover_expected_scenes(self):
        """发现预期处理的场景数量"""
        if not self.data_root or not os.path.exists(self.data_root):
            return 0
        
        scene_count = 0
        for split in ['train']:
            split_path = os.path.join(self.data_root, split)
            if os.path.exists(split_path):
                tar_files = glob.glob(os.path.join(split_path, "ca1m-*.tar"))
                scene_count += len(tar_files)
        
        if scene_count == 0:
            tar_files = glob.glob(os.path.join(self.data_root, "ca1m-*.tar"))
            scene_count = len(tar_files)
        
        return scene_count
    
    def scan_completed_scenes(self):
        """扫描已完成的场景"""
        if not os.path.exists(self.output_root):
            return {}
        
        completed_scenes = {}
        
        for split in ['train', 'unknown']:
            split_path = os.path.join(self.output_root, split)
            if os.path.exists(split_path):
                for scene_dir in os.listdir(split_path):
                    scene_path = os.path.join(split_path, scene_dir)
                    if os.path.isdir(scene_path):
                        scene_info = self.analyze_scene_completion(scene_path)
                        if scene_info:
                            completed_scenes[f"{split}/{scene_dir}"] = scene_info
        
        return completed_scenes
    
    def analyze_scene_completion(self, scene_path):
        """分析单个场景的完成情况"""
        scene_info = {
            'completed': False,
            'nocs_count': 0,
            'instance_count': 0,
            'has_bbox_info': False,
            'has_predictions': False,
            'last_modified': None
        }
        
        # 检查NOCS图像
        nocs_dir = os.path.join(scene_path, "nocs_images")
        if os.path.exists(nocs_dir):
            nocs_files = glob.glob(os.path.join(nocs_dir, "*.png"))
            scene_info['nocs_count'] = len(nocs_files)
        
        # 检查实例对象
        objects_dir = os.path.join(scene_path, "objects")
        if os.path.exists(objects_dir):
            instance_dirs = [d for d in os.listdir(objects_dir) 
                           if os.path.isdir(os.path.join(objects_dir, d))]
            scene_info['instance_count'] = len(instance_dirs)
        
        # 检查场景信息文件
        bbox_file = os.path.join(scene_path, "scene_bbox_info.json")
        scene_info['has_bbox_info'] = os.path.exists(bbox_file)
        
        predictions_file = os.path.join(scene_path, "predictions.json")
        scene_info['has_predictions'] = os.path.exists(predictions_file)
        
        # 获取最后修改时间
        try:
            mtime = os.path.getmtime(scene_path)
            scene_info['last_modified'] = datetime.fromtimestamp(mtime)
        except:
            pass
        
        # 判断是否完成（至少有NOCS图或实例数据）
        scene_info['completed'] = (scene_info['nocs_count'] > 0 or 
                                 scene_info['instance_count'] > 0)
        
        return scene_info if scene_info['completed'] else None
    
    def get_system_stats(self):
        """获取系统资源统计"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'load_avg': os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
        }
    
    def format_time_delta(self, seconds):
        """格式化时间差"""
        if seconds < 60:
            return f"{seconds:.0f}秒"
        elif seconds < 3600:
            return f"{seconds/60:.1f}分钟"
        else:
            return f"{seconds/3600:.1f}小时"
    
    def estimate_remaining_time(self, completed_count, total_count, elapsed_time):
        """估算剩余时间"""
        if completed_count == 0:
            return "未知"
        
        avg_time_per_scene = elapsed_time / completed_count
        remaining_scenes = total_count - completed_count
        remaining_time = avg_time_per_scene * remaining_scenes
        
        return self.format_time_delta(remaining_time)
    
    def print_progress_report(self):
        """打印进度报告"""
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # 清屏
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("🔄 CA-1M多线程处理进度监控")
        print("=" * 80)
        print(f"📁 输出目录: {self.output_root}")
        print(f"⏰ 开始时间: {datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  运行时间: {self.format_time_delta(elapsed_time)}")
        print()
        
        # 扫描完成情况
        completed_scenes = self.scan_completed_scenes()
        expected_count = self.discover_expected_scenes()
        completed_count = len(completed_scenes)
        
        # 进度统计
        print("📊 处理进度")
        print("-" * 40)
        if expected_count > 0:
            progress_percent = (completed_count / expected_count) * 100
            print(f"完成场景: {completed_count}/{expected_count} ({progress_percent:.1f}%)")
            
            # 进度条
            bar_length = 50
            filled_length = int(bar_length * completed_count / expected_count)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            print(f"进度条: [{bar}]")
            
            # 估算剩余时间
            if completed_count > 0:
                remaining_time = self.estimate_remaining_time(completed_count, expected_count, elapsed_time)
                print(f"预计剩余: {remaining_time}")
        else:
            print(f"已完成场景: {completed_count}")
        
        print()
        
        # 按split统计
        split_stats = defaultdict(lambda: {'count': 0, 'nocs': 0, 'instances': 0})
        for scene_path, scene_info in completed_scenes.items():
            split = scene_path.split('/')[0]
            split_stats[split]['count'] += 1
            split_stats[split]['nocs'] += scene_info['nocs_count']
            split_stats[split]['instances'] += scene_info['instance_count']
        
        if split_stats:
            print("📂 按数据集划分")
            print("-" * 40)
            for split, stats in split_stats.items():
                print(f"{split:8}: {stats['count']:3d} 场景, {stats['nocs']:5d} NOCS图, {stats['instances']:4d} 实例")
        
        print()
        
        # 最近完成的场景
        if completed_scenes:
            recent_scenes = sorted(completed_scenes.items(), 
                                 key=lambda x: x[1]['last_modified'] or datetime.min, 
                                 reverse=True)[:5]
            
            print("🕐 最近完成的场景")
            print("-" * 40)
            for scene_path, scene_info in recent_scenes:
                scene_name = scene_path.split('/')[-1]
                last_mod = scene_info['last_modified']
                if last_mod:
                    time_str = last_mod.strftime('%H:%M:%S')
                    print(f"{time_str} | {scene_name[:20]:20} | NOCS:{scene_info['nocs_count']:3d} 实例:{scene_info['instance_count']:2d}")
        
        print()
        
        # 系统资源
        try:
            sys_stats = self.get_system_stats()
            print("💻 系统资源")
            print("-" * 40)
            print(f"CPU使用率: {sys_stats['cpu_percent']:5.1f}%")
            print(f"内存使用: {sys_stats['memory_percent']:5.1f}%")
            print(f"磁盘使用: {sys_stats['disk_usage']:5.1f}%")
            if hasattr(os, 'getloadavg'):
                load1, load5, load15 = sys_stats['load_avg']
                print(f"系统负载: {load1:.2f} {load5:.2f} {load15:.2f}")
        except:
            print("💻 系统资源信息获取失败")
        
        print()
        print("按 Ctrl+C 退出监控")
        print("=" * 80)
    
    def monitor_loop(self, update_interval=10):
        """监控循环"""
        print("🚀 开始监控多线程处理进度...")
        print(f"📊 更新间隔: {update_interval}秒")
        
        try:
            while self.monitoring:
                self.print_progress_report()
                time.sleep(update_interval)
        except KeyboardInterrupt:
            print("\n👋 监控已停止")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="监控CA-1M多线程处理进度")
    parser.add_argument("--output-root", 
                       default="/baai-cwm-vepfs/cwm/chongjie.ye/data/CA-1M_output",
                       help="输出根目录")
    parser.add_argument("--data-root", 
                       default="/baai-cwm-vepfs/cwm/chongjie.ye/data/CA-1M/data",
                       help="数据根目录")
    parser.add_argument("--interval", type=int, default=10,
                       help="更新间隔(秒)")
    parser.add_argument("--once", action="store_true",
                       help="只显示一次，不持续监控")
    
    args = parser.parse_args()
    
    monitor = ProgressMonitor(args.output_root, args.data_root)
    
    if args.once:
        monitor.print_progress_report()
    else:
        monitor.monitor_loop(args.interval)

if __name__ == "__main__":
    main()
