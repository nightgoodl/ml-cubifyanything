#!/usr/bin/env python3
"""
分布式启动脚本
用于启动多个CA-1M处理worker
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path

def create_worker_script(worker_id, total_workers, base_args, output_dir):
    """为每个worker创建启动脚本"""
    script_name = f"worker_{worker_id}.sh"
    script_path = os.path.join(output_dir, script_name)
    
    # 构建命令
    cmd_parts = [
        "python", 
        os.path.abspath("demo_multithread_ca1m.py"),
        f"--worker-id {worker_id}",
        f"--total-workers {total_workers}"
    ]
    
    # 添加基础参数
    for arg, value in base_args.items():
        if value is True:
            cmd_parts.append(f"--{arg}")
        elif value is not False and value is not None:
            cmd_parts.append(f"--{arg} {value}")
    
    command = " ".join(cmd_parts)
    
    # 创建脚本内容
    script_content = f"""#!/bin/bash
# Worker {worker_id} 启动脚本
# 自动生成于 {time.strftime('%Y-%m-%d %H:%M:%S')}

echo "🚀 启动Worker {worker_id}/{total_workers}"
echo "📝 命令: {command}"
echo "⏰ 开始时间: $(date)"
echo ""

export OMP_NUM_THREADS=1  # 禁用OpenMP避免线程冲突

# 运行命令
{command} 2>&1 | tee worker_{worker_id}.log

echo ""
echo "⏰ 结束时间: $(date)"
echo "✅ Worker {worker_id} 完成"
"""
    
    # 写入脚本文件
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # 设置执行权限
    os.chmod(script_path, 0o755)
    
    return script_path

def main():
    parser = argparse.ArgumentParser(description="CA-1M分布式处理启动器")
    
    # 分布式参数
    parser.add_argument("--total-workers", type=int, required=True,
                       help="总worker数量")
    parser.add_argument("--start-worker", type=int, default=0,
                       help="起始worker ID")
    parser.add_argument("--end-worker", type=int, default=None,
                       help="结束worker ID (不包含)")
    
    # 执行模式
    parser.add_argument("--mode", choices=["scripts", "local", "tmux"], default="scripts",
                       help="执行模式: scripts(生成脚本), local(本地执行), tmux(tmux会话)")
    
    # 基础处理参数
    parser.add_argument("--data-dir", default="/baai-cwm-vepfs/cwm/chongjie.ye/data/CA-1M/data",
                       help="数据目录")
    parser.add_argument("--output-dir", default="/baai-cwm-vepfs/cwm/chongjie.ye/data/CA-1M_output",
                       help="输出目录")
    parser.add_argument("--split", choices=["train", "val"], default="train",
                       help="数据集划分")
    parser.add_argument("--voxel-size", type=float, default=0.004,
                       help="体素下采样尺寸")
    parser.add_argument("--disable-downsampling", action="store_true",
                       help="禁用体素下采样")
    parser.add_argument("--compute-workers", type=int, default=4,
                       help="计算线程数")
    parser.add_argument("--io-workers", type=int, default=2,
                       help="I/O线程数")
    parser.add_argument("--scene-workers", type=int, default=2,
                       help="场景处理线程数")
    parser.add_argument("--max-scenes", type=int, default=None,
                       help="最大处理场景数")
    
    args = parser.parse_args()
    
    # 计算worker范围
    if args.end_worker is None:
        args.end_worker = args.total_workers
    
    worker_ids = list(range(args.start_worker, min(args.end_worker, args.total_workers)))
    
    if not worker_ids:
        print("❌ 没有要启动的worker")
        sys.exit(1)
    
    print("="*80)
    print("🚀 CA-1M分布式处理启动器")
    print("="*80)
    print(f"📊 总workers: {args.total_workers}")
    print(f"🎯 启动范围: {args.start_worker} - {args.end_worker-1}")
    print(f"📝 实际启动: {worker_ids}")
    print(f"🎛️  执行模式: {args.mode}")
    print(f"📁 数据目录: {args.data_dir}")
    print(f"📤 输出目录: {args.output_dir}")
    print(f"📊 处理划分: {args.split}")
    print("="*80)
    
    # 创建脚本输出目录
    scripts_dir = "distributed_scripts"
    os.makedirs(scripts_dir, exist_ok=True)
    
    # 准备基础参数
    base_args = {
        "data-dir": args.data_dir,
        "output-dir": args.output_dir,
        "split": args.split,
        "voxel-size": args.voxel_size,
        "disable-downsampling": args.disable_downsampling,
        "compute-workers": args.compute_workers,
        "io-workers": args.io_workers,
        "scene-workers": args.scene_workers,
        "max-scenes": args.max_scenes,
    }
    
    # 创建脚本
    script_paths = []
    for worker_id in worker_ids:
        script_path = create_worker_script(worker_id, args.total_workers, base_args, scripts_dir)
        script_paths.append(script_path)
        print(f"📝 创建脚本: {script_path}")
    
    # 创建主启动脚本
    main_script_path = os.path.join(scripts_dir, "start_all_workers.sh")
    with open(main_script_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# 启动所有workers\n")
        f.write(f"# 生成于 {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for i, script_path in enumerate(script_paths):
            f.write(f"echo \"启动Worker {worker_ids[i]}...\"\n")
            f.write(f"bash {os.path.basename(script_path)} &\n")
            f.write(f"sleep 2\n\n")
        
        f.write("echo \"所有workers已启动，等待完成...\"\n")
        f.write("wait\n")
        f.write("echo \"所有workers已完成\"\n")
    
    os.chmod(main_script_path, 0o755)
    print(f"📝 创建主脚本: {main_script_path}")
    
    # 根据模式执行
    if args.mode == "scripts":
        print(f"\n✅ 脚本生成完成！")
        print(f"📁 脚本目录: {scripts_dir}")
        print(f"🚀 启动所有workers: bash {main_script_path}")
        print(f"🔍 查看日志: tail -f {scripts_dir}/worker_*.log")
        
    elif args.mode == "local":
        print(f"\n🚀 本地启动所有workers...")
        subprocess.run(["bash", main_script_path])
        
    elif args.mode == "tmux":
        print(f"\n🚀 使用tmux启动workers...")
        # 创建tmux会话
        session_name = f"ca1m_workers_{int(time.time())}"
        subprocess.run(["tmux", "new-session", "-d", "-s", session_name])
        
        for i, (worker_id, script_path) in enumerate(zip(worker_ids, script_paths)):
            window_name = f"worker_{worker_id}"
            if i == 0:
                subprocess.run(["tmux", "rename-window", "-t", session_name, window_name])
            else:
                subprocess.run(["tmux", "new-window", "-t", session_name, "-n", window_name])
            
            subprocess.run(["tmux", "send-keys", "-t", f"{session_name}:{window_name}", 
                          f"bash {script_path}", "Enter"])
        
        print(f"📺 Tmux会话: {session_name}")
        print(f"🔗 连接命令: tmux attach -t {session_name}")
        print(f"📋 查看窗口: tmux list-windows -t {session_name}")

if __name__ == "__main__":
    main()
