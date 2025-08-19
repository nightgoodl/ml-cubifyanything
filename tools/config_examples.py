#!/usr/bin/env python3
"""
CA-1M处理工具配置示例
根据不同的硬件配置提供推荐参数
"""

# 配置模板
CONFIGS = {
    "high_performance": {
        "description": "高性能服务器配置 (32+ CPU cores, 64+ GB RAM)",
        "scene_workers": 8,
        "max_workers": 8,
        "voxel_size": 0.004,
        "disable_downsampling": False,
        "expected_memory_gb": 64,
        "expected_processing_time_per_scene_sec": 2.0
    },
    
    "medium_performance": {
        "description": "中等性能工作站配置 (16+ CPU cores, 32+ GB RAM)",
        "scene_workers": 4,
        "max_workers": 6,
        "voxel_size": 0.004,
        "disable_downsampling": False,
        "expected_memory_gb": 32,
        "expected_processing_time_per_scene_sec": 4.0
    },
    
    "low_performance": {
        "description": "低性能配置 (8+ CPU cores, 16+ GB RAM)",
        "scene_workers": 2,
        "max_workers": 4,
        "voxel_size": 0.006,  # 更粗的体素以节省内存
        "disable_downsampling": False,
        "expected_memory_gb": 16,
        "expected_processing_time_per_scene_sec": 8.0
    },
    
    "memory_limited": {
        "description": "内存受限配置 (任意CPU, 8-16 GB RAM)",
        "scene_workers": 1,
        "max_workers": 2,
        "voxel_size": 0.008,  # 更粗的体素
        "disable_downsampling": False,
        "expected_memory_gb": 8,
        "expected_processing_time_per_scene_sec": 15.0
    },
    
    "testing": {
        "description": "测试配置 (快速验证)",
        "scene_workers": 1,
        "max_workers": 2,
        "voxel_size": 0.01,   # 粗体素，快速处理
        "disable_downsampling": False,
        "max_scenes": 5,      # 仅处理5个场景
        "expected_memory_gb": 4,
        "expected_processing_time_per_scene_sec": 3.0
    }
}

def print_config_recommendations():
    """打印所有配置建议"""
    print("CA-1M处理工具配置建议")
    print("=" * 60)
    
    for config_name, config in CONFIGS.items():
        print(f"\n🔧 {config_name.upper()} 配置")
        print(f"   描述: {config['description']}")
        print(f"   场景处理线程数: {config['scene_workers']}")
        print(f"   单场景线程数: {config['max_workers']}")
        print(f"   体素下采样尺寸: {config['voxel_size']}m")
        print(f"   预期内存需求: {config['expected_memory_gb']}GB")
        print(f"   预期每场景处理时间: {config['expected_processing_time_per_scene_sec']}秒")
        
        # 生成命令行示例
        cmd_parts = [
            "python tools/demo_multithread_ca1m.py",
            "--split train",
            f"--scene-workers {config['scene_workers']}",
            f"--max-workers {config['max_workers']}",
            f"--voxel-size {config['voxel_size']}"
        ]
        
        if config.get('disable_downsampling', False):
            cmd_parts.append("--disable-downsampling")
        
        if 'max_scenes' in config:
            cmd_parts.append(f"--max-scenes {config['max_scenes']}")
        
        print(f"   命令行示例:")
        print(f"   {' '.join(cmd_parts)}")

def get_config(config_name):
    """获取指定配置"""
    return CONFIGS.get(config_name)

def estimate_processing_time(config_name, num_scenes):
    """估算处理时间"""
    config = get_config(config_name)
    if not config:
        return None
    
    time_per_scene = config['expected_processing_time_per_scene_sec']
    total_time_sec = (num_scenes * time_per_scene) / config['scene_workers']
    
    hours = int(total_time_sec // 3600)
    minutes = int((total_time_sec % 3600) // 60)
    seconds = int(total_time_sec % 60)
    
    return {
        'total_seconds': total_time_sec,
        'hours': hours,
        'minutes': minutes,
        'seconds': seconds,
        'formatted': f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CA-1M处理配置助手")
    parser.add_argument("--show-configs", action="store_true", help="显示所有配置建议")
    parser.add_argument("--config", choices=CONFIGS.keys(), help="显示特定配置")
    parser.add_argument("--estimate-time", type=int, metavar="NUM_SCENES", 
                       help="估算处理指定场景数的时间")
    parser.add_argument("--config-for-estimation", choices=CONFIGS.keys(), 
                       default="medium_performance", help="用于时间估算的配置")
    
    args = parser.parse_args()
    
    if args.show_configs:
        print_config_recommendations()
    
    elif args.config:
        config = get_config(args.config)
        print(f"🔧 {args.config.upper()} 配置详情:")
        for key, value in config.items():
            print(f"   {key}: {value}")
    
    elif args.estimate_time:
        time_info = estimate_processing_time(args.config_for_estimation, args.estimate_time)
        if time_info:
            print(f"📊 处理 {args.estimate_time} 个场景的时间估算 ({args.config_for_estimation} 配置):")
            print(f"   预计总时间: {time_info['formatted']} ({time_info['total_seconds']:.0f}秒)")
            print(f"   配置: {CONFIGS[args.config_for_estimation]['description']}")
    
    else:
        print("使用 --help 查看可用选项")
        print("\n快速示例:")
        print("  python config_examples.py --show-configs")
        print("  python config_examples.py --config medium_performance")
        print("  python config_examples.py --estimate-time 100 --config-for-estimation high_performance")
