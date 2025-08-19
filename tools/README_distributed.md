# CA-1M数据集分布式多线程处理工具

## 🚀 主要改进

### 1. 任务分离架构
- **计算线程池**: 专门处理点云归一化、体素下采样等计算密集型任务
- **I/O线程池**: 专门处理文件读写操作
- **场景线程池**: 并行处理多个场景

### 2. 分布式处理支持
- 支持多机器并行处理大规模train数据集
- 自动分配场景给不同worker
- 独立的输出目录，避免冲突

### 3. 内存优化
- 避免大量数据在线程间传递
- 计算结果直接传递给I/O线程
- 更好的资源管理

## 📋 使用方法

### 单机处理

```bash
# 处理val数据集（单机）
python demo_multithread_ca1m.py \
    --split val \
    --compute-workers 6 \
    --io-workers 2 \
    --scene-workers 2

# 处理train数据集（单机，小规模测试）
python demo_multithread_ca1m.py \
    --split train \
    --max-scenes 10 \
    --compute-workers 8 \
    --io-workers 2 \
    --scene-workers 2
```

### 分布式处理（推荐用于train数据集）

#### 方式1: 手动启动多个worker

```bash
# 机器1 - Worker 0
python demo_multithread_ca1m.py \
    --split train \
    --worker-id 0 \
    --total-workers 4 \
    --compute-workers 8 \
    --io-workers 2 \
    --scene-workers 2

# 机器2 - Worker 1  
python demo_multithread_ca1m.py \
    --split train \
    --worker-id 1 \
    --total-workers 4 \
    --compute-workers 8 \
    --io-workers 2 \
    --scene-workers 2

# 机器3 - Worker 2
python demo_multithread_ca1m.py \
    --split train \
    --worker-id 2 \
    --total-workers 4 \
    --compute-workers 8 \
    --io-workers 2 \
    --scene-workers 2

# 机器4 - Worker 3
python demo_multithread_ca1m.py \
    --split train \
    --worker-id 3 \
    --total-workers 4 \
    --compute-workers 8 \
    --io-workers 2 \
    --scene-workers 2
```

#### 方式2: 使用分布式启动器

```bash
# 生成所有worker的启动脚本
python start_distributed_workers.py \
    --total-workers 4 \
    --split train \
    --compute-workers 8 \
    --io-workers 2 \
    --scene-workers 2

# 在每台机器上运行对应的脚本
# 机器1: bash distributed_scripts/worker_0.sh
# 机器2: bash distributed_scripts/worker_1.sh  
# 机器3: bash distributed_scripts/worker_2.sh
# 机器4: bash distributed_scripts/worker_3.sh
```

### 合并分布式结果

```bash
# 所有worker完成后，合并结果
python merge_worker_results.py \
    --base-output-dir /path/to/output \
    --split train \
    --total-workers 4 \
    --verify
```

## ⚙️ 参数说明

### 核心处理参数
- `--split`: 数据集划分 (train/val)
- `--data-dir`: CA-1M数据集根目录
- `--output-dir`: 输出目录

### 分布式参数
- `--worker-id`: 当前worker ID (0开始)
- `--total-workers`: 总worker数量
- 两个参数必须同时提供或同时省略

### 线程配置参数
- `--compute-workers`: 计算线程数 (建议6-8)
- `--io-workers`: I/O线程数 (建议2-4)  
- `--scene-workers`: 场景处理线程数 (建议2-4)

### 处理选项
- `--voxel-size`: 体素下采样尺寸 (默认0.004)
- `--disable-downsampling`: 禁用体素下采样
- `--max-scenes`: 最大处理场景数 (测试用)

## 📊 性能调优建议

### 线程数配置

```bash
# 计算密集型机器 (高CPU核数)
--compute-workers 12 --io-workers 2 --scene-workers 2

# I/O密集型机器 (快速存储)  
--compute-workers 6 --io-workers 4 --scene-workers 3

# 平衡配置 (推荐)
--compute-workers 8 --io-workers 2 --scene-workers 2
```

### GPU配置
脚本会自动设置`CUDA_VISIBLE_DEVICES`循环使用GPU：
- Worker 0: GPU 0
- Worker 1: GPU 1  
- Worker 8: GPU 0 (循环)

### 内存考虑
- 每个场景通常需要2-4GB内存
- `scene-workers`数量 × 4GB ≈ 总内存需求
- 建议预留8-16GB系统内存

## 📁 输出结构

```
output_dir/
├── train/                    # 单机模式
│   ├── ca1m-scene1/
│   │   ├── nocs_images/
│   │   ├── objects/
│   │   └── scene_bbox_info.json
│   └── ca1m-scene2/
│       └── ...
├── train/                    # 分布式模式
│   ├── worker_0/
│   │   ├── ca1m-scene1/
│   │   └── ca1m-scene2/
│   ├── worker_1/
│   │   ├── ca1m-scene3/
│   │   └── ca1m-scene4/
│   └── ...
└── train_merged/             # 合并后结果
    ├── ca1m-scene1/
    ├── ca1m-scene2/
    ├── ca1m-scene3/
    ├── ca1m-scene4/
    └── merge_report.json
```

## 🔍 监控和日志

### 实时监控
```bash
# 查看worker日志
tail -f distributed_scripts/worker_*.log

# 监控GPU使用
watch nvidia-smi

# 监控磁盘I/O
iotop -a
```

### 进度估算
每个worker会显示：
- 当前处理场景数/总分配场景数
- 每场景平均处理时间
- 预估剩余时间

## 🛠️ 故障排除

### 常见问题

1. **内存不足**
   ```bash
   # 减少并行度
   --scene-workers 1 --compute-workers 4
   ```

2. **磁盘I/O瓶颈**
   ```bash
   # 增加I/O线程，减少场景并行
   --io-workers 4 --scene-workers 1
   ```

3. **GPU显存不足**
   ```bash
   # 限制每个worker使用的GPU
   export CUDA_VISIBLE_DEVICES=0  # 只使用GPU 0
   ```

4. **Worker失败重启**
   ```bash
   # 查看失败的场景范围
   python demo_multithread_ca1m.py --split train --worker-id 2 --total-workers 4 --max-scenes 5
   ```

### 断点续传
每个场景独立处理，可以通过检查输出目录来跳过已完成的场景：

```bash
# 手动清理不完整的场景目录，重新运行worker
```

## 📈 预期性能

基于8核CPU + SSD存储的机器：

- **单个场景**: 30-60秒 (取决于帧数和实例数)
- **Val数据集**: 2-4小时 (约200个场景)
- **Train数据集**: 
  - 单机: 数天到数周
  - 4机器分布式: 6-12小时
  - 8机器分布式: 3-6小时

## 🎯 最佳实践

1. **数据集测试**: 先用`--max-scenes 5`测试流程
2. **分阶段处理**: 可以先处理一部分worker，确认无误后再启动其他
3. **资源监控**: 密切监控CPU、内存、磁盘使用情况
4. **结果验证**: 使用`--verify`参数检查合并结果完整性
5. **备份策略**: 定期备份重要的中间结果

## 📞 技术支持

如遇到问题，请检查：
1. 日志文件中的错误信息
2. 系统资源使用情况  
3. 数据目录权限
4. Python环境和依赖包
