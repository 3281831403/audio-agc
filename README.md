# audio-agc

## 题面

给会议录音加自动增益后，句子开头总被压一下、块边界还有台阶：`smooth` 对攻击与释放共用一个时间常数，把 attack 用在 release 上，增益回落过快并产生泵浦；`process_block` 返回新状态时把 `prev_gain` 记成本块输入增益而不是本块实际输出增益，下一块从错误起点继续，块边界出现台阶；`noise_floor` 用均值代替分位估计，语音块把底噪抬高，后续增益整体偏小；`target_gain` 的 `max_gain` 只钳制上限、不钳制 0 dB，负增益会在输出端造成相位反转；`rms_db` 对全零块返回 0 dB 而不是 `-inf`，静音段被当作满电平，增益被压死；`apply_blocks` 用 `len(samples)//block` 而不是向上取整，尾部不足一块的样本被丢弃；`smooth` 的时间常数按块数而不是按 ms 换算，改变 block 大小时响应曲线跟着改变；`validate` 只检查数值范围，不检查“目标 − 底噪 ≈ 当前增益”的自洽关系，脏状态未被报告；`process_block` 对块内样本逐点更新状态而不是块级更新，块内增益随时间漂移；`agc_batch` 浅拷贝复用同一 state 对象，第二条轨道从第一条的最终状态开始。坑在状态机上：块内/块间、攻击/释放、底噪/增益四处互相耦合。agc/control.py 是核心，tests/test_control.py 15个红。要修的是：把上述 10 处状态机与不变量缺陷逐条修正，并保持全部 15 个测试通过。

## 运行方式与初始状态

要求 Python 3.10+，只使用标准库和 pytest，离线运行：

```powershell
python -m pytest tests -q
```

初始仓库中 `agc/control.py` 的 8 个函数全部为 `raise NotImplementedError` 桩，因此测试收集和导入正常，但 15 个测试会各自在调用桩时失败。预期输出尾部为：

```text
15 failed in <time>s
```

不存在 collection error、import error 或 syntax error。`tests/test_control.py` 是评测契约，**不可修改**；实现只能改 `agc/control.py`（以及必要时新增模块内私有辅助函数）。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 音视频处理 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 约定摘要

- 样本为 `[-1.0, 1.0]` 归一化幅度；电平用 dBFS：`20 * log10(rms)`，数字静音为 `-inf`。
- 控制量（目标增益、平滑后增益、底噪、`prev_gain`）均以 dB 表示；输出乘数为 `10 ** (gain_db / 20)`。
- 时间参数以 ms 计；块大小以样本数计；块内保持恒定增益。
- 浮点比较容差：线性幅度与 dB 控制量默认 `1e-9`，跨 block 大小的稳态增益容差 `0.01 dB`。
- 状态为 `{"prev_gain": float, "floor_db": float, "levels": list[float]}`；`prev_gain` 必须是上一块实际施加到输出上的 dB 增益。
- 静音块必须保持状态（增益、底噪和历史不变），只输出等长零样本。

## 文件清单

```text
audio-agc/
├── README.md
├── conftest.py
├── .gitignore
├── agc/
│   ├── __init__.py
│   └── control.py
└── tests/
    └── test_control.py
```

`tests/` 不可修改。评测只通过 `python -m pytest tests -q` 观察 15 个红转绿。
