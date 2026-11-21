# gpu-async-compute

## 题面

把粒子计算挪到异步队列后，偶尔读到上一帧数据、GPU 还卡死：依赖只按提交顺序执行，后置命令先于其前置命令运行；栅栏只在同队列内生效，跨队列不等待；分批按固定大小切分，忽略依赖关系；依赖成环时检测不报错并直接死锁；信号量等待计数可为负导致状态错乱；图形/计算重叠比例按命令数而不是按 tick 时间计算；未知依赖被静默忽略导致命令漏跑；超过并发上限不校验导致命令同时下发；批量调度共用同一个队列，批次之间相互串扰；concurrency=0 不校验，调度陷入死循环。async/scheduler.py 是核心，tests/test_scheduler.py 15个红。要修的是：上述十类异步调度缺陷。

## 运行方式与初始红

在仓库根目录运行：

```powershell
python -m pytest tests -q
```

初始状态 8 个公开函数全部抛出 `NotImplementedError`，预期输出尾部为：

```text
FFFFFFFFFFFFFFF                                                          [100%]
15 failed in 0.07s
```

即恰好 `15 failed`，且没有 collection error、import error 或 syntax error；耗时随机器不同而略有差异。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 图形渲染 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 约束

`tests/test_scheduler.py` 不可修改，15 条测试是验收标准。实现只使用 Python 标准库（`heapq`、`collections`）与 pytest，离线、无随机、不调用 `sleep`，所有时间推进都通过显式 tick 完成。
