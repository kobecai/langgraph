# LangGraph Memory 机制深度解析

> 基于 LangGraph 源码（`libs/checkpoint/` 和 `libs/langgraph/`）的深度分析笔记。

---

## 目录

1. [LangGraph 的 Memory 抽象层次](#1-langgraph-的-memory-抽象层次)
2. [BaseCheckpointSaver 详解](#2-basecheckpointsaver-详解)
3. [BaseStore 详解](#3-basestore-详解)
4. [用 Mem0 替换 LangGraph 原生 Memory](#4-用-mem0-替换-langgraph-原生-memory)
5. [Checkpointer 和 Store 的调用时机（源码级别）](#5-checkpointer-和-store-的调用时机源码级别)

---

## 1. LangGraph 的 Memory 抽象层次

LangGraph 把"记忆"拆成**两个互相独立的抽象**，职责完全不同：

```python
graph.compile(
    checkpointer=...,   # BaseCheckpointSaver 实例 —— 控制 "当前对话状态"
    store=...,          # BaseStore 实例       —— 控制 "跨对话长期记忆"
)
```

| 抽象 | 源码位置 | 职责 | 数据范围 |
|------|----------|------|----------|
| `BaseCheckpointSaver` | `libs/checkpoint/langgraph/checkpoint/base/__init__.py` | 保存图每一步的完整状态快照，支持中断/恢复、时间旅行 | 单个 `thread_id` 内 |
| `BaseStore` | `libs/checkpoint/langgraph/store/base/__init__.py` | 跨线程共享的持久化键值存储，支持命名空间和向量语义搜索 | 跨 `thread_id` |

---

## 2. BaseCheckpointSaver 详解

### 核心数据结构

```python
class Checkpoint(TypedDict):
    v: int                           # checkpoint 格式版本，目前为 1
    id: str                          # 唯一且单调递增，可用于排序
    ts: str                          # ISO 8601 时间戳
    channel_values: dict[str, Any]   # 各 channel 在此时刻的反序列化快照值
    channel_versions: ChannelVersions # channel 名 -> 单调递增版本号
    versions_seen: dict[str, ChannelVersions]  # 各节点已看到的 channel 版本
    updated_channels: list[str] | None  # 此 checkpoint 中被更新的 channels

class CheckpointTuple(NamedTuple):
    config: RunnableConfig
    checkpoint: Checkpoint
    metadata: CheckpointMetadata
    parent_config: RunnableConfig | None = None
    pending_writes: list[PendingWrite] | None = None

class CheckpointMetadata(TypedDict, total=False):
    source: Literal["input", "loop", "update", "fork"]
    step: int        # -1 表示第一个 input checkpoint，0 表示第一个 loop checkpoint
    parents: dict[str, str]  # namespace -> checkpoint_id 的父映射
    run_id: str
```

### 必须实现的接口

```python
class BaseCheckpointSaver(Generic[V]):
    serde: SerializerProtocol = JsonPlusSerializer()  # 序列化器，默认 JsonPlus

    # ── 同步接口 ──────────────────────────────────────────────────────────────
    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """按 config（thread_id + 可选 checkpoint_id）取 checkpoint。"""
        raise NotImplementedError

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """列举满足条件的历史 checkpoints，用于时间旅行。"""
        raise NotImplementedError

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """写入新的完整 checkpoint，返回更新后的 config（含新 checkpoint_id）。"""
        raise NotImplementedError

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """写入某个节点任务的中间 writes（用于从 interrupt 恢复）。"""
        raise NotImplementedError

    # ── 异步接口（与同步接口一一对应）──────────────────────────────────────────
    async def aget_tuple(self, config) -> CheckpointTuple | None: ...
    async def alist(self, config, *, filter, before, limit) -> AsyncIterator[CheckpointTuple]: ...
    async def aput(self, config, checkpoint, metadata, new_versions) -> RunnableConfig: ...
    async def aput_writes(self, config, writes, task_id, task_path) -> None: ...

    # ── 可选扩展接口 ───────────────────────────────────────────────────────────
    def delete_thread(self, thread_id: str) -> None: ...
    def delete_for_runs(self, run_ids: Sequence[str]) -> None: ...
    def copy_thread(self, source_thread_id, target_thread_id) -> None: ...
    def prune(self, thread_ids, *, strategy="keep_latest") -> None: ...
```

### 使用方式

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.postgres import PostgresSaver

# 使用时通过 config 传入 thread_id
config = {"configurable": {"thread_id": "conversation-001"}}
graph.invoke(input, config)

# 时间旅行：传入 checkpoint_id 可恢复到任意历史状态
config = {"configurable": {"thread_id": "conversation-001", "checkpoint_id": "xxx"}}
```

---

## 3. BaseStore 详解

### 核心数据类型

```python
class Item:
    """存储的一条记录。"""
    value: dict[str, Any]          # 存储的数据，键值可过滤
    key: str                       # namespace 内的唯一标识
    namespace: tuple[str, ...]     # 层级路径，如 ("documents", "user123")
    created_at: datetime
    updated_at: datetime

class SearchItem(Item):
    score: float | None            # 语义搜索时的相似度分数
```

### 四种操作类型（Op）

```python
# 1. 精确获取
GetOp(namespace=("users", "profiles"), key="user123")

# 2. 写入/更新（value=None 表示删除）
PutOp(
    namespace=("docs", "user123"),
    key="report1",
    value={"content": "...", "type": "report"},
    index=["content"],   # 指定哪些字段做向量索引；False=不索引；None=用默认配置
    ttl=60.0,            # 分钟，过期时间（需 store 实现支持）
)

# 3. 语义/过滤搜索
SearchOp(
    namespace_prefix=("docs",),
    query="machine learning in healthcare",  # 语义搜索 query（可选）
    filter={"type": "report", "status": {"$gt": 3}},  # 支持 $eq/$ne/$gt/$gte/$lt/$lte
    limit=10,
    offset=0,
)

# 4. 列举命名空间
ListNamespacesOp(
    match_conditions=(
        MatchCondition(match_type="prefix", path=("users",)),
    ),
    max_depth=3,
    limit=100,
)
```

### 必须实现的接口

```python
class BaseStore(ABC):
    supports_ttl: bool = False
    ttl_config: TTLConfig | None = None

    @abstractmethod
    def batch(self, ops: Iterable[Op]) -> list[Result]:
        """同步批量执行操作，结果顺序与 ops 一一对应。"""

    @abstractmethod
    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """异步批量执行操作。"""
```

上层所有便利方法（`get` / `put` / `delete` / `search` / `list_namespaces` 及对应 async 版本）都只是对 `batch` / `abatch` 的封装，**只需实现这两个方法即可**。

### 命名空间设计规范

```python
# 常见命名空间约定：
("memories", user_id)        # 用户级别的记忆
("preferences", user_id)     # 用户偏好
("documents", user_id, "reports")  # 用户报告文档
("cache", "embeddings", "v1")      # 缓存
```

### 向量搜索配置

```python
from langgraph.store.memory import InMemoryStore
from langchain.embeddings import init_embeddings

store = InMemoryStore(
    index={
        "dims": 1536,                           # embedding 维度
        "embed": init_embeddings("openai:text-embedding-3-small"),
        "fields": ["text", "summary"],          # 哪些字段做 embedding（默认 ["$"] 即整个 JSON）
    }
)
```

### 节点中访问 Store 的两种方式

```python
from langgraph.store.base import BaseStore
from langgraph.runtime import Runtime

# 方式 A：直接注入 store 参数（推荐）
def my_node(state: State, *, store: BaseStore) -> dict:
    items = store.search(("memories", "user_123"), query="用户喜好")
    store.put(("memories", "user_123"), "pref_001", {"content": "喜欢 Python"})
    return {...}

# 方式 B：通过 Runtime 对象
def my_node(state: State, runtime: Runtime) -> dict:
    if runtime.store:
        items = runtime.store.search(...)
    return {...}
```

---

## 4. 用 Mem0 替换 LangGraph 原生 Memory

### 设计选择

Mem0 的 API 哲学是"输入对话 → LLM 自动提取结构化记忆 → 语义搜索召回"：

- **适合替换 `BaseStore`**（长期跨会话记忆）
- **不适合替换 `BaseCheckpointSaver`**（对话内状态快照是 LangGraph 的运行机制基础，不应替换）

### 方案一：将 Mem0 封装为 BaseStore（推荐）

原理：将 LangGraph 的 `namespace` 最后一段映射为 Mem0 的 `user_id`，通过 `batch()` 方法适配四种 Op。

```python
import asyncio
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

from langgraph.store.base import (
    BaseStore, Item, SearchItem,
    GetOp, PutOp, SearchOp, ListNamespacesOp,
    Op, Result,
)
from mem0 import MemoryClient


def _namespace_to_user_id(namespace: tuple[str, ...]) -> str:
    """约定：namespace 最后一段作为 Mem0 的 user_id。
    例如 ("memories", "user_123") -> user_id="user_123"
    """
    return namespace[-1] if namespace else "default"


def _mem0_result_to_item(
    m: dict[str, Any],
    namespace: tuple[str, ...],
    score: float | None = None,
) -> Item | SearchItem:
    """将 Mem0 返回的 memory dict 转换为 LangGraph Item。"""
    # key 优先从我们自己写入的 metadata 中取，否则用 Mem0 的 memory_id
    key = m.get("metadata", {}).get("_lg_key", m["id"])
    # 把 memory 文本和所有 metadata 合并为 value
    value = {"memory": m["memory"], **m.get("metadata", {})}
    now = datetime.now(tz=timezone.utc)
    created_at = datetime.fromisoformat(m["created_at"]) if m.get("created_at") else now
    updated_at = datetime.fromisoformat(m["updated_at"]) if m.get("updated_at") else now

    if score is not None:
        return SearchItem(
            namespace=namespace, key=key, value=value,
            created_at=created_at, updated_at=updated_at, score=score,
        )
    return Item(
        namespace=namespace, key=key, value=value,
        created_at=created_at, updated_at=updated_at,
    )


class Mem0Store(BaseStore):
    """将 Mem0 封装为 LangGraph BaseStore，用于跨线程的长期记忆。

    命名空间约定：
        namespace 最后一个元素作为 Mem0 的 user_id。
        例如：store.get(("memories", "user_123"), "key")
        等价于 mem0.get(memory_id)  # user_id="user_123"
    """

    def __init__(self, api_key: str):
        self.client = MemoryClient(api_key=api_key)

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        results: list[Result] = []
        for op in ops:

            # ── GetOp：精确取一条记忆 ──────────────────────────────────────────
            if isinstance(op, GetOp):
                try:
                    # op.key 就是 Mem0 的 memory_id（我们存时保存在 metadata._lg_key）
                    m = self.client.get(op.key)
                    results.append(_mem0_result_to_item(m, op.namespace))
                except Exception:
                    results.append(None)

            # ── PutOp：写入或删除记忆 ──────────────────────────────────────────
            elif isinstance(op, PutOp):
                user_id = _namespace_to_user_id(op.namespace)
                if op.value is None:
                    # value=None 表示删除，op.key 是 Mem0 memory_id
                    try:
                        self.client.delete(op.key)
                    except Exception:
                        pass
                else:
                    # 优先取 value["memory"] 或 value["text"] 作为记忆文本
                    text = (
                        op.value.get("memory")
                        or op.value.get("text")
                        or str(op.value)
                    )
                    # 将 LangGraph key 和 namespace 存入 metadata，以便 GetOp 恢复
                    metadata = {
                        "_lg_key": op.key,
                        "_namespace": list(op.namespace),
                    }
                    self.client.add(
                        messages=[{"role": "user", "content": text}],
                        user_id=user_id,
                        metadata=metadata,
                    )
                results.append(None)

            # ── SearchOp：语义/过滤搜索 ────────────────────────────────────────
            elif isinstance(op, SearchOp):
                user_id = _namespace_to_user_id(op.namespace_prefix)
                if op.query:
                    # 有 query，走 Mem0 的语义搜索（底层向量检索）
                    memories = self.client.search(
                        op.query, user_id=user_id, limit=op.limit
                    )
                    items = [
                        _mem0_result_to_item(
                            m, op.namespace_prefix, score=m.get("score")
                        )
                        for m in memories
                    ]
                else:
                    # 无 query，取全部再做分页（Mem0 无结构化过滤，需自行过滤）
                    all_memories = self.client.get_all(user_id=user_id)
                    items = [
                        _mem0_result_to_item(m, op.namespace_prefix)
                        for m in all_memories[op.offset : op.offset + op.limit]
                    ]
                results.append(items)

            # ── ListNamespacesOp：Mem0 无命名空间概念，返回空 ──────────────────
            elif isinstance(op, ListNamespacesOp):
                results.append([])

        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """Mem0 Python SDK 目前为同步，用线程池避免阻塞事件循环。"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.batch, list(ops))
```

#### 完整用法

```python
from langgraph.checkpoint.memory import MemorySaver

store = Mem0Store(api_key="your-mem0-api-key")
checkpointer = MemorySaver()   # 对话内状态仍用原生实现

graph = your_graph.compile(checkpointer=checkpointer, store=store)

# 在节点中通过 store 参数注入
def agent_node(state: State, *, store: BaseStore) -> dict:
    user_id = "user_123"
    # 语义检索相关记忆
    memories = store.search(
        ("memories", user_id),
        query=state["messages"][-1].content,
        limit=5,
    )
    context = "\n".join(f"- {m.value['memory']}" for m in memories)
    # 将记忆注入到 prompt 中...
    return {"context": context}

config = {"configurable": {"thread_id": "session-001"}}
graph.invoke({"messages": [{"role": "user", "content": "我喜欢 Python"}]}, config)
```

---

### 方案二：在节点中直接调用 Mem0（更灵活）

不替换 `BaseStore`，而是在特定节点里显式控制记忆的读写，更适合需要充分利用 Mem0 独特 API 的场景：

```python
from mem0 import MemoryClient

mem0 = MemoryClient(api_key="your-mem0-api-key")

def load_memories_node(state: State, config: RunnableConfig) -> dict:
    """对话开始时，从 Mem0 检索相关记忆，注入 system prompt。"""
    user_id = config["configurable"]["user_id"]
    query = state["messages"][-1].content
    memories = mem0.search(query, user_id=user_id, limit=5)
    memory_text = "\n".join(f"- {m['memory']}" for m in memories)
    system_msg = f"关于用户的历史记忆：\n{memory_text}"
    return {"system_memory": system_msg}

def save_memories_node(state: State, config: RunnableConfig) -> dict:
    """对话结束后，将本轮对话存入 Mem0，由 Mem0 的 LLM 自动提取结构化记忆。"""
    user_id = config["configurable"]["user_id"]
    messages = [
        {"role": m.type, "content": m.content}
        for m in state["messages"]
    ]
    mem0.add(messages, user_id=user_id)
    return {}

graph = (
    StateGraph(State)
    .add_node("load_memories", load_memories_node)
    .add_node("agent", agent_node)
    .add_node("save_memories", save_memories_node)
    .add_edge(START, "load_memories")
    .add_edge("load_memories", "agent")
    .add_edge("agent", "save_memories")
    .compile(checkpointer=MemorySaver())  # store 不传，不用原生 store
)
```

### 两种方案对比

| | 方案一：封装为 `BaseStore` | 方案二：节点直接调用 |
|---|---|---|
| 与图的耦合方式 | 原生集成，节点通过参数注入访问 | 显式，逻辑清晰 |
| 适合场景 | 多个节点都需要访问记忆 | 记忆只在特定节点使用 |
| Mem0 特性利用 | 受限（要适配 KV Op 接口） | 完整（可用所有 Mem0 API） |
| 维护难度 | 需维护适配层 | 简单直接 |

---

## 5. Checkpointer 和 Store 的调用时机（源码级别）

核心执行引擎在 `libs/langgraph/langgraph/pregel/_loop.py` 的 `PregelLoop` 类中。

### Checkpointer 的调用全流程

#### 调用 ① — `__enter__`：读取历史状态

**源码**：[`_loop.py` ~L1614](libs/langgraph/langgraph/pregel/_loop.py#L1614)

```python
def __enter__(self) -> Self:
    if not self.checkpointer:
        saved = None
    elif self.checkpoint_config[CONF].get(CONFIG_KEY_CHECKPOINT_ID):
        # 指定了具体 checkpoint_id → 时间旅行模式，精确取该 checkpoint
        saved = self.checkpointer.get_tuple(self.checkpoint_config)
    elif replay_state := self.config[CONF].get(CONFIG_KEY_REPLAY_STATE):
        # 子图回放模式：从父图传来的 replay_state 中查找子图对应的 checkpoint
        saved = replay_state.get_checkpoint(
            self.config[CONF].get(CONFIG_KEY_CHECKPOINT_NS, ""),
            self.checkpointer,
            self.checkpoint_config,
        )
    else:
        # 普通调用：按 thread_id 取最新 checkpoint，首次调用返回 None
        saved = self.checkpointer.get_tuple(self.checkpoint_config)

    if saved is None:
        # 首次运行，构建空 checkpoint 作为起点
        saved = CheckpointTuple(
            self.checkpoint_config, empty_checkpoint(), {"step": -2}, None, []
        )
    # 从 checkpoint 恢复所有 channels 的状态
    self.channels, self.managed = channels_from_checkpoint(
        self.specs, self.checkpoint, saver=self.checkpointer, config=...
    )
```

**作用**：`graph.invoke(input, config)` 一进入执行，立刻按 `thread_id` 加载历史 checkpoint，将其中的 `channel_values` 还原为内存中的 channel 对象。

---

#### 调用 ② — `_put_checkpoint`（source="input"）：持久化输入 checkpoint

**源码**：[`_loop.py` ~L1017](libs/langgraph/langgraph/pregel/_loop.py#L1017)（在 `_first()` 中触发）

```python
# _first() 末尾
self._put_checkpoint({"source": "input"})
```

**作用**：将用户的输入 apply 到 channels 后，立即保存一个 `source="input"` 的 checkpoint，记录"这轮调用接收到了什么输入"。

---

#### 调用 ③ — `commit()` 中的 `put_writes`：节点执行完毕即时持久化

**源码**：`_runner.py` 的 `commit()` → `PregelLoop.put_writes()` → `checkpointer.put_writes()`

```python
# _runner.py ~L583  Runner.commit()
def commit(self, task, exception):
    if isinstance(exception, asyncio.CancelledError):
        task.writes.append((ERROR, exception))
        self.put_writes()(task.id, task.writes)     # 节点被取消

    elif exception:
        if isinstance(exception, GraphInterrupt):
            writes = [(INTERRUPT, exception.args[0])]
            self.put_writes()(task.id, writes)      # 节点触发 interrupt
        else:
            task.writes.append((ERROR, exception))
            self.put_writes()(task.id, task.writes) # 节点抛出异常

    else:
        if not task.writes:
            task.writes.append((NO_WRITES, None))
        self.put_writes()(task.id, task.writes)     # 节点正常完成
```

```python
# _loop.py ~L408  PregelLoop.put_writes()
def put_writes(self, task_id: str, writes: WritesT) -> None:
    # 去重、过滤 UntrackedValue 等预处理...
    self.checkpoint_pending_writes.extend((task_id, c, v) for c, v in writes)

    if self.durability != "exit" and self.checkpointer_put_writes is not None:
        # checkpointer_put_writes 就是 checkpointer.put_writes
        # 通过 BackgroundExecutor 异步提交，不阻塞主循环
        fut = self.submit(
            self.checkpointer_put_writes,
            config,           # 含 thread_id + checkpoint_id
            writes_to_save,
            task_id,
            task_path,        # 节点调用栈路径
        )
```

**作用**：每个节点结束（无论成功/中断/报错）后，节点的 writes 立即异步写入 checkpointer。这让**中断后恢复**成为可能 —— 下次 resume 时，引擎从 `checkpoint_pending_writes` 中知道哪些节点已完成，直接跳过。

---

#### 调用 ④ — `after_tick()` 中的 `_put_checkpoint`：每个超步结束保存 checkpoint

**源码**：[`_loop.py` ~L707](libs/langgraph/langgraph/pregel/_loop.py#L707)

```python
def after_tick(self) -> None:
    writes = [w for t in self.tasks.values() for w in t.writes]
    # 把本超步所有节点的 writes 合并 apply 到 channels
    self.updated_channels = apply_writes(
        self.checkpoint,
        self.channels,
        self.tasks.values(),
        self.checkpointer_get_next_version,
        self.trigger_to_nodes,
    )
    # 清空本轮 pending_writes
    self.checkpoint_pending_writes.clear()
    # 只有第一个 tick 会回放（重放）已完成的任务
    self.is_replaying = False
    # ↓ 每个超步完成后，保存完整快照
    self._put_checkpoint({"source": "loop"})
    # 检查是否需要在执行后 interrupt
    if self.interrupt_after and should_interrupt(...):
        self.status = "interrupt_after"
        raise GraphInterrupt()
```

```python
# _loop.py ~L1522  同步版的 _checkpointer_put_after_previous
def _checkpointer_put_after_previous(self, prev, config, checkpoint, metadata, new_versions):
    # 先等待所有 delta-channel write futures 完成（顺序保证）
    if self._delta_write_futs:
        futs, self._delta_write_futs = self._delta_write_futs, []
        concurrent.futures.wait(futs)
    # 再等待上一个 checkpoint put future（防止乱序）
    if prev is not None:
        prev.result()
    # 最终调用 checkpointer.put()
    cast(BaseCheckpointSaver, self.checkpointer).put(
        config, checkpoint, metadata, new_versions
    )
```

**作用**：每个超步（所有当前并行节点都执行完算一个超步）结束后，将当前全量 channel 状态打包成新的完整 checkpoint 写入。Checkpoints 之间有 future 依赖链保证写入顺序。

---

#### 完整调用时序图

```
graph.invoke(input, {"configurable": {"thread_id": "abc"}})
│
├─── PregelLoop.__enter__()
│    └─── checkpointer.get_tuple(config)          ── ① 读取最新 checkpoint（或首次返回 None）
│
├─── _first() ──应用 input 到 channels
│    └─── _put_checkpoint({"source": "input"})
│         └─── checkpointer.put(...)              ── ② source="input" 的初始快照
│
├─── [Tick Loop: 每个超步]
│    │
│    ├─── tick() ── 准备本超步任务
│    │
│    ├─── [节点并发执行]
│    │    ├─── 节点 A 执行完毕 → Runner.commit()
│    │    │    └─── checkpointer.put_writes(...)  ── ③ 节点 A 的 writes 即时持久化
│    │    ├─── 节点 B 执行完毕 → Runner.commit()
│    │    │    └─── checkpointer.put_writes(...)  ── ③ 节点 B 的 writes 即时持久化
│    │    └─── 节点 C 触发 interrupt → Runner.commit()
│    │         └─── checkpointer.put_writes([(INTERRUPT, ...)]) ── ③ interrupt 持久化
│    │
│    └─── after_tick() ── 合并 writes，保存超步快照
│         └─── _put_checkpoint({"source": "loop"})
│              └─── checkpointer.put(...)         ── ④ source="loop" 完整超步快照
│
└─── PregelLoop.__exit__()
     └─── _suppress_interrupt()
          └─── _put_checkpoint(self.checkpoint_metadata) ── ⑤ 最终退出快照（如有变化）
```

---

### Store 的调用全流程

Store 的调用方式与 checkpointer **完全不同**：引擎从不主动调用 store，而是将其打包进 `Runtime` 对象，由节点函数主动使用。

#### Task 准备阶段 — store 注入到 Runtime

**源码**：[`_algo.py` ~L689](libs/langgraph/langgraph/pregel/_algo.py#L689)（`prepare_next_tasks()` 函数中）

```python
def prepare_next_tasks(..., store: BaseStore | None = None, ...):
    for name, proc in processes.items():
        # 为每个将要执行的节点构建 PregelExecutableTask
        ...
        runtime = cast(
            Runtime, configurable.get(CONFIG_KEY_RUNTIME, DEFAULT_RUNTIME)
        )
        # ↓ store 在这里被注入到 Runtime 对象中
        runtime = runtime.override(
            previous=checkpoint["channel_values"].get(PREVIOUS, None),
            store=store,
            execution_info=ExecutionInfo(
                checkpoint_id=checkpoint["id"],
                checkpoint_ns=task_checkpoint_ns,
                task_id=task_id,
                thread_id=configurable.get(CONFIG_KEY_THREAD_ID),
                run_id=str(rid) if (rid := config.get("run_id")) else None,
            ),
        )
        # runtime 随 task 的 config 传递给节点
        return PregelExecutableTask(
            name, val, node, writes,
            patch_config(
                ...,
                configurable={
                    CONFIG_KEY_RUNTIME: runtime,  # ← 挂在这里
                    ...
                },
            ),
            ...
        )
```

**`Runtime` 数据类**（[`runtime.py` ~L125](libs/langgraph/langgraph/runtime.py#L125)）：

```python
@dataclass
class Runtime(Generic[ContextT]):
    context: ContextT = field(default=None)
    """静态上下文，如 user_id、db_conn 等运行依赖。"""

    store: BaseStore | None = field(default=None)
    """图运行的 Store 实例，即 compile(store=...) 传入的对象。"""

    stream_writer: StreamWriter = field(default=_no_op_stream_writer)
    """写入自定义流的函数。"""

    heartbeat: Callable[[], None] = field(default=_no_op_heartbeat)
    """长时运行节点中用于重置 idle_timeout 的心跳函数。"""

    previous: Any = field(default=None)
    """函数式 API 中，该线程上一次调用的返回值（需有 checkpointer）。"""

    execution_info: ExecutionInfo | None = field(default=None)
    """当前节点运行的只读元信息（checkpoint_id、task_id、thread_id 等）。"""
```

#### 节点执行中 — 框架自动完成参数注入

LangGraph 框架会检查节点函数的签名，如果有 `store: BaseStore` 或 `runtime: Runtime` 参数，自动从 `Runtime` 中提取并传入：

```python
# 节点在每次超步的 tick() 准备任务时，store 就已经准备好
# 节点执行时，框架根据函数签名自动注入

def my_node(state: State, *, store: BaseStore) -> dict:
    # store 就是 compile(store=...) 传入的那个对象
    # ↓ 节点代码主动决定何时读、何时写
    memories = store.search(("memories", "user_123"), query="用户喜好", limit=5)
    store.put(("memories", "user_123"), "new_mem", {"content": "新记忆"})
    return {...}
```

#### Store 调用位置总结

```
graph.invoke(input, config)
│
├─── PregelLoop.__enter__()
│    └─── (store 不被调用，仅 checkpointer.get_tuple)
│
├─── [Tick Loop]
│    └─── tick() → prepare_next_tasks(..., store=self.store, ...)
│         └─── 为每个节点构建 task，将 store 注入 runtime.store
│              （仅传引用，store 此时未被调用）
│
│    └─── [节点并发执行]
│         └─── 节点函数内部 —— 节点代码主动调用 store.get/put/search/...
│              （store 只在节点函数执行期间被调用，时机完全由业务代码决定）
│
└─── PregelLoop.__exit__()
     └─── (store 不被调用)
```

---

### 两者核心区别对比

| 维度 | Checkpointer | Store |
|------|--------------|-------|
| **调用者** | `PregelLoop` 引擎自动调用 | 节点函数代码主动调用 |
| **调用时机** | `__enter__`（读）、`after_tick`（写整体）、`commit`（写 writes） | 节点函数执行期间，随时调用 |
| **数据范围** | 当前 `thread_id` 的图状态快照（channel values + versions） | 跨 `thread_id` 的用户级持久数据 |
| **写入驱动** | 引擎框架自动驱动，节点代码无需感知 | 节点业务逻辑主动决定写什么、何时写 |
| **持久化粒度** | 每个超步一个完整快照 + 每个节点的即时 writes | 任意粒度，由调用方决定 |
| **主要用途** | 中断恢复、时间旅行、对话上下文 | 用户画像、长期偏好、跨会话知识库 |
| **键结构** | `(thread_id, checkpoint_id)` → 完整图状态 | `(namespace_tuple, key)` → value dict |
| **替换建议** | 不建议替换（是 LangGraph 运行基础） | 可替换（Mem0、Redis、自定义实现均可） |

---

## 附录：关键源码文件索引

| 文件 | 内容 |
|------|------|
| `libs/checkpoint/langgraph/checkpoint/base/__init__.py` | `BaseCheckpointSaver`、`Checkpoint`、`CheckpointTuple`、`CheckpointMetadata` 定义 |
| `libs/checkpoint/langgraph/store/base/__init__.py` | `BaseStore`、`Item`、`SearchItem`、`GetOp`、`PutOp`、`SearchOp`、`ListNamespacesOp` 定义 |
| `libs/checkpoint/langgraph/store/memory/__init__.py` | `InMemoryStore`：内存 KV + 向量搜索的参考实现 |
| `libs/langgraph/langgraph/pregel/_loop.py` | `PregelLoop`：核心执行引擎，checkpointer 调用全部在此 |
| `libs/langgraph/langgraph/pregel/_algo.py` | `prepare_next_tasks()`：store 注入到 Runtime 的位置 |
| `libs/langgraph/langgraph/pregel/_runner.py` | `Runner.commit()`：节点执行完毕后触发 `put_writes` 的位置 |
| `libs/langgraph/langgraph/runtime.py` | `Runtime` 数据类定义，含 `store`、`context`、`execution_info` 等字段 |
