# Policy Checker 和 Router 触发机制详解

## 概述

当使用 `@os_instance.instruction()` 装饰器时，ArbiterOS 会在节点函数周围添加一个 wrapper，这个 wrapper 会在特定时机触发 policy checker 和 router。

## 触发时机

### 1. Policy Checker 触发时机（`_check_before`）

**触发时机：节点函数执行之前**

```python
# 在 core.py 的 wrapper 函数中：
self.history[-1].check_policy_results, all_passed = self._check_before()  # ← 这里触发
result = func(*args, **kwargs)  # ← 然后才执行原始函数
```

**执行流程：**
1. 当 LangGraph 调用被装饰的节点函数时（如 `plan_step`, `execute_step`, `replan_step`）
2. 首先执行 `_check_before()`，遍历所有注册的 `PolicyChecker`
3. 每个 checker 调用 `check_before(history)` 方法
4. 如果 checker 返回 `False`，会记录错误（但不会阻止执行，除非抛出异常）
5. 然后才执行原始的节点函数

**示例：**
```python
@os_instance.instruction(Instr.GENERATE)
def plan_step(state: PlanExecute) -> PlanExecute:
    # 在执行这行代码之前，会先执行所有 PolicyChecker.check_before()
    plan_result = planner.invoke({"messages": [("user", state["input"])]})
    return {"plan": plan_result.steps}
```

### 2. Policy Router 触发时机（`_route_after`）

**触发时机：节点函数执行之后，返回结果之前**

```python
# 在 core.py 的 wrapper 函数中：
result = func(*args, **kwargs)  # ← 先执行原始函数
# ... 存储输出到 history ...
self.history[-1].route_policy_results, destination = self._route_after()  # ← 这里触发

if destination:
    return Command(update=result, goto=destination)  # ← 如果 router 决定路由，返回 Command
return result  # ← 否则返回原始结果
```

**执行流程：**
1. 节点函数执行完成，结果存储在 `history[-1].output_state` 中
2. 执行 `_route_after()`，遍历所有注册的 `PolicyRouter`
3. 每个 router 调用 `route_after(history)` 方法
4. 如果 router 返回非 None 的目标节点名，会返回 `Command(goto=destination)`
5. LangGraph 会根据这个 Command 路由到指定节点，而不是继续正常的边

**示例：**
```python
@os_instance.instruction(Instr.GENERATE)
def replan_step(state: PlanExecute) -> PlanExecute:
    output = replanner.invoke(replanner_input)
    result = {"plan": output.action.steps}  # ← 执行完成
    
    # ← 在这里，会执行所有 PolicyRouter.route_after()
    # 如果 PlanQualityPolicyRouter 发现 plan quality < 0.5
    # 会返回 Command(goto="planner")
    # LangGraph 会路由到 "planner" 节点，而不是继续到 "agent"
    
    return result
```

## 以 PlanQualityPolicyRouter 为例

### 完整执行流程

假设 workflow 是：`START -> planner -> agent -> replan -> agent -> END`

#### 步骤 1: planner 节点执行

```python
@os_instance.instruction(Instr.GENERATE)
def plan_step(state: PlanExecute) -> PlanExecute:
    # 1. _check_before() 执行（所有 PolicyChecker）
    #    - PlanQualityAlertChecker.check_before() 被调用
    #    - 此时 history[-1] 是 planner 的输入状态
    #    - 因为还没有输出，所以不会触发路由
    
    # 2. 执行原始函数
    plan_result = planner.invoke({"messages": [("user", state["input"])]})
    result = {"plan": plan_result.steps}
    
    # 3. 结果存储在 history[-1].output_state = {"plan": [...]}
    
    # 4. _route_after() 执行（所有 PolicyRouter）
    #    - PlanQualityPolicyRouter.route_after() 被调用
    #    - 检查 history[-1].output_state["plan"]
    #    - 计算 plan_quality = 0.8（假设）
    #    - 因为 0.8 >= 0.5 (threshold)，返回 None
    #    - 不触发路由，继续正常流程
    
    return result  # 正常返回，LangGraph 继续到 "agent" 节点
```

#### 步骤 2: agent 节点执行

```python
@os_instance.instruction(Instr.TOOL_CALL)
def execute_step(state: PlanExecute) -> PlanExecute:
    # 1. _check_before() 执行
    #    - StepCountAlertChecker.check_before() 被调用
    #    - 检查 step_count，如果超过阈值会记录警告
    
    # 2. 执行原始函数
    agent_response = agent_executor.invoke(...)
    result = {"past_steps": [(task, result_content)]}
    
    # 3. _route_after() 执行
    #    - PlanQualityPolicyRouter.route_after() 被调用
    #    - 但此时 output_state 中没有 "plan"（只有 "past_steps"）
    #    - 所以不会触发路由
    
    return result  # 正常返回，LangGraph 继续到 "replan" 节点
```

#### 步骤 3: replan 节点执行（关键）

```python
@os_instance.instruction(Instr.GENERATE)
def replan_step(state: PlanExecute) -> PlanExecute:
    # 1. _check_before() 执行
    
    # 2. 执行原始函数
    output = replanner.invoke(replanner_input)
    result = {"plan": output.action.steps}  # 假设生成了新的 plan
    
    # 3. 结果存储在 history[-1].output_state = {"plan": [...]}
    
    # 4. _route_after() 执行（关键步骤）
    #    - PlanQualityPolicyRouter.route_after() 被调用
    #    - 检查 history[-1].output_state["plan"]
    #    - 计算 plan_quality = 0.3（假设质量较低）
    #    - 因为 0.3 < 0.5 (threshold)，返回 "planner"
    #    - wrapper 返回 Command(update=result, goto="planner")
    
    # 5. LangGraph 收到 Command(goto="planner")
    #    - 忽略正常的边（replan -> agent）
    #    - 路由到 "planner" 节点
    #    - 导致 workflow 变成：replan -> planner -> agent -> replan -> ...
    
    return Command(update=result, goto="planner")  # 被 router 修改
```

### PlanQualityPolicyRouter.route_after() 详细逻辑

```python
def route_after(self, history: list) -> str | None:
    """在 replan 节点执行后触发"""
    
    # 1. 检查 history 长度
    if not history or len(history) < 2:
        return None  # 历史记录不足，不路由
    
    # 2. 获取最后一个节点的输出状态
    last_entry = history[-1]  # replan 节点的执行记录
    output_state = last_entry.output_state  # {"plan": [...]}
    
    # 3. 防止无限循环
    prev_entry = history[-2]  # 前一个节点（agent）
    prev_node = prev_entry.node_name  # "agent"
    if prev_node == self.target:  # "planner"
        return None  # 如果前一个节点就是 planner，不路由回去
    
    # 4. 计算 plan quality
    plan = output_state.get("plan", [])
    if not plan:
        plan_quality = 0.0
    else:
        step_count = len(plan)
        # 根据步骤数量计算质量
        if step_count > 10:
            plan_quality = 0.3
        elif step_count < 2:
            plan_quality = 0.5
        else:
            plan_quality = min(1.0, 0.7 + 0.1 * (7 - abs(step_count - 5)))
        
        # 添加清晰度奖励
        avg_step_length = sum(len(step) for step in plan) / max(len(plan), 1)
        clarity_bonus = min(0.2, avg_step_length / 100.0)
        plan_quality = min(1.0, plan_quality + clarity_bonus)
    
    # 5. 决定是否路由
    if plan_quality < self.threshold:  # 0.5
        logger.info(f"Plan quality {plan_quality:.2f} below threshold, routing to {self.target}")
        return self.target  # 返回 "planner"，触发路由
    else:
        return None  # 不路由，继续正常流程
```

## 关键点总结

1. **Policy Checker** (`check_before`):
   - 在节点函数执行**之前**触发
   - 用于验证输入状态或历史记录
   - 如果返回 `False` 或抛出异常，可能阻止执行

2. **Policy Router** (`route_after`):
   - 在节点函数执行**之后**触发
   - 可以访问节点的输出状态（`history[-1].output_state`）
   - 如果返回目标节点名，会改变执行流程
   - 通过返回 `Command(goto=destination)` 实现路由

3. **触发顺序**:
   ```
   节点被调用
   ↓
   _check_before() → 所有 PolicyChecker.check_before()
   ↓
   执行原始节点函数
   ↓
   存储输出到 history[-1].output_state
   ↓
   _route_after() → 所有 PolicyRouter.route_after()
   ↓
   如果 router 返回目标节点 → 返回 Command(goto=target)
   否则 → 返回原始结果
   ```

4. **History 结构**:
   ```python
   history = [
       History(
           node_name="planner",
           input_state={...},
           output_state={"plan": [...]},  # ← router 可以访问这个
           check_policy_results={...},
           route_policy_results={...}
       ),
       History(
           node_name="agent",
           input_state={...},
           output_state={"past_steps": [...]},
           ...
       ),
       History(
           node_name="replan",
           input_state={...},
           output_state={"plan": [...]},  # ← PlanQualityPolicyRouter 检查这个
           ...
       )
   ]
   ```

## 节点过滤机制（Node Filtering）

为了避免 policy 在不相关的节点上触发（导致误判和无限循环），现在支持节点过滤机制。

### 问题场景

例如，`PlanQualityPolicyRouter` 在 `execute_step` 节点执行后也会被触发，但 `execute_step` 的输出中没有 `plan` 字段，导致：
- 检查到 `plan = []`（空）
- 计算 `plan_quality = 0.0`
- 触发路由回 `planner`
- 造成无限循环

### 解决方案

每个 `PolicyChecker` 和 `PolicyRouter` 现在都有一个 `target_nodes` 属性：

```python
class PlanQualityPolicyRouter(PolicyRouter):
    def __init__(
        self,
        name: str = "replan_on_low_plan_quality",
        threshold: float = 0.5,
        target: str = "planner",
        target_nodes: list[str] | None = None  # ← 新增参数
    ):
        self.name = name
        self.threshold = threshold
        self.target = target
        # 只在 "planner" 和 "replan" 节点上触发（因为只有这些节点输出 plan）
        self.target_nodes = target_nodes if target_nodes is not None else ["planner", "replan"]
```

### 工作原理

在 `_check_before()` 和 `_route_after()` 中，会先检查当前节点是否在 `target_nodes` 中：

```python
def _route_after(self):
    current_node_name = self.history[-1].node_name
    for router in self.policy_routers:
        # 检查是否应该在这个节点上应用
        if hasattr(router, 'should_apply'):
            if not router.should_apply(current_node_name):
                continue  # 跳过这个 router
        
        decision = router.route_after(self.history)
        # ...
```

### 默认配置

在 `custom_policy.py` 中，每个 policy 都有合理的默认 `target_nodes`：

- **PlanQualityPolicyRouter**: `["planner", "replan"]` - 只有这些节点输出 plan
- **StepCountPolicyRouter**: `["agent"]` - 只有这个节点执行步骤
- **ExecutionSuccessPolicyRouter**: `["agent"]` - 只有这个节点执行步骤
- **PlanQualityAlertChecker**: `["planner", "replan"]` - 只有这些节点输出 plan
- **StepCountAlertChecker**: `["agent"]` - 只有这个节点执行步骤
- **ErrorCountAlertChecker**: `["agent"]` - 只有这个节点可能出错

### 使用示例

```python
# 只在特定节点上触发
PlanQualityPolicyRouter(
    name="replan_on_low_plan_quality",
    threshold=0.5,
    target="planner",
    target_nodes=["planner", "replan"]  # 明确指定
)

# 在所有节点上触发（默认行为）
SomeOtherRouter(
    name="global_router",
    target_nodes=None  # None 表示所有节点
)

# 禁用某个 policy（不推荐，但可以）
DisabledRouter(
    name="disabled",
    target_nodes=[]  # 空列表表示从不触发
)
```

## 注意事项

1. **Router 会改变 workflow**：如果 router 返回目标节点，会覆盖正常的边（edge）
2. **防止无限循环**：代码中检查前一个节点是否是目标节点，避免立即路由回去
3. **节点过滤**：现在每个 policy 可以指定它应该在哪些节点上触发，避免误判
4. **所有 router 都会执行**：但只有第一个返回非 None 的 router 会生效
5. **History 是累积的**：router 可以访问完整的执行历史，不仅仅是当前节点

