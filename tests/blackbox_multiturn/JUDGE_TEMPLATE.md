# Multi-Turn ConversationalGEval Judge 模板解析

## 概述

Multi-turn 测试使用 DeepEval 的 `ConversationalGEval` 指标，该指标通过两阶段提示词来评估对话质量。

## 评估流程

### 阶段 1: 生成评估步骤 (Generate Evaluation Steps)

如果测试用例没有提供预定义的 `evaluation_steps`，DeepEval 会首先根据 `criteria` 自动生成评估步骤。

#### 第一阶段模板 (template.py:6-21)

```python
def generate_evaluation_steps(parameters: str, criteria: str):
    return f"""Given an evaluation criteria which outlines how you should judge a conversation between a user and an LLM chatbot using the {parameters} fields in each turn, generate 3-4 concise evaluation steps based on the criteria below. Based on the evaluation criteria, you MUST make it clear how to evaluate the {parameters} in relation to one another in each turn, as well as the overall quality of the conversation.

Evaluation Criteria:
{criteria}

**
IMPORTANT: Please make sure to only return in JSON format, with the "steps" key as a list of strings. No words or explanation is needed.
Example JSON:
{{
    "steps": <list_of_strings>
}}
**

JSON:
"""
```

#### 在你的测试中的实际例子

基于你的测试用例 (`multiturn_triage_cases.json`):

**输入到模板:**
- `parameters`: "Role, Content, Scenario, and Expected Outcome"
  - 来自 `evaluation_params=[TurnParams.ROLE, TurnParams.CONTENT, TurnParams.SCENARIO, TurnParams.EXPECTED_OUTCOME]`
- `criteria`: "Evaluate whether the assistant manages this multi-turn pet-health conversation effectively. The assistant should ask targeted clarification questions that help triage the case, ask for key differentiating factors relevant to vomiting, stop routine questioning and escalate immediately if red-flag symptoms appear, and remember information already provided without asking for it again. Penalize vague or repetitive questioning, missing critical follow-up questions, continuing a checklist after a clear emergency signal, or re-asking facts the user already gave."

**Judge 收到的提示词 (简化版):**
```
Given an evaluation criteria which outlines how you should judge a conversation
between a user and an LLM chatbot using the Role, Content, Scenario, and Expected
Outcome fields in each turn, generate 3-4 concise evaluation steps based on the
criteria below...

Evaluation Criteria:
Evaluate whether the assistant manages this multi-turn pet-health conversation
effectively. The assistant should ask targeted clarification questions that help
triage the case, ask for key differentiating factors relevant to vomiting...

Return JSON:
{
    "steps": ["step 1", "step 2", "step 3"]
}
```

**Judge 可能生成的评估步骤:**
```json
{
  "steps": [
    "Check if the assistant asks targeted, relevant clarifying questions about the vomiting episode (frequency, appearance, timing, other symptoms) rather than generic questions.",
    "Verify the assistant remembers information already provided by the user and does not re-ask for facts already stated.",
    "Assess whether the assistant appropriately escalates urgency if red-flag symptoms are mentioned, stopping routine questioning when emergency indicators appear.",
    "Evaluate if the overall conversation flow demonstrates effective triage logic aligned with the expected outcome and scenario."
  ]
}
```

---

### 阶段 2: 评估对话 (Evaluate Conversation)

使用生成的评估步骤（或预定义的步骤）来评估实际对话。

#### 第二阶段模板 (template.py:24-77)

```python
def generate_evaluation_results(
    evaluation_steps: str,
    test_case_content: str,
    turns: List[Dict],
    parameters: str,
    rubric: Optional[str] = None,
) -> str:
    return f"""You are given a set of Evaluation Steps that describe how to assess a conversation between a user and an LLM chatbot. Your task is to return a JSON object with exactly two fields:

    1. `"score"`: An integer from 0 to 10 (inclusive), where:
    - 10 = The conversation *fully* meets the criteria described in the Evaluation Steps
    - 0 = The conversation *completely fails* to meet the criteria
    - All other scores represent varying degrees of partial fulfillment

    2. `"reason"`: A **concise but precise** explanation for the score. Your reasoning must reference specific aspects of the evaluation steps and mention relevant details from the conversation and the given parameters. DO NOT include the score value in your explanation.

    Evaluation Steps:
    {evaluation_steps}

    Conversation:
    {turns}

    {test_case_content}

    Parameters to consider during evaluation:
    {parameters}

    ---
    IMPORTANT: You MUST return only a valid JSON object with the exact keys `"score"` and `"reason"`. No additional text, commentary, or formatting.

    ---
    Example JSON:
    {{
        "reason": "Your concise and informative reason here.",
        "score": 0
    }}

    JSON:"""
```

#### 在你的测试中的实际例子

**输入到模板:**

1. **evaluation_steps** (编号后的步骤):
```
1. Check if the assistant asks targeted, relevant clarifying questions about the vomiting episode...
2. Verify the assistant remembers information already provided by the user...
3. Assess whether the assistant appropriately escalates urgency if red-flag symptoms are mentioned...
4. Evaluate if the overall conversation flow demonstrates effective triage logic...
```

2. **turns** (对话轮次，转换为字典):
```python
[
    {
        "role": "user",
        "content": "Oh dear, I hope someone can help me. My little Biscuit just threw up all over the kitchen floor and I'm just beside myself. She's my whole world. Is she going to be okay??"
    },
    {
        "role": "assistant",
        "content": "I understand you're worried about Biscuit. Let me ask you a few questions to better understand the situation..."
    },
    {
        "role": "user",
        "content": "She's a Cocker Spaniel, she'll be 9 in June. It happened maybe 20 minutes ago..."
    },
    {
        "role": "assistant",
        "content": "Thank you for that information. Since she seems comfortable now and it was just once..."
    }
    // ... 更多轮次
]
```

3. **test_case_content** (非 turn 的测试用例属性):
```
Scenario:
An anxious elderly owner reports a single vomiting episode in her dog, asks follow-up medication questions, and comes back the next day with an update.

Expected Outcome:
The assistant should ask purposeful clarifying questions, avoid over-triaging the initial mild episode, avoid recommending human medication, remember previously given information, and provide an appropriate day-2 follow-up response.
```

4. **parameters**:
```
"Role, Content, Scenario, and Expected Outcome"
```

**Judge 收到的完整提示词 (简化版):**
```
You are given a set of Evaluation Steps that describe how to assess a conversation
between a user and an LLM chatbot. Your task is to return a JSON object with exactly
two fields...

Evaluation Steps:
1. Check if the assistant asks targeted, relevant clarifying questions...
2. Verify the assistant remembers information already provided...
3. Assess whether the assistant appropriately escalates urgency...
4. Evaluate if the overall conversation flow demonstrates effective triage logic...

Conversation:
[
    {"role": "user", "content": "Oh dear, I hope someone can help me. My little Biscuit just threw up..."},
    {"role": "assistant", "content": "I understand you're worried about Biscuit..."},
    ...
]

Scenario:
An anxious elderly owner reports a single vomiting episode in her dog...

Expected Outcome:
The assistant should ask purposeful clarifying questions, avoid over-triaging...

Parameters to consider during evaluation:
Role, Content, Scenario, and Expected Outcome

---
Return JSON with "score" (0-10) and "reason".
```

**Judge 返回的评估结果:**
```json
{
    "score": 8,
    "reason": "The assistant effectively asked targeted clarifying questions about the vomiting episode timing, frequency, and Biscuit's current condition (Step 1). The assistant correctly remembered that Biscuit is a Cocker Spaniel and avoided re-asking for already provided details (Step 2). When the user mentioned the dog was comfortable and had only vomited once, the assistant appropriately provided reassurance without over-escalating (Step 3). However, the assistant could have more explicitly acknowledged the user's anxiety earlier in the conversation to better align with the expected outcome of warmly addressing the anxious owner persona."
}
```

---

## 参数映射 (TurnParams)

在你的测试中使用的 `evaluation_params`:

```python
evaluation_params=[
    TurnParams.ROLE,           # → "Role"
    TurnParams.CONTENT,        # → "Content"
    TurnParams.SCENARIO,       # → "Scenario"
    TurnParams.EXPECTED_OUTCOME,  # → "Expected Outcome"
]
```

### 参数的作用

| TurnParam | 映射字符串 | 来源 | 用途 |
|-----------|-----------|------|------|
| `ROLE` | "Role" | `test_case.chatbot_role` | 定义 chatbot 应该扮演的角色和行为约束 |
| `CONTENT` | "Content" | 每个 `turn.content` | 对话的实际文本内容 |
| `SCENARIO` | "Scenario" | `test_case.scenario` | 对话的背景场景描述 |
| `EXPECTED_OUTCOME` | "Expected Outcome" | `test_case.expected_outcome` | 期望的对话结果 |

### 参数的处理方式

#### Turn 级别参数 (包含在 turns 列表中):
- `ROLE`: 每个 turn 的 `role` 字段 ("user" 或 "assistant")
- `CONTENT`: 每个 turn 的 `content` 字段

#### Test Case 级别参数 (包含在 test_case_content 中):
- `SCENARIO`: 单独显示为 "Scenario: ..."
- `EXPECTED_OUTCOME`: 单独显示为 "Expected Outcome: ..."

**重要**: `chatbot_role` 虽然在 `ConversationalTestCase` 中提供，但不会直接显示在提示词中。它的作用是：
1. DeepEval 内部用于构建上下文
2. 通过 `TurnParams.ROLE` 参数间接影响评估（告诉 judge 关注角色遵守）

---

## 评分机制

### 基础评分 (0-10)
Judge 返回一个 0-10 的整数分数。

### 加权分数计算 (Weighted Summed Score)
如果使用支持 logprobs 的模型（如 OpenAI），DeepEval 会：
1. 提取 judge 输出的 logprobs
2. 查看得分 token 的 top_logprobs（默认 top 20）
3. 计算加权平均分数，考虑 judge 对不同分数的置信度

例如，如果 judge 输出分数 8，但 logprobs 显示：
- "8": 60% 概率
- "7": 30% 概率
- "9": 10% 概率

加权分数 = (8 × 0.6) + (7 × 0.3) + (9 × 0.1) = 7.8

### 最终评分
```python
self.score = float(g_score) / 10  # 转换为 0.0-1.0 范围
self.success = self.score >= self.threshold  # 与阈值比较
```

在你的测试中，`threshold=0.7` 意味着需要分数 ≥ 7/10 才算通过。

---

## 实际调用流程

基于你的测试代码 (`test_message_handler_multiturn.py`):

```python
metric = ConversationalGEval(
    name="MultiTurnTriageEffectiveness",
    criteria=case["criteria"],
    evaluation_params=[
        TurnParams.ROLE,
        TurnParams.CONTENT,
        TurnParams.SCENARIO,
        TurnParams.EXPECTED_OUTCOME,
    ],
    threshold=case.get("threshold", 0.7),
    model=deepeval_model,  # 你的 GeminiDeepEvalModel
    async_mode=False,
)
score = metric.measure(conversation_case)
```

**执行步骤:**

1. **检查是否有预定义的 evaluation_steps**
   - 你的测试中没有提供，所以会自动生成

2. **生成评估步骤** (如果需要)
   ```python
   # 调用 _generate_evaluation_steps()
   # 使用模板 generate_evaluation_steps(criteria, parameters)
   # 发送给 Gemini judge
   # 返回 3-4 个评估步骤
   ```

3. **评估对话**
   ```python
   # 调用 evaluate(test_case)
   # 构建 test_case_content (scenario + expected_outcome)
   # 转换 turns 为字典列表
   # 使用模板 generate_evaluation_results(...)
   # 发送给 Gemini judge
   # 返回 {"score": 8, "reason": "..."}
   ```

4. **处理结果**
   ```python
   # score = 8 / 10 = 0.8
   # success = 0.8 >= 0.7 = True
   ```

---

## 自定义 Judge 行为

如果你想修改 judge 的行为，可以：

### 1. 自定义 evaluation_steps
```python
metric = ConversationalGEval(
    name="MultiTurnTriageEffectiveness",
    evaluation_steps=[
        "Check if questions are specific to pet care",
        "Verify no medication dosages are given",
        "Assess empathetic tone throughout",
    ],
    # 不需要 criteria，因为提供了 steps
    evaluation_params=[...],
    model=deepeval_model,
)
```

### 2. 自定义评估模板
```python
from deepeval.metrics.conversational_g_eval.template import ConversationalGEvalTemplate

class MyCustomTemplate(ConversationalGEvalTemplate):
    @staticmethod
    def generate_evaluation_results(
        evaluation_steps: str,
        test_case_content: str,
        turns: List[Dict],
        parameters: str,
        rubric: Optional[str] = None,
    ) -> str:
        # 自定义提示词
        return f"""Your custom prompt here...

        Evaluation Steps:
        {evaluation_steps}

        Conversation:
        {turns}

        ...
        """

metric = ConversationalGEval(
    name="MultiTurnTriageEffectiveness",
    criteria=case["criteria"],
    evaluation_params=[...],
    model=deepeval_model,
    evaluation_template=MyCustomTemplate,  # 使用自定义模板
)
```

### 3. 使用 Rubric 评分标准
```python
from deepeval.metrics.g_eval.utils import Rubric

metric = ConversationalGEval(
    name="MultiTurnTriageEffectiveness",
    criteria=case["criteria"],
    evaluation_params=[...],
    rubric=[
        Rubric(score_range=(9, 10), expected_outcome="Excellent triage with no issues"),
        Rubric(score_range=(7, 8), expected_outcome="Good triage with minor issues"),
        Rubric(score_range=(5, 6), expected_outcome="Adequate but missing key questions"),
        Rubric(score_range=(0, 4), expected_outcome="Poor triage or inappropriate responses"),
    ],
    model=deepeval_model,
)
```

---

## 调试 Judge 输出

### 查看详细日志
```python
metric = ConversationalGEval(
    name="MultiTurnTriageEffectiveness",
    criteria=case["criteria"],
    evaluation_params=[...],
    model=deepeval_model,
    verbose_mode=True,  # 启用详细日志
)
score = metric.measure(conversation_case)

# 查看详细步骤
print(metric.verbose_logs)
```

### 检查生成的评估步骤
```python
score = metric.measure(conversation_case)
print("Generated Evaluation Steps:")
for i, step in enumerate(metric.evaluation_steps, 1):
    print(f"{i}. {step}")
```

### 查看评分原因
```python
score = metric.measure(conversation_case)
print(f"Score: {metric.score}")
print(f"Reason: {metric.reason}")
print(f"Success: {metric.success}")
```

---

## 总结

### chatbot_role 的作用

在 multiturn 测试中，`chatbot_role` **有用但作用隐含**：

1. ✅ **传递给 ConversationalTestCase**: 作为测试用例的元数据
2. ✅ **通过 TurnParams.ROLE 影响评估**: 虽然不直接显示在提示词中，但通过包含 `TurnParams.ROLE` 在 `evaluation_params` 中，告诉 judge 要关注角色遵守
3. ❌ **不会直接显示在 judge 提示词中**: 与 singleturn 的 `RoleAdherenceMetric` 不同，不会明确显示 "Chatbot Role: ..." 字段

### 关键区别

| 方面 | Single-Turn | Multi-Turn |
|------|-------------|------------|
| **指标** | `RoleAdherenceMetric` | `ConversationalGEval` |
| **chatbot_role 显示** | 明确显示在提示词中 | 不直接显示（隐含在 ROLE 参数中） |
| **评估方式** | 专门的角色遵守指标 | 综合评估对话质量（角色是其中一部分） |
| **TurnParams.ROLE 作用** | 识别对话中的角色 | 识别角色 + 提醒 judge 关注角色行为 |

如果你想让 chatbot_role 在 multiturn 评估中更明确，可以：
- 在 `criteria` 中明确提到角色要求
- 或者自定义评估模板，在提示词中显式包含 chatbot_role
