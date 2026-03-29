# Multi-Turn Blackbox Tests

这个目录放多轮对话黑盒测试。

## 安装

先安装项目测试依赖：

```bash
pip install -e ".[dev]"
```

跑评测测试还需要设置：

```bash
export GOOGLE_API_KEY="your-api-key"
```

如果要跑 `ui_app.py`，还需要额外安装 `streamlit`：

```bash
pip install streamlit
```

## 运行测试

运行整个目录：

```bash
pytest tests/blackbox_multiturn/
```

运行主测试：

```bash
pytest tests/blackbox_multiturn/test_message_handler_multiturn.py
```

## 运行 UI

```bash
streamlit run tests/blackbox_multiturn/ui_app.py
```

## 说明

- 测试数据在 `tests/blackbox_multiturn/test_data/multiturn_triage_cases.json`
- 结果会写到 `tests/blackbox_multiturn/results/multiturn_triage_report.json`
