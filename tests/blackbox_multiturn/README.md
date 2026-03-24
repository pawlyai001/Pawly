# Multi-Turn Blackbox Tests

## Overview

This directory contains blackbox integration tests for multi-turn conversational interactions with the Pawly pet care assistant. These tests evaluate the bot's ability to maintain context, ask relevant follow-up questions, and handle extended conversations across multiple user messages.

## Purpose

These tests verify that the message handler:
- Maintains conversation context across multiple turns
- Asks targeted clarifying questions appropriate to the situation
- Remembers information provided by users without re-asking
- Escalates appropriately when red-flag symptoms appear
- Avoids repetitive or vague questioning
- Provides consistent guidance throughout the conversation

## Test Framework

The tests use:
- **pytest** for test execution
- **DeepEval** for LLM-based conversation evaluation
- **ConversationalGEval** metric for multi-turn assessment
- **aiogram** Dispatcher for realistic message routing
- **Google Gemini** as the evaluation judge model

## Test Metrics

### Conversational G-Eval Metric
Evaluates the effectiveness of multi-turn triage conversations based on custom criteria.
- Default threshold: 0.7
- Assesses conversation flow, question quality, and information retention
- Evaluates against expected outcomes and scenario requirements

## Directory Structure

```
blackbox_multiturn/
├── conftest.py                              # Test fixtures and utilities
├── test_message_handler_multiturn.py        # Main test cases
├── test_data/
│   └── multiturn_triage_cases.json         # Test case definitions
└── results/
    └── multiturn_triage_report.json         # Generated test report
```

## Test Cases

Test cases are defined in `test_data/multiturn_triage_cases.json`. Each case includes:

- `name`: Unique identifier for the test case
- `user_display_name`: Name of the simulated user
- `scenario`: High-level description of the conversation scenario
- `expected_outcome`: What the bot should accomplish in this conversation
- `chatbot_role`: Description of the bot's expected behavior
- `criteria`: Specific evaluation criteria for this conversation
- `threshold`: Minimum score required to pass (default: 0.7)
- `pet_profile`: Pet information (name, species, breed, age, gender, etc.)
- `memories`: Simulated long-term, mid-term, and short-term memories
- `recent_turns`: Initial conversation history (typically empty)
- `user_turns`: List of user messages to send sequentially
- `metadata`: Additional context

## Running the Tests

### Prerequisites

1. Install dev dependencies:
```bash
pip install -e ".[dev]"
```

2. Set the Google API key (required for DeepEval judge):
```bash
export GOOGLE_API_KEY="your-api-key-here"
```

### Execute Tests

Run all multi-turn tests:
```bash
pytest tests/blackbox_multiturn/
```

Run with verbose output:
```bash
pytest tests/blackbox_multiturn/ -v
```

Run specific test:
```bash
pytest tests/blackbox_multiturn/test_message_handler_multiturn.py::test_handle_message_multiturn_with_conversational_geval
```

### Test Execution Flow

1. **Load test cases** from JSON file
2. **For each test case**:
   - Build user and pet objects from case data
   - Create conversation runtime to track state
   - Set up bot, dispatcher, and middleware stack
   - Mock runtime dependencies (database, memory loading, etc.)
3. **Process each user turn**:
   - Create Update object with user message
   - Feed update through full dispatcher pipeline
   - Collect bot responses from fake API
   - Record exchange in conversation runtime
4. **Evaluate full conversation** using ConversationalGEval:
   - Assess against custom criteria
   - Check if expected outcome was achieved
   - Validate role adherence throughout conversation
5. **Generate report** with scores, turn-by-turn transcript, and reasoning

## Test Results

After running tests, a detailed report is generated at:
```
results/multiturn_triage_report.json
```

### Report Structure

```json
{
  "summary": {
    "report_path": "...",
    "total_cases": 5,
    "passed_threshold": 4,
    "below_threshold": 1
  },
  "cases": [
    {
      "name": "test_case_name",
      "status": "passed_threshold | below_threshold",
      "score": 0.85,
      "threshold": 0.7,
      "reason": "detailed explanation from judge",
      "turn_count": 8,
      "turns": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
      ],
      "metadata": {...}
    }
  ]
}
```

## Configuration

### Environment Variables

The test suite sets default values for:
- `DATABASE_URL`: Test database connection (default: `postgresql+asyncpg://test:test@localhost:5432/test`)
- `TELEGRAM_BOT_TOKEN`: Test bot token (default: `test-bot-token`)
- `GOOGLE_API_KEY`: Required for DeepEval, no default (must be provided)
- `DEEPEVAL_TELEMETRY_OPT_OUT`: Disables telemetry (default: `1`)

### Mocked Components

The following components are mocked to enable isolated testing:

**Database & Memory:**
- `load_pet_context`: Returns pre-configured memories and recent turns
- `load_related_memories`: Returns short-term memories
- `load_user_pets`: Returns test pet list

**Session Management:**
- `get_or_create_session`: Returns fake session ID
- `get_or_create_dialogue`: Returns fake dialogue ID

**Storage:**
- `store_raw_message`: Records message without actual DB write
- `store_enriched_messages`: No-op storage
- `enqueue_extraction`: No-op extraction queue

**Triage:**
- `_store_triage_record`: No-op triage storage

**Bot API:**
- `Bot.__call__`: Intercepts all bot API calls
- Fake implementations for SendMessage, SendChatAction, DeleteMessage, EditMessageText

**Redis:**
- In-memory Redis implementation for rate limiting and session storage

## Fixtures

### Session-Scoped Fixtures

- `real_gemini_client`: Actual Gemini client for evaluation (requires API key)
- `deepeval_model`: DeepEval-compatible wrapper around Gemini client
- `load_test_cases`: Function to load test cases from JSON files

### Function-Scoped Fixtures

- `build_user_and_pet`: Creates User and Pet objects from test case data
- `build_update`: Creates aiogram Update objects with Telegram message structure
- `mock_multiturn_runtime`: Applies monkeypatches and returns ConversationRuntime
- `build_router_runtime`: Sets up full bot, dispatcher, and middleware stack

## Key Components

### ConversationRuntime
Tracks conversation state across turns:
```python
class ConversationRuntime:
    def __init__(self, pet, memories, recent_turns)
    def record_exchange(user_text, assistant_text)
```

### FakeBot
Captures all bot API calls:
```python
class FakeBot:
    chat_actions: list[dict]     # SendChatAction calls
    sent_messages: list[dict]    # SendMessage calls
    deleted_messages: list[dict] # DeleteMessage calls
    edited_messages: list[dict]  # EditMessageText calls
```

### InMemoryRedis
Provides Redis functionality without external dependencies:
```python
class InMemoryRedis:
    async def get(key) -> str | None
    async def set(key, value, ex=None) -> bool
    async def incr(key) -> int
    async def expire(key, ttl) -> bool
    def pipeline() -> InMemoryPipeline
```

### TestUserContextMiddleware
Injects test user and pet into dispatcher data:
```python
class TestUserContextMiddleware(BaseMiddleware):
    def __init__(self, user: User, pet: Pet)
    async def __call__(handler, event, data)
```

## Middleware Stack

The test dispatcher includes:
1. **SessionMiddleware**: Manages session state
2. **TestUserContextMiddleware**: Injects test user/pet context
3. **RateLimiterMiddleware**: Rate limiting (uses in-memory Redis)
4. **Message Router**: Routes to actual message handlers

## Adding New Test Cases

To add a new multi-turn test case, edit `test_data/multiturn_triage_cases.json`:

```json
{
  "name": "new_multiturn_scenario",
  "user_display_name": "TestUser",
  "scenario": "Owner reports dog limping, provides details over several messages, then reports improvement",
  "expected_outcome": "Bot should ask targeted questions about onset, severity, affected leg, then provide appropriate advice and acknowledge improvement",
  "chatbot_role": "Pawly is a pet care assistant...",
  "criteria": "Evaluate whether the assistant asks relevant follow-up questions, remembers previously stated information, and adapts recommendations based on new information",
  "threshold": 0.7,
  "pet_profile": {
    "name": "Max",
    "species": "dog",
    "breed": "Labrador",
    "age_in_months": 48,
    "gender": "male",
    "neutered_status": "neutered",
    "weight_latest": 30.5
  },
  "memories": [],
  "recent_turns": [],
  "user_turns": [
    "My dog has been limping today",
    "It's his front right leg. Started this morning after our walk",
    "He can still walk on it but seems uncomfortable. No swelling that I can see",
    "Good news - he's walking much better now after resting all afternoon!"
  ],
  "metadata": {
    "scenario_type": "injury_with_resolution",
    "turn_count": 4
  }
}
```

## Evaluation Criteria Examples

Good criteria are specific and measurable:

**Good:**
```
"The assistant should ask targeted clarification questions that help triage the case,
ask for key differentiating factors relevant to vomiting, stop routine questioning
and escalate immediately if red-flag symptoms appear, and remember information
already provided without asking for it again."
```

**Too Vague:**
```
"The assistant should be helpful and accurate."
```

## Troubleshooting

### Test Skipped: GOOGLE_API_KEY Required

Set the API key:
```bash
export GOOGLE_API_KEY="your-key"
```

### DeepEval JSON Parsing Errors

The custom `GeminiDeepEvalModelMixin` includes retry logic. If parsing fails:
1. Check raw response in error message
2. Verify Gemini API is responding correctly
3. Check for rate limiting or quota issues

### Dispatcher Not Routing Messages

If messages aren't reaching handlers:
1. Verify router is included: `dp.include_router(message_handler.router)`
2. Check middleware order
3. Ensure Update object has required fields

### Fake Redis Not Working

Verify patches are applied correctly:
```python
monkeypatch.setattr("src.db.redis.get_redis", lambda: fake_redis)
monkeypatch.setattr("src.bot.middleware.session.get_redis", lambda: fake_redis)
monkeypatch.setattr("src.bot.middleware.rate_limiter.get_redis", lambda: fake_redis)
```

### No Bot Responses Captured

Check `FakeBot.sent_messages` after `dp.feed_update()`:
```python
before_count = len(fake_api.sent_messages)
await dp.feed_update(bot, update)
new_messages = fake_api.sent_messages[before_count:]
```

## Best Practices

1. **Design realistic conversations**: Use natural progression of questions and answers
2. **Test context retention**: Ensure later turns reference earlier information
3. **Include decision points**: Test how bot adapts to new information
4. **Cover escalation scenarios**: Test transitions from mild to urgent
5. **Validate memory**: Ensure bot doesn't re-ask for provided information
6. **Test clarification**: Include ambiguous inputs that require follow-up
7. **Check conversation closure**: Ensure satisfying endings when appropriate

## Comparison with Single-Turn Tests

| Aspect | Single-Turn | Multi-Turn |
|--------|-------------|------------|
| **Focus** | Individual response quality | Conversation flow and context |
| **Metrics** | Answer Relevancy, Role Adherence | Conversational G-Eval |
| **Test Data** | One input per case | Multiple sequential inputs |
| **State** | Stateless (mocked) | Stateful (ConversationRuntime) |
| **Bot Setup** | Direct function call | Full dispatcher pipeline |
| **Complexity** | Lower | Higher |
| **Use Case** | Basic response validation | Triage logic, context handling |

## Related Tests

- **Single-turn tests**: See `tests/blackbox_singleturn/` for individual response evaluation
- **Unit tests**: See other test directories for component-level testing

## Performance Notes

Multi-turn tests are slower than single-turn tests because:
- Each turn involves full dispatcher pipeline
- Multiple LLM calls per test case (one per turn)
- DeepEval evaluation processes entire conversation
- Typical runtime: 5-15 seconds per test case

For faster iteration during development, consider:
- Testing fewer cases: `pytest -k "specific_case_name"`
- Reducing `user_turns` in test cases
- Using smaller evaluation models (if supported)
