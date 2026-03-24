import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

REPORT_PATH = Path(__file__).parent / "results" / "multiturn_triage_report.json"


def _write_report(report: dict[str, Any]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=True)


def test_handle_message_multiturn_with_conversational_geval(
    load_test_cases,
    build_user_and_pet,
    build_update,
    mock_multiturn_runtime,
    build_router_runtime,
    deepeval_model,
) -> None:
    pytest.importorskip("deepeval")
    from deepeval.metrics import ConversationalGEval
    from deepeval.test_case import ConversationalTestCase, Turn
    from deepeval.test_case.conversational_test_case import TurnParams

    cases = load_test_cases("multiturn_triage_cases.json")
    report_cases: list[dict[str, Any]] = []

    for case in cases:
        user, pet = build_user_and_pet(case)
        runtime = mock_multiturn_runtime(case, user, pet)
        bot, dp, fake_api, _redis = build_router_runtime(user, pet)
        full_turns: list[Turn] = []

        for index, user_text in enumerate(case["user_turns"], start=1):
            before_count = len(fake_api.sent_messages)
            update = build_update(user_text, message_id=index, telegram_user_id=10001)
            asyncio.run(
                dp.feed_update(bot, update)
            )
            new_messages = fake_api.sent_messages[before_count:]
            assistant_text = "\n".join(item["text"] for item in new_messages)
            full_turns.append(Turn(role="user", content=user_text))
            full_turns.append(Turn(role="assistant", content=assistant_text))
            runtime.record_exchange(user_text, assistant_text)

        conversation_case = ConversationalTestCase(
            name=case["name"],
            scenario=case["scenario"],
            expected_outcome=case["expected_outcome"],
            chatbot_role=case["chatbot_role"],
            turns=full_turns,
            additional_metadata=case.get("metadata"),
        )
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
            model=deepeval_model,
            async_mode=False,
            verbose_mode=False,
        )
        score = metric.measure(
            conversation_case,
            _show_indicator=False,
            _log_metric_to_confident=False,
        )

        report_cases.append(
            {
                "name": case["name"],
                "status": "passed_threshold" if score >= metric.threshold else "below_threshold",
                "score": score,
                "threshold": metric.threshold,
                "reason": metric.reason,
                "turn_count": len(full_turns),
                "turns": [{"role": turn.role, "content": turn.content} for turn in full_turns],
                "metadata": case.get("metadata"),
            }
        )

    summary = {
        "report_path": str(REPORT_PATH),
        "total_cases": len(report_cases),
        "passed_threshold": sum(1 for item in report_cases if item["status"] == "passed_threshold"),
        "below_threshold": sum(1 for item in report_cases if item["status"] == "below_threshold"),
    }
    _write_report({"summary": summary, "cases": report_cases})
