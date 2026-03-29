"""
Streamlit UI for visualizing multi-turn blackbox test results.

Run with: streamlit run tests/blackbox_multiturn/ui_app.py
"""

import json
from pathlib import Path
from typing import Any

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Pawly Multi-Turn Test Results",
    page_icon="🐾",
    layout="wide",
)


def get_available_reports() -> list[str]:
    """Get list of all JSON report files in results folder."""
    results_dir = Path(__file__).parent / "results"
    if not results_dir.exists():
        return []
    return sorted([f.name for f in results_dir.glob("*.json")])


def load_report(filename: str) -> dict[str, Any]:
    """Load the specified test report JSON file."""
    report_path = Path(__file__).parent / "results" / filename

    if not report_path.exists():
        st.error(f"Report file not found: {report_path}")
        st.info("Run tests first: `pytest tests/blackbox_multiturn/`")
        st.stop()

    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)


def render_summary(summary: dict[str, Any], selected_report: str) -> None:
    """Render the summary statistics with report selector."""
    # Title with report selector
    col_title, col_selector = st.columns([2, 1])

    with col_title:
        st.title("🐾 Pawly Multi-Turn Test Results")

    with col_selector:
        available_reports = get_available_reports()
        if available_reports:
            selected = st.selectbox(
                "Select Report",
                available_reports,
                index=available_reports.index(selected_report) if selected_report in available_reports else 0,
                key="report_selector",
            )
            if selected != selected_report:
                st.session_state.selected_report = selected
                st.rerun()

    total = summary.get("total_cases", 0)
    passed = summary.get("passed_threshold", 0)
    failed = summary.get("below_threshold", 0)
    pass_rate = (passed / total * 100) if total > 0 else 0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Cases", total)

    with col2:
        st.metric("Passed", passed, delta=None)

    with col3:
        st.metric("Failed", failed, delta=None)

    with col4:
        st.metric("Pass Rate", f"{pass_rate:.1f}%")

    st.divider()


def get_status_emoji(status: str) -> str:
    """Get emoji for test status."""
    return "✅" if status == "passed_threshold" else "❌"


def get_score_color(score: float, threshold: float) -> str:
    """Get color for score display."""
    if score >= threshold:
        return "green"
    elif score >= threshold * 0.8:
        return "orange"
    else:
        return "red"


def render_case_list_item(case: dict[str, Any], index: int, is_selected: bool) -> None:
    """Render a compact list item for a test case."""
    name = case.get("name", "Unknown")
    status = case.get("status", "unknown")
    score = case.get("score", 0)
    threshold = case.get("threshold", 0.7)

    button_type = "primary" if is_selected else "secondary"

    if st.button(
        f"{get_status_emoji(status)} {name}\n📊 {score:.2f} / {threshold:.2f}",
        key=f"case_{index}",
        type=button_type,
        use_container_width=True,
    ):
        st.session_state.selected_case_index = index


def render_case_details(case: dict[str, Any]) -> None:
    """Render full details of the selected test case."""
    name = case.get("name", "Unknown")
    status = case.get("status", "unknown")
    score = case.get("score", 0)
    threshold = case.get("threshold", 0.7)
    reason = case.get("reason", "No reason provided")
    turn_count = case.get("turn_count", 0)
    turns = case.get("turns", [])

    # Header
    st.header(f"{get_status_emoji(status)} {name}")

    # Metrics row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Score", f"{score:.2f}")
    with col2:
        st.metric("Threshold", f"{threshold:.2f}")
    with col3:
        st.metric("Turn Count", turn_count)

    # Status
    status_color = "green" if status == "passed_threshold" else "red"
    st.markdown(f"**Status:** :{status_color}[{status.replace('_', ' ').title()}]")

    st.divider()

    # Evaluation reason
    st.subheader("📝 Evaluation Reason")
    st.write(reason)

    st.divider()

    # Conversation transcript
    st.subheader("💬 Conversation Transcript")
    if not turns:
        st.info("No conversation turns available")
    else:
        for i, turn in enumerate(turns):
            role = turn.get("role", "unknown")
            content = turn.get("content", "")

            if role == "user":
                st.markdown(f"**👤 User (Turn {i+1}):**")
                st.info(content)
            elif role == "assistant":
                st.markdown(f"**🤖 Assistant (Turn {i+1}):**")
                # Remove HTML tags for better display
                clean_content = content.replace("<i>", "_").replace("</i>", "_")
                clean_content = clean_content.replace("<b>", "**").replace("</b>", "**")
                clean_content = clean_content.replace("<blockquote>", "").replace("</blockquote>", "")
                st.success(clean_content)

            if i < len(turns) - 1:
                st.markdown("---")


def main():
    """Main application."""
    # Initialize session state for selected report
    if "selected_report" not in st.session_state:
        available_reports = get_available_reports()
        st.session_state.selected_report = available_reports[0] if available_reports else "multiturn_triage_report.json"

    # Load report data
    report = load_report(st.session_state.selected_report)
    summary = report.get("summary", {})
    cases = report.get("cases", [])

    # Render summary with report selector
    render_summary(summary, st.session_state.selected_report)

    # Sidebar filters
    st.sidebar.title("🔍 Filters")

    status_filter = st.sidebar.radio(
        "Filter by Status",
        ["All", "Passed", "Failed"],
        index=0,
    )

    # Score range filter
    st.sidebar.markdown("### Score Range")
    min_score = st.sidebar.slider(
        "Minimum Score",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
    )

    # Filter cases
    filtered_cases = cases

    if status_filter == "Passed":
        filtered_cases = [c for c in filtered_cases if c.get("status") == "passed_threshold"]
    elif status_filter == "Failed":
        filtered_cases = [c for c in filtered_cases if c.get("status") == "below_threshold"]

    filtered_cases = [c for c in filtered_cases if c.get("score", 0) >= min_score]

    # Sort options
    sort_by = st.sidebar.selectbox(
        "Sort By",
        ["Name", "Score (High to Low)", "Score (Low to High)", "Turn Count"],
        index=0,
    )

    if sort_by == "Score (High to Low)":
        filtered_cases = sorted(filtered_cases, key=lambda x: x.get("score", 0), reverse=True)
    elif sort_by == "Score (Low to High)":
        filtered_cases = sorted(filtered_cases, key=lambda x: x.get("score", 0))
    elif sort_by == "Turn Count":
        filtered_cases = sorted(filtered_cases, key=lambda x: x.get("turn_count", 0), reverse=True)
    else:  # Name
        filtered_cases = sorted(filtered_cases, key=lambda x: x.get("name", ""))

    # Display filtered count
    st.markdown(f"### Showing {len(filtered_cases)} of {len(cases)} test cases")

    # Initialize session state for selected case
    if "selected_case_index" not in st.session_state:
        st.session_state.selected_case_index = 0

    # Ensure selected index is valid
    if st.session_state.selected_case_index >= len(filtered_cases):
        st.session_state.selected_case_index = 0

    # Two-column layout: left for list, right for details
    if not filtered_cases:
        st.warning("No test cases match the current filters.")
    else:
        left_col, right_col = st.columns([1, 2])

        # Left column: Case list
        with left_col:
            st.markdown("#### Test Cases")
            for i, case in enumerate(filtered_cases):
                is_selected = i == st.session_state.selected_case_index
                render_case_list_item(case, i, is_selected)

        # Right column: Case details
        with right_col:
            selected_case = filtered_cases[st.session_state.selected_case_index]
            render_case_details(selected_case)


if __name__ == "__main__":
    main()
