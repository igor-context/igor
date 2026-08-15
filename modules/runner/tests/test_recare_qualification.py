"""Deterministic ReCaRe legal amendment qualification."""
import json
import shutil
from pathlib import Path
from igor_runner.recare_qualification import _enrichment_input_text, qualify_recare


def test_recare_deterministic_qualification(tmp_path):
    scenario = Path("/opt/scenarios/recare-legal/v0.1")
    result = qualify_recare(str(scenario), str(tmp_path / "output"))
    assert result["valid"]
    # Verify three amendment events processed
    assert result["amendment_events"] == 3
    # Verify snapshots preserved
    assert result["snapshots"]["historical_queryable"]
    assert result["snapshots"]["current_queryable"]
    # Verify authority resolution
    assert result["resolution"]["current_selected"]
    assert result["resolution"]["superseded_rejected"]
    # Verify contract outcomes
    assert result["contract"]["allowed_count"] >= 1
    assert result["contract"]["denied_count"] >= 1
    assert result["contract"]["abstained_count"] >= 1
    # Verify mutation
    assert result["mutation"]["invalidated_count"] >= 1
    assert result["mutation"]["reused_count"] >= 1
    assert result["retrieval"]["query_vector_identity"]
    assert "ANN" in result["retrieval"]["execution"]["plan"]
    assert result["contract"]["denied_package_count"] == 0
    assert result["contract"]["abstained_package_count"] == 0
    # Verify report exists
    output = tmp_path / "output"
    assert (output / "qualification.json").is_file()
    assert (output / "report.html").is_file()


def test_recare_incomplete_source_record_is_recorded_not_crashed(tmp_path):
    source = Path("/opt/scenarios/recare-legal/v0.1")
    scenario = tmp_path / "scenario"
    shutil.copytree(source, scenario)
    payload = json.loads((scenario / "fixtures.json").read_text())
    incomplete = dict(payload["amendments"][0]["articles"][0])
    incomplete["article_id_after"] = None
    incomplete["text_after"] = None
    payload["amendments"][0]["articles"].append(incomplete)
    (scenario / "fixtures.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    result = qualify_recare(str(scenario), str(tmp_path / "output"))

    assert result["valid"]
    assert result["dataset"]["acquired_rows"] == result["dataset"]["processed_rows"] + 1
    assert len(result["source_issues"]) == 1
    issue = result["source_issues"][0]
    assert issue["record_index"] == 1
    assert issue["row_number"] is None
    assert issue["event_id"] == payload["amendments"][0]["amendment_law_id"]
    assert issue["law_id"] == payload["amendments"][0]["law_id"]
    assert issue["reason_code"] == "missing_required_source_concepts"
    assert issue["missing_concepts"] == ["article_id_after", "text_after"]
    assert issue["action"] == "excluded_from_context_ingestion"


def test_recare_live_prompt_guidance_is_concise():
    event_snapshots = [
        {
            "amendment_reason": "Because the directive changed.",
            "article_number": "2",
            "version": "before",
            "text": "Before text.",
        },
        {
            "amendment_reason": "Because the directive changed.",
            "article_number": "2",
            "version": "after",
            "text": "After text.",
        },
    ]
    prompt = _enrichment_input_text(event_snapshots)
    assert "Return only a JSON object matching the schema." in prompt
    assert "Keep rationale_summary to one concise sentence under 160 characters." in prompt
    assert "Do not copy the source text verbatim." in prompt
    assert "Article 2 before version" in prompt
    assert "Article 2 after version" in prompt
