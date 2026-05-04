"""Dry-run test for the per-flag 80% aggregation gate in run_m_critics.

Demonstrates the gating behavior the new prompt + aggregation introduces.
Stubs out the LLM call by monkey-patching critic_evaluate so the test does
not require Azure / network access.

Run:
  python -m pytest tests/test_critic_flag_aggregation.py -v
or:
  python tests/test_critic_flag_aggregation.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "mini-swe-agent" / "src"))

from orchestra import critics
from orchestra.critics import (
    run_m_critics,
    filter_proposals,
    FLAG_NAMES,
    FLAG_PASS_THRESHOLD,
)


def _critic_response(flag_overrides=None, structural_failure=None):
    """Build a single critic response dict mimicking the parser's output."""
    if structural_failure:
        return {
            "score": 0,
            "sound": False,
            "reasoning": f"structural failure: {structural_failure}",
            "red_flags": [structural_failure],
            "flags": {},
        }
    flags = {n: True for n in FLAG_NAMES}
    flags.update(flag_overrides or {})
    failed = [n for n, v in flags.items() if not v]
    sound = not failed
    return {
        "score": 10 if sound else max(0, 5 - len(failed)),
        "sound": sound,
        "reasoning": "all flags pass" if sound else f"failed: {failed}",
        "red_flags": failed,
        "flags": flags,
    }


def _make_proposal(k_index=0, patch="diff --git a/x b/x\n+1\n"):
    return {
        "instance_id": "test-instance",
        "k_index": k_index,
        "model_patch": patch,
        "nonempty": bool(patch.strip()),
        "cost": 0.0,
        "steps": 1,
        "exit_status": "Submitted",
    }


def stub_critic_evaluate(scripted_responses):
    """Return a critic_evaluate stand-in that pops responses in order."""
    state = {"i": 0, "responses": list(scripted_responses)}

    def _stub(problem_statement, patch, model_name):
        i = state["i"]
        state["i"] += 1
        return state["responses"][i % len(state["responses"])]

    return _stub


def test_all_flags_pass_proposal_survives(monkeypatch):
    """Control: M=5, all critics vote every flag true → proposal survives."""
    proposal = _make_proposal()
    responses = [_critic_response() for _ in range(5)]
    monkeypatch.setattr(critics, "critic_evaluate", stub_critic_evaluate(responses))

    out = run_m_critics(proposal, "issue", m=5, model_name="stub")

    assert out["filtered"] is False, f"expected survives, got filtered: {out.get('filter_reason')}"
    assert out["critic_sound"] is True
    assert all(out["critic_flag_pass"].values())
    assert out["filter_reason"] == "all_flags_pass_threshold"
    print("PASS: control — all flags pass, proposal survives")


def test_one_flag_below_threshold_blocks_proposal(monkeypatch):
    """The headline case: 5 critics, all give 3/4 flags true but only 3/5
    agree on `addresses_root_cause`. 3/5 = 60% < 80% → flag fails gate →
    proposal filtered. Under the OLD score>=6 logic this would have
    averaged ~6.5 (sound) and passed. Under the new gate it fails."""
    proposal = _make_proposal(k_index=1)
    responses = [
        _critic_response({"addresses_root_cause": True}),
        _critic_response({"addresses_root_cause": True}),
        _critic_response({"addresses_root_cause": True}),
        _critic_response({"addresses_root_cause": False}),
        _critic_response({"addresses_root_cause": False}),
    ]
    monkeypatch.setattr(critics, "critic_evaluate", stub_critic_evaluate(responses))

    out = run_m_critics(proposal, "issue", m=5, model_name="stub")

    # Old logic would have given avg score >= 6 (3 critics with all flags
    # = score 10, 2 with one false = score 4). Avg = (10*3 + 4*2)/5 = 7.6
    # Old "score >= 6" → sound, would have passed Option-C structural-only
    # filter. New per-flag gate filters because addresses_root_cause = 3/5.
    assert out["critic_avg"] >= 6.0, f"sanity: old logic would call avg>=6, got {out['critic_avg']}"
    assert out["filtered"] is True, f"expected filtered, got survives: {out.get('filter_reason')}"
    assert out["critic_flag_votes"]["addresses_root_cause"] == 3
    assert out["critic_flag_pass"]["addresses_root_cause"] is False
    assert out["critic_flag_pass"]["preserves_existing_behavior"] is True  # 5/5
    assert "addresses_root_cause" in out["filter_reason"]
    print(f"PASS: headline — flag at 3/5 (60%) blocks proposal that old avg-score "
          f"({out['critic_avg']:.1f}) would have passed")


def test_threshold_boundary_4_of_5_passes(monkeypatch):
    """Boundary: 4/5 = exactly 0.8 → must pass (>= 0.8)."""
    proposal = _make_proposal(k_index=2)
    responses = [
        _critic_response({"no_unrelated_changes": True}),
        _critic_response({"no_unrelated_changes": True}),
        _critic_response({"no_unrelated_changes": True}),
        _critic_response({"no_unrelated_changes": True}),
        _critic_response({"no_unrelated_changes": False}),
    ]
    monkeypatch.setattr(critics, "critic_evaluate", stub_critic_evaluate(responses))

    out = run_m_critics(proposal, "issue", m=5, model_name="stub")
    assert out["critic_flag_votes"]["no_unrelated_changes"] == 4
    assert out["critic_flag_pass"]["no_unrelated_changes"] is True, \
        "4/5 = 0.8 should clear FLAG_PASS_THRESHOLD = 0.8"
    assert out["filtered"] is False
    print("PASS: boundary — 4/5 (= 80% exactly) clears threshold")


def test_all_unevaluable_passes_through(monkeypatch):
    """Safety net: if every critic returned a structural failure (parse
    error / refusal / empty / critic_error), the proposal passes through
    so the tournament can still try."""
    proposal = _make_proposal(k_index=3)
    responses = [
        _critic_response(structural_failure="parse_error"),
        _critic_response(structural_failure="empty_response"),
        _critic_response(structural_failure="refusal"),
        _critic_response(structural_failure="critic_error"),
        _critic_response(structural_failure="parse_error"),
    ]
    monkeypatch.setattr(critics, "critic_evaluate", stub_critic_evaluate(responses))

    out = run_m_critics(proposal, "issue", m=5, model_name="stub")
    assert out["filtered"] is False
    assert out["filter_reason"] == "all_unevaluable_pass_through"
    print("PASS: safety — all-unevaluable passes through to tournament")


def test_empty_patch_is_filtered(monkeypatch):
    """Empty patches are auto-filtered without consulting critics."""
    proposal = _make_proposal(k_index=4, patch="")
    out = run_m_critics(proposal, "issue", m=5, model_name="stub")
    assert out["filtered"] is True
    print("PASS: empty patch auto-filtered")


def main():
    """Manual runner for environments without pytest. Uses a simple
    monkeypatch shim."""
    class _MP:
        def __init__(self):
            self._undo = []
        def setattr(self, target, name, value):
            old = getattr(target, name)
            setattr(target, name, value)
            self._undo.append((target, name, old))
        def undo(self):
            for target, name, old in reversed(self._undo):
                setattr(target, name, old)

    tests = [
        test_all_flags_pass_proposal_survives,
        test_one_flag_below_threshold_blocks_proposal,
        test_threshold_boundary_4_of_5_passes,
        test_all_unevaluable_passes_through,
        test_empty_patch_is_filtered,
    ]
    for fn in tests:
        mp = _MP()
        try:
            fn(mp)
        finally:
            mp.undo()
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    main()
