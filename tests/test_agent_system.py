from __future__ import annotations

import json
import sys
from types import SimpleNamespace

from agent_system.commands import (
    Eval1v1Config,
    HumanLoopConfig,
    TacticalDemoConfig,
    TrainConfig,
    VisualizeConfig,
    _path,
    build_eval_1v1_command,
    build_human_loop_command,
    build_tactical_demo_command,
    build_train_command,
    build_visualize_command,
)
from agent_system.executor import run_command
from agent_system.result_analysis import infer_result_dir, load_result_brief
from agent_system.routing import keyword_route, parse_route_json
from agent_system.settings import AgentSettings
from agent_system.settings import DEFAULT_TACTICAL_ACTOR_PATH
from agent_system.settings import REPO_ROOT
from agent_system.tactical_actions import ACTION_BY_ID, action_name_matches
from agent_system.tactical_parser import (
    TACTICAL_SYSTEM_PROMPT,
    keyword_parse_tactical_instruction,
    parse_tactical_instruction,
    parse_tactical_json,
)
from agent_system.tactical_policy import resolve_actor_checkpoint_path
from agent_system.tactical_safety import TacticalSafetyState, apply_tactical_safety
from agent_system.tactical_scheduler import TacticalActionScheduler
from experiments import agent_llm_fusion_experiments as fusion_experiments
from experiments.agent_llm_fusion_experiments import (
    build_instruction_dataset,
    build_safety_cases,
    evaluate_safety_cases,
    parse_manual_schedule,
)


def test_keyword_route_eval() -> None:
    decision = keyword_route("帮我评估两个 actor 的 1v1 胜率")
    assert decision.route == "evaluate_1v1"


def test_parse_route_json_strips_markdown_like_text() -> None:
    decision = parse_route_json('```json\n{"route":"train","confidence":0.9,"reason":"训练","extracted":{}}\n```')
    assert decision.route == "train"
    assert decision.confidence == 0.9


def test_build_train_command_uses_argument_list() -> None:
    spec = build_train_command(
        TrainConfig(
            scenario_name="1v1/NoWeapon/HierarchySelfplay",
            experiment_name="unit_train",
            cuda=False,
            swanlab_mode="disabled",
        )
    )
    assert spec.command[0] == AgentSettings.load().runtime_python
    assert "--scenario-name" in spec.command
    assert "1v1/NoWeapon/HierarchySelfplay" in spec.command
    assert "--clip-param" in spec.command
    assert "--clip-params" not in spec.command
    assert spec.command[spec.command.index("--swanlab-mode") + 1] == "disabled"
    default_command = build_train_command(TrainConfig()).command
    assert default_command[default_command.index("--swanlab-mode") + 1] == "offline"


def test_build_eval_command_exposes_cli_contract() -> None:
    spec = build_eval_1v1_command(
        Eval1v1Config(
            actor_a_path="envs/JSBSim/model/actor_latest.pt",
            actor_b_path="envs/JSBSim/model/actor_latest.pt",
            num_episodes=1,
            save_acmi=False,
        )
    )
    assert "--eval-scenario-name" in spec.command
    assert "--actor-a-scenario-name" in spec.command
    assert "--actor-b-scenario-name" in spec.command
    assert "--save-acmi" in spec.command
    assert "false" in spec.command
    assert spec.validation_errors == []


def test_train_analysis_uses_reported_run_directory(tmp_path) -> None:
    run_dir = tmp_path / "run2"
    run_dir.mkdir()
    (run_dir / "training-summary.json").write_text(
        json.dumps({"step": 32, "metrics": {"value_loss": 0.25}}), encoding="utf-8",
    )
    assert infer_result_dir(f"输出目录: {run_dir}", tmp_path) == run_dir
    assert '"value_loss": 0.25' in load_result_brief(run_dir)


def test_path_normalization_accepts_slashes_quotes_and_reports_missing() -> None:
    normalized = _path('"envs\\JSBSim/model\\actor_heading.pt"')
    assert normalized == (REPO_ROOT / "envs" / "JSBSim" / "model" / "actor_heading.pt").resolve()

    spec = build_eval_1v1_command(
        Eval1v1Config(
            actor_a_path="scripts\\results/missing/filesl/actor_latest.pt",
            actor_b_path="envs/JSBSim/model/actor_latest.pt",
            lowlevel_actor_path="envs\\JSBSim/model\\actor_heading.pt",
        )
    )
    assert any("Actor A 文件不存在" in item for item in spec.validation_errors)


def test_build_human_and_visualize_commands() -> None:
    human = build_human_loop_command(HumanLoopConfig(mode="shoot_1v1", cuda=False))
    assert "human_shoot_1v1.py" in human.preview()
    assert "--use-prior" in human.command

    visualize = build_visualize_command(VisualizeConfig(model_dir="scripts/results/demo"))
    assert "render_jsbsim.py" in visualize.preview()
    assert "--model-dir" in visualize.command


def test_tactical_action_catalog_has_12_named_actions() -> None:
    assert len(ACTION_BY_ID) == 12
    assert ACTION_BY_ID[0].code == "PURE_PURSUIT"
    assert ACTION_BY_ID[11].chinese_name == "低悠悠"
    assert action_name_matches(1, "LEAD_PURSUIT")
    assert not action_name_matches(1, "LAG_PURSUIT")


def test_tactical_keyword_parser_maps_chinese_phrases() -> None:
    assert keyword_parse_tactical_instruction("追击敌机").action_id == 0
    assert keyword_parse_tactical_instruction("抢占敌机前方").action_id == 1
    assert keyword_parse_tactical_instruction("减速避免冲过头").action_id == 7
    assert keyword_parse_tactical_instruction("向左防御转弯").action_id == 8
    assert not keyword_parse_tactical_instruction("先提前量追击超过他再爬升占位").valid
    assert not keyword_parse_tactical_instruction("先搜索目标再规划复杂任务").valid


def test_tactical_llm_json_validation() -> None:
    valid = parse_tactical_json(
        '```json\n{"scene":"1v1","agent_id":"A0100","tactical_action_id":1,'
        '"tactical_action_name":"LEAD_PURSUIT","reason":"抢占前方"}\n```'
    )
    assert valid.valid
    assert valid.action_id == 1

    mismatch = parse_tactical_json(
        '{"scene":"1v1","agent_id":"A0100","tactical_action_id":1,'
        '"tactical_action_name":"LAG_PURSUIT","reason":"错误"}'
    )
    assert not mismatch.valid

    out_of_range = parse_tactical_json(
        '{"scene":"1v1","agent_id":"A0100","tactical_action_id":99,'
        '"tactical_action_name":"UNKNOWN","reason":"错误"}'
    )
    assert not out_of_range.valid

    invalid_instruction = parse_tactical_json(
        '{"scene":"1v1","agent_id":"A0100","tactical_action_id":-1,'
        '"tactical_action_name":"INVALID","reason":"与空战战术无关，拒绝执行。"}'
    )
    assert not invalid_instruction.valid
    assert invalid_instruction.action_id is None
    assert invalid_instruction.tactical_action_name == "INVALID"
    assert "拒绝执行" in invalid_instruction.reason


def test_tactical_llm_prompt_describes_invalid_instruction_handling() -> None:
    assert "无效指令" in TACTICAL_SYSTEM_PROMPT
    assert 'tactical_action_id": -1' in TACTICAL_SYSTEM_PROMPT
    assert 'tactical_action_name": "INVALID"' in TACTICAL_SYSTEM_PROMPT
    assert "不要把无关指令或超范围任务强行映射为 3" in TACTICAL_SYSTEM_PROMPT


def test_tactical_llm_parser_disables_thinking_for_json() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.kwargs = {}

        def chat(self, messages, **kwargs):
            self.kwargs = kwargs
            return json.dumps(
                {
                    "scene": "1v1",
                    "agent_id": "A0100",
                    "tactical_action_id": 4,
                    "tactical_action_name": "CLIMB_POSITION",
                    "reason": "用户要求爬升占位。",
                },
                ensure_ascii=False,
            )

    client = FakeClient()
    decision = parse_tactical_instruction("爬升占位", client=client)

    assert decision.valid
    assert decision.source == "llm"
    assert decision.action_id == 4
    assert client.kwargs["enable_thinking"] is False


def test_tactical_llm_invalid_decision_does_not_keyword_fallback() -> None:
    class FakeClient:
        def chat(self, messages, **kwargs):
            return json.dumps(
                {
                    "scene": "1v1",
                    "agent_id": "A0100",
                    "tactical_action_id": -1,
                    "tactical_action_name": "INVALID",
                    "reason": "该输入是复杂任务链，当前版本只支持单个高层战术动作。",
                },
                ensure_ascii=False,
            )

    decision = parse_tactical_instruction("先提前量追击超过他再爬升占位", client=FakeClient())

    assert not decision.valid
    assert decision.action_id is None
    assert decision.tactical_action_name == "INVALID"


def test_tactical_scheduler_uses_actor_fallback_then_manual_hold() -> None:
    scheduler = TacticalActionScheduler(hold_steps=2)
    first = scheduler.select(actor_action_id=6)
    assert first.source == "actor_fallback"
    assert first.action_id == 6

    manual = keyword_parse_tactical_instruction("爬升占位")
    second = scheduler.select(actor_action_id=6, manual_decision=manual)
    third = scheduler.select(actor_action_id=6)
    fourth = scheduler.select(actor_action_id=6)

    assert second.source == "manual"
    assert second.action_id == 4
    assert third.source == "manual"
    assert third.action_id == 4
    assert fourth.source == "actor_fallback"
    assert fourth.action_id == 6


def test_tactical_safety_overrides_risky_actions() -> None:
    low_altitude = TacticalSafetyState(altitude_m=3200.0, altitude_limit_m=2500.0)
    low_result = apply_tactical_safety(5, state=low_altitude, fallback_action_id=0)
    assert low_result.action_id == 4
    assert low_result.overridden

    close_fast = TacticalSafetyState(distance_m=900.0, speed_mps=300.0)
    close_result = apply_tactical_safety(1, state=close_fast, fallback_action_id=0)
    assert close_result.action_id == 7
    assert close_result.overridden

    invalid = apply_tactical_safety(99, fallback_action_id=2)
    assert invalid.action_id == 2
    assert invalid.overridden


def test_build_tactical_demo_command_accepts_run_directory(tmp_path) -> None:
    actor_dir = tmp_path / "run1"
    actor_dir.mkdir()
    checkpoint = actor_dir / "actor_latest.pt"
    checkpoint.touch()
    assert TacticalDemoConfig().actor_path == str(DEFAULT_TACTICAL_ACTOR_PATH)
    spec = build_tactical_demo_command(
        TacticalDemoConfig(
            actor_path=str(actor_dir),
            max_steps=5,
            render_mode="none",
            disable_llm=True,
        )
    )
    assert "agent_tactical_1v1_demo.py" in spec.preview()
    assert "--actor-path" in spec.command
    assert "--enemy-path" in spec.command
    assert "--enemy-action" not in spec.command
    assert "--render-mode" in spec.command
    assert "none" in spec.command
    assert "--status-interval" in spec.command
    assert "--step-sleep" in spec.command
    assert "--disable-llm" in spec.command
    assert spec.validation_errors == []
    assert spec.command.count(str(checkpoint)) == 2

    fixed_enemy = build_tactical_demo_command(TacticalDemoConfig(
        actor_path=str(actor_dir), enemy_action="PURE_PURSUIT", render_mode="none",
    ))
    assert "--enemy-action" in fixed_enemy.command
    assert "PURE_PURSUIT" in fixed_enemy.command
    assert fixed_enemy.validation_errors == []

    missing_enemy = build_tactical_demo_command(
        TacticalDemoConfig(actor_path=str(actor_dir), enemy_path="missing_enemy.pt", render_mode="none")
    )
    assert any("Enemy tactical actor" in item for item in missing_enemy.validation_errors)

    invalid_enemy_action = build_tactical_demo_command(
        TacticalDemoConfig(actor_path=str(actor_dir), enemy_action="NOT_A_TACTIC", render_mode="none")
    )
    assert any("enemy-action" in item for item in invalid_enemy_action.validation_errors)

    missing = build_tactical_demo_command(TacticalDemoConfig(actor_path="missing_actor.pt"))
    assert any("Tactical actor 文件不存在" in item for item in missing.validation_errors)


def test_resolve_actor_checkpoint_accepts_files_directory(tmp_path) -> None:
    actor_dir = tmp_path / "files"
    actor_dir.mkdir()
    assert resolve_actor_checkpoint_path(actor_dir) == actor_dir / "actor_latest.pt"


def test_agent_llm_experiment_manual_schedule_parser() -> None:
    schedule = parse_manual_schedule("30:爬升占位;90:减速避免冲过头;150:向左防御转弯")
    assert schedule == {
        30: "爬升占位",
        90: "减速避免冲过头",
        150: "向左防御转弯",
    }
    assert parse_manual_schedule("") == {}


def test_agent_llm_experiment_dataset_covers_12_actions() -> None:
    cases = build_instruction_dataset()
    expected_ids = {case.expected_action_id for case in cases if case.expected_action_id >= 0}
    invalid_cases = [case for case in cases if case.expected_action_id == -1]
    assert expected_ids == set(ACTION_BY_ID)
    assert len(invalid_cases) == 12

    for action_id in ACTION_BY_ID:
        action_cases = [case for case in cases if case.expected_action_id == action_id]
        assert len(action_cases) == 10
        assert any(any(ch.isascii() and ch.isalpha() for ch in case.instruction) for case in action_cases)


def test_agent_llm_experiment_keyword_dataset_is_unambiguous() -> None:
    rows, summary = fusion_experiments.evaluate_instruction_cases(
        build_instruction_dataset(),
        parser_mode="keyword",
        agent_id="A0100",
    )
    assert summary["base_case_total"] == 132
    assert summary["repeat_count"] == 1
    assert summary["correct_count"] == summary["total"]
    assert [row for row in rows if not row["correct"]] == []


def test_agent_llm_experiment_llm_mode_repeats_each_case(monkeypatch) -> None:
    monkeypatch.setattr(fusion_experiments, "make_llm_client", lambda parser_mode: None)
    rows, summary = fusion_experiments.evaluate_instruction_cases(
        build_instruction_dataset()[:1],
        parser_mode="llm_fallback",
        agent_id="A0100",
    )
    assert len(rows) == 3
    assert summary["base_case_total"] == 1
    assert summary["repeat_count"] == 3
    assert {row["repeat_index"] for row in rows} == {1, 2, 3}


def test_agent_llm_experiment_safety_cases_cover_reasons() -> None:
    cases = build_safety_cases()
    rows, summary = evaluate_safety_cases(cases)
    categories = {row["reason_category"] for row in rows}
    by_case = {row["case_id"]: row for row in rows}

    assert len(cases) == 50
    assert summary["total"] == 50
    assert {"low_altitude", "close_fast", "bad_posture", "invalid_action", "none"}.issubset(categories)
    assert summary["overridden_count"] >= 20
    assert by_case["low_dive_at_margin"]["final_action_id"] == 4
    assert by_case["low_dive_above_margin"]["overridden"] is False
    assert by_case["close_fast_pure_at_distance"]["final_action_id"] == 7
    assert by_case["close_fast_lead_speed_below"]["overridden"] is False
    assert by_case["bad_posture_pure"]["final_action_id"] == 3
    assert by_case["bad_posture_ta_equal"]["overridden"] is False
    assert by_case["invalid_99_fallback_lag"]["final_action_id"] == 2


def test_infer_result_dir_from_json_output() -> None:
    result_dir = infer_result_dir('{"output_dir": "E:/clone/demo1/experiments/results/x"}')
    assert result_dir is not None
    assert result_dir.as_posix().endswith("/experiments/results/x")


def test_run_command_without_shell() -> None:
    from agent_system.commands import CommandSpec

    result = run_command(CommandSpec("unit", [sys.executable, "-c", "print('ok')"]))
    assert result.ok
    assert "ok" in result.output


def test_rag_adapter_uses_runtime_python(monkeypatch, tmp_path) -> None:
    from agent_system import rag_adapter

    rag_root = tmp_path / "fake_rag_root"
    kb_dir = rag_root / "vector_store" / "faiss"
    kb_dir.mkdir(parents=True, exist_ok=True)
    (kb_dir / "index.faiss").write_bytes(b"fake")
    (kb_dir / "documents.jsonl").write_text("{}", encoding="utf-8")
    (kb_dir / "manifest.json").write_text(json.dumps({"dimension": 4096}), encoding="utf-8")
    settings = AgentSettings(siliconflow_api_key="", rag_project_root=rag_root)
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"answer": "ok", "sources": "", "retrieval_json": "{}"}),
            stderr="",
        )

    monkeypatch.setattr(rag_adapter.subprocess, "run", fake_run)
    response = rag_adapter.answer_with_rag("测试问题", settings=settings, knowledge_base_dir=kb_dir)

    assert response.answer == "ok"
    assert captured["command"][0] == settings.runtime_python
    assert captured["command"][1:] == ["-m", "agent_system.rag_worker"]
    assert captured["kwargs"]["shell"] is False
    payload = json.loads(captured["kwargs"]["input"])
    assert payload["knowledge_base_dir"] == str(kb_dir.resolve())
    assert payload["embedding_dimensions"] == 4096
    assert captured["kwargs"]["env"]["AGENTIC_RAG_VECTOR_STORE_DIR"] == str(kb_dir.resolve())
    assert captured["kwargs"]["env"]["SILICONFLOW_EMBEDDING_DIMENSIONS"] == "4096"


def test_rag_knowledge_base_discovery(tmp_path) -> None:
    from agent_system.rag_adapter import discover_knowledge_bases, resolve_knowledge_base_dir, validate_knowledge_base_dir

    rag_root = tmp_path / "rag"
    kb_dir = rag_root / "vector_store" / "faiss"
    kb_dir.mkdir(parents=True)
    (kb_dir / "index.faiss").write_bytes(b"fake")
    (kb_dir / "documents.jsonl").write_text("{}", encoding="utf-8")
    (kb_dir / "manifest.json").write_text(
        json.dumps({"doc_count": 3, "dimension": 4096}, ensure_ascii=False),
        encoding="utf-8",
    )
    settings = AgentSettings(siliconflow_api_key="", rag_project_root=rag_root)

    assert resolve_knowledge_base_dir("vector_store/faiss", settings) == kb_dir.resolve()
    assert validate_knowledge_base_dir(kb_dir) == []
    options = discover_knowledge_bases(settings)
    assert len(options) == 1
    assert options[0].path == kb_dir.resolve()
    assert "3 chunks" in options[0].label
