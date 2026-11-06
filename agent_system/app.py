from __future__ import annotations

from pathlib import Path

import streamlit as st

from agent_system.commands import (
    Eval1v1Config,
    HumanLoopConfig,
    TrainConfig,
    VisualizeConfig,
    build_eval_1v1_command,
    build_human_loop_command,
    build_train_command,
    build_visualize_command,
    training_profile,
)
from agent_system.executor import run_command
from agent_system.llm import SiliconFlowClient
from agent_system.rag_adapter import (
    answer_with_rag,
    discover_knowledge_bases,
    resolve_knowledge_base_dir,
    validate_knowledge_base_dir,
)
from agent_system.result_analysis import analyze_result, infer_result_dir
from runner.experiment_tracking import SWANLAB_MODES
from agent_system.routing import build_task_graph
from agent_system.settings import AgentSettings, REPO_ROOT


st.set_page_config(page_title="航电智能 Agent", layout="wide")


def settings_and_client() -> tuple[AgentSettings, SiliconFlowClient | None]:
    settings = AgentSettings.load()
    client = SiliconFlowClient(settings) if settings.has_llm_credentials else None
    return settings, client


def run_spec_panel(key: str, spec, task_name: str, enable_analysis: bool = False) -> None:
    st.code(spec.preview(), language="powershell")
    if spec.expected_output_dir is not None:
        st.text_input("结果目录", str(spec.expected_output_dir), disabled=True, key=f"{key}_expected")
    if getattr(spec, "validation_errors", None):
        st.error("参数校验未通过：\n" + "\n".join(f"- {item}" for item in spec.validation_errors))

    col_run, col_hint = st.columns([1, 3])
    with col_run:
        should_run = st.button("执行", key=f"{key}_run", type="primary", disabled=bool(getattr(spec, "validation_errors", None)))
    with col_hint:
        st.write(f"工作目录：{spec.cwd}")

    if not should_run:
        return

    with st.status("正在执行", expanded=True) as status:
        result = run_command(spec)
        st.code(result.output or "(无输出)", language="text")
        if result.ok:
            status.update(label="执行完成", state="complete")
        else:
            status.update(label=f"执行失败：{result.returncode}", state="error")

    if enable_analysis and result.ok:
        settings, client = settings_and_client()
        result_dir = infer_result_dir(result.output, spec.expected_output_dir)
        if result_dir is not None and Path(result_dir).exists():
            analysis_path = analyze_result(Path(result_dir), task_name=task_name, client=client)
            st.success(f"分析已保存：{analysis_path}")
            st.markdown(analysis_path.read_text(encoding="utf-8"))
        else:
            st.warning("未识别到可分析的结果目录。")


def render_router() -> None:
    settings, client = settings_and_client()
    st.subheader("任务管理 Agent")
    user_input = st.text_area("用户输入", key="router_input", height=120)
    if st.button("识别流程", key="route_button", type="primary"):
        graph = build_task_graph(client)
        state = graph.invoke({"user_input": user_input})
        st.json(state.get("route_decision", {}))
    with st.expander("当前模型配置", expanded=False):
        st.write(
            {
                "model": settings.agent_chat_model,
                "base_url": settings.siliconflow_base_url,
                "has_api_key": settings.has_llm_credentials,
                "rag_project_root": str(settings.rag_project_root),
                "python_env": str(settings.python_env),
                "runtime_python": settings.runtime_python,
                "repo_root": str(REPO_ROOT),
            }
        )


def render_train() -> None:
    st.subheader("训练 Agent")
    profile_name = st.selectbox(
        "训练预设",
        ["hierarchy_no_weapon_selfplay", "hierarchy_shoot_selfplay", "heading_control"],
        format_func={
            "hierarchy_no_weapon_selfplay": "1v1 分层无导弹自博弈",
            "hierarchy_shoot_selfplay": "1v1 分层导弹自博弈",
            "heading_control": "单机低层航向控制",
        }.get,
    )
    default = training_profile(profile_name)
    with st.form("train_form"):
        env_name = st.selectbox("环境", ["SingleCombat", "SingleControl", "MultipleCombat"], index=["SingleCombat", "SingleControl", "MultipleCombat"].index(default.env_name))
        scenario_name = st.text_input("场景", default.scenario_name)
        experiment_name = st.text_input("实验名", default.experiment_name)
        seed = st.number_input("随机种子", min_value=0, value=default.seed, step=1)
        n_rollout_threads = st.number_input("rollout 线程数", min_value=1, value=default.n_rollout_threads, step=1)
        num_env_steps = st.text_input("环境步数", default.num_env_steps)
        lr = st.text_input("学习率", default.lr)
        buffer_size = st.number_input("buffer size", min_value=1, value=default.buffer_size, step=1)
        use_selfplay = st.checkbox("自博弈", value=default.use_selfplay)
        use_eval = st.checkbox("训练中评估", value=default.use_eval)
        use_prior = st.checkbox("导弹发射先验", value=default.use_prior)
        cuda = st.checkbox("CUDA", value=default.cuda)
        swanlab_mode = st.selectbox("SwanLab 记录模式", SWANLAB_MODES,
                                   index=SWANLAB_MODES.index(default.swanlab_mode))
        submitted = st.form_submit_button("生成训练命令")
    if submitted:
        config = TrainConfig(
            profile=profile_name,
            env_name=env_name,
            scenario_name=scenario_name,
            experiment_name=experiment_name,
            seed=int(seed),
            n_rollout_threads=int(n_rollout_threads),
            num_env_steps=str(num_env_steps),
            lr=str(lr),
            buffer_size=int(buffer_size),
            use_selfplay=use_selfplay,
            use_eval=use_eval,
            use_prior=use_prior,
            cuda=cuda,
            swanlab_mode=swanlab_mode,
        )
        st.session_state.train_spec = build_train_command(config)
    if "train_spec" in st.session_state:
        run_spec_panel("train", st.session_state.train_spec, "训练", enable_analysis=True)


def render_eval() -> None:
    st.subheader("评估 Agent")
    with st.form("eval_form"):
        eval_scenario_name = st.text_input("评估场景", "1v1/NoWeapon/Selfplay")
        actor_a_path = st.text_input("Actor A", "envs/JSBSim/model/actor_latest.pt")
        actor_a_scenario_name = st.text_input("Actor A 场景", "1v1/NoWeapon/Selfplay")
        actor_b_path = st.text_input("Actor B", "envs/JSBSim/model/actor_latest.pt")
        actor_b_scenario_name = st.text_input("Actor B 场景", "1v1/NoWeapon/Selfplay")
        lowlevel_actor_path = st.text_input("低层控制器", "envs/JSBSim/model/actor_heading.pt")
        experiment_name = st.text_input("实验名", "agent_1v1_eval")
        num_episodes = st.number_input("回合数", min_value=1, value=50, step=1)
        device = st.selectbox("设备", ["auto", "cpu", "cuda:0"], index=0)
        save_acmi = st.checkbox("保存 ACMI", value=True)
        acmi_episodes = st.text_input("ACMI 回合", "0")
        save_plots = st.checkbox("保存图表", value=True)
        output_dir = st.text_input("输出目录", "")
        submitted = st.form_submit_button("生成评估命令")
    if submitted:
        config = Eval1v1Config(
            eval_scenario_name=eval_scenario_name,
            actor_a_path=actor_a_path,
            actor_a_scenario_name=actor_a_scenario_name,
            actor_b_path=actor_b_path,
            actor_b_scenario_name=actor_b_scenario_name,
            lowlevel_actor_path=lowlevel_actor_path,
            experiment_name=experiment_name,
            num_episodes=int(num_episodes),
            device=device,
            save_acmi=save_acmi,
            acmi_episodes=acmi_episodes,
            save_plots=save_plots,
            output_dir=output_dir,
        )
        st.session_state.eval_spec = build_eval_1v1_command(config)
    if "eval_spec" in st.session_state:
        run_spec_panel("eval", st.session_state.eval_spec, "1v1 评估", enable_analysis=True)


def render_human() -> None:
    st.subheader("人机交互 Agent")
    mode = st.selectbox(
        "模式",
        ["free_fly", "no_weapon_1v1", "shoot_1v1"],
        index=0,
        format_func={
            "free_fly": "单人操控",
            "no_weapon_1v1": "无导弹 1v1",
            "shoot_1v1": "带导弹 1v1",
        }.get,
    )
    seed = st.number_input("随机种子", min_value=0, value=5, step=1, key="human_seed")
    cuda = st.checkbox("CUDA", value=True, key="human_cuda")
    if st.button("生成人机命令", key="human_build"):
        st.session_state.human_spec = build_human_loop_command(HumanLoopConfig(mode=mode or "free_fly", seed=int(seed), cuda=cuda))
    if "human_spec" in st.session_state:
        st.info("Tacview Advanced 实时遥测连接信息会在脚本输出中打印。")
        run_spec_panel("human", st.session_state.human_spec, "人机交互")


def render_visualize() -> None:
    st.subheader("可视化 Agent")
    with st.form("visualize_form"):
        model_dir = st.text_input("模型目录", "")
        env_name = st.selectbox("环境", ["SingleCombat", "SingleControl", "MultipleCombat"])
        scenario_name = st.text_input("场景", "1v1/NoWeapon/Selfplay")
        experiment_name = st.text_input("实验名", "agent_visualize")
        num_agents = st.number_input("智能体数", min_value=1, value=1, step=1)
        episode_length = st.number_input("回合长度", min_value=1, value=1000, step=50)
        use_selfplay = st.checkbox("自博弈模型池", value=False)
        render_index = st.text_input("己方模型索引", "latest")
        render_opponent_index = st.text_input("对手模型索引", "latest")
        submitted = st.form_submit_button("生成可视化命令")
    if submitted:
        config = VisualizeConfig(
            model_dir=model_dir,
            env_name=env_name,
            scenario_name=scenario_name,
            experiment_name=experiment_name,
            num_agents=int(num_agents),
            episode_length=int(episode_length),
            use_selfplay=use_selfplay,
            render_index=render_index,
            render_opponent_index=render_opponent_index,
        )
        try:
            st.session_state.visualize_spec = build_visualize_command(config)
        except ValueError as exc:
            st.error(str(exc))
    if "visualize_spec" in st.session_state:
        run_spec_panel("visualize", st.session_state.visualize_spec, "可视化")


def render_rag() -> None:
    st.subheader("RAG Agent")
    settings, _ = settings_and_client()
    knowledge_bases = discover_knowledge_bases(settings)
    option_labels = [item.label for item in knowledge_bases] + ["手动输入知识库路径"]
    selected_label = st.selectbox("知识库", option_labels, key="rag_kb_label") if option_labels else "手动输入知识库路径"

    selected_option = next((item for item in knowledge_bases if item.label == selected_label), None)
    if selected_option is None:
        default_kb_dir = resolve_knowledge_base_dir(None, settings)
        knowledge_base_dir = st.text_input("知识库目录", str(default_kb_dir), key="rag_kb_custom")
    else:
        knowledge_base_dir = st.text_input("知识库目录", str(selected_option.path), disabled=True, key="rag_kb_selected_path")

    resolved_kb_dir = resolve_knowledge_base_dir(knowledge_base_dir, settings)
    kb_errors = validate_knowledge_base_dir(resolved_kb_dir)
    if kb_errors:
        st.error("知识库校验未通过：\n" + "\n".join(f"- {item}" for item in kb_errors))
    else:
        st.caption(f"当前知识库：{resolved_kb_dir}")

    question = st.text_area("问题", key="rag_question", height=120)
    if st.button("回答", key="rag_answer", type="primary", disabled=bool(kb_errors)):
        response = None
        with st.status("正在检索生成", expanded=True) as status:
            try:
                response = answer_with_rag(question, settings=settings, knowledge_base_dir=resolved_kb_dir)
                status.update(label="回答完成", state="complete", expanded=False)
            except Exception as exc:
                status.update(label="回答失败", state="error")
                st.error(str(exc))
        if response is not None:
            st.markdown(response.answer)
            with st.expander("来源", expanded=True):
                st.text(response.sources)
            with st.expander("检索 JSON", expanded=False):
                st.code(response.retrieval_json, language="json")


def main() -> None:
    st.title("面向航电智能化应用的分层强化学习自主决策 Agent 系统")
    tabs = st.tabs(["任务管理", "训练", "评估", "人机交互", "可视化", "RAG 问答"])
    with tabs[0]:
        render_router()
    with tabs[1]:
        render_train()
    with tabs[2]:
        render_eval()
    with tabs[3]:
        render_human()
    with tabs[4]:
        render_visualize()
    with tabs[5]:
        render_rag()


if __name__ == "__main__":
    main()
