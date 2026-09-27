"""Streamlit entry point for the frozen-model electrolyte design demonstration.

Run from the repository root:
    streamlit run apps/electrolyte_designer/app.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# ``streamlit run apps/electrolyte_designer/app.py`` executes the file with
# its package directory on ``sys.path`` rather than the repository root.
# Add the root explicitly so the app works from an IDE terminal or any cwd.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from apps.electrolyte_designer.services.catalog_service import (
    catalogue_summary,
    filter_catalogue,
    automatic_selection,
    molecule_png,
    presentation_rows,
    property_constraints,
    selection_csv,
)
from apps.electrolyte_designer.services.data_access import (
    default_paths,
    load_generation_assets,
    load_metric_rows,
)
from apps.electrolyte_designer.services.generation_service import card_json, rank_frozen_samples
from apps.electrolyte_designer.services.predictor_adapter import FrozenPredictorAdapter


st.set_page_config(page_title="Electrolyte Designer", page_icon="⚗️", layout="wide")
st.markdown(
    """
    <style>
      .stApp { background: #f4f7fb; color: #102a56; }
      [data-testid="stSidebar"] { background: linear-gradient(180deg, #092d61, #061d42); }
      [data-testid="stSidebar"] * { color: #eef6ff !important; }
      [data-testid="stSidebar"] [data-baseweb="radio"] label {
        border-radius: 8px; padding: 5px 8px; margin: 2px 0;
      }
      [data-testid="stSidebar"] [data-baseweb="radio"] label:has(input:checked) {
        background: #0d6efd;
      }
      .block-container { max-width: 1480px; padding-top: 1.4rem; padding-bottom: 2.5rem; }
      .demo-hero {
        background: linear-gradient(112deg, #ffffff 0%, #f3f8ff 72%, #edf5ff 100%);
        border: 1px solid #d7e7fa; border-radius: 16px; padding: 1.05rem 1.25rem .95rem;
        margin: 0 0 1.1rem; box-shadow: 0 4px 18px rgba(28, 72, 130, .07);
      }
      .demo-kicker { color: #2474d8; font-size: .72rem; font-weight: 800; letter-spacing: .095em; text-transform: uppercase; }
      .demo-hero h1 { margin: .08rem 0 .2rem !important; font-size: 2rem !important; }
      .demo-hero p { color: #557193; margin: 0; font-size: .94rem; }
      .section-label {
        color: #0b3c78; font-size: 1rem; font-weight: 760; margin: .2rem 0 .55rem;
        padding-left: .65rem; border-left: 4px solid #2583e8;
      }
      h1, h2, h3 { color: #0a2e61 !important; font-weight: 750 !important; }
      h1 { letter-spacing: -0.035em; }
      [data-testid="stMetric"], [data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff; border: 1px solid #dce7f5; border-radius: 13px; box-shadow: 0 3px 14px rgba(28, 72, 130, .07);
      }
      [data-testid="stMetric"] { padding: .75rem .95rem; }
      [data-testid="stMetricLabel"] { color: #42658f !important; }
      [data-testid="stMetricValue"] { color: #075dd7 !important; }
      [data-testid="stMetricDelta"] { font-weight: 650; }
      [data-testid="stHorizontalBlock"] > div:has([data-testid="stMetric"]) [data-testid="stMetric"] {
        min-height: 105px;
      }
      .stButton > button, .stDownloadButton > button {
        border: 0; border-radius: 8px; color: white; background: linear-gradient(90deg, #0969da, #3b5bdb);
        font-weight: 650; box-shadow: 0 3px 8px rgba(9, 105, 218, .20);
      }
      .stButton > button:hover, .stDownloadButton > button:hover { background: #0755b7; color: white; }
      [data-testid="stDataFrame"], [data-testid="stImage"] img {
        border: 1px solid #d8e5f4; border-radius: 10px; overflow: hidden; background: #ffffff;
      }
      [data-testid="stImage"] { background: #ffffff; border-radius: 12px; }
      [data-testid="stExpander"] { border: 1px solid #dce7f5; border-radius: 10px; background: #ffffff; }
      [data-baseweb="tab-list"] { gap: 9px; }
      [data-baseweb="tab"] { background: #eaf2ff; border-radius: 8px 8px 0 0; padding: 7px 15px; }
      [aria-selected="true"] { color: #075dd7 !important; font-weight: 750 !important; }
      div[data-testid="stAlert"] { border-radius: 10px; border: 1px solid #cfe1fb; }
    </style>
    """,
    unsafe_allow_html=True,
)


def fragment_topology_png(smiles: str) -> bytes | None:
    """Draw the fragment-labelled structure used by the recommendation view."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        from formula_design.fragment import build_fragment_graph

        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return None
        fragment_graph = build_fragment_graph(molecule)
        drawer = Draw.MolDraw2DCairo(440, 260)
        options = drawer.drawOptions()
        for atom_index, fragment_id in enumerate(fragment_graph.atom_to_fragment):
            atom = molecule.GetAtomWithIdx(atom_index)
            options.atomLabels[atom_index] = f"{atom.GetSymbol()}:{fragment_id + 1}"
        cut_bonds = list(fragment_graph.cut_bond_indices)
        drawer.DrawMolecule(
            molecule, [], cut_bonds, None,
            {bond: (0.9, 0.15, 0.15) for bond in cut_bonds},
        )
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def cached_metrics():
    return load_metric_rows(default_paths())


@st.cache_data(show_spinner="Loading the screened GDB17 catalogue...")
def cached_catalogue(root: str, path: str):
    return catalogue_summary(root, path)


@st.cache_data(show_spinner=False)
def cached_generation_assets():
    return load_generation_assets(default_paths())


@st.cache_resource(show_spinner="Loading frozen ordinary and explanation heads...")
def cached_oracle(config_path: str, checkpoint_path: str):
    return FrozenPredictorAdapter(Path(config_path), Path(checkpoint_path))


def page_hero(kicker: str, title: str, description: str) -> None:
    """A compact visual header shared by the three customer-facing pages."""
    st.markdown(
        f"""<div class=\"demo-hero\"><div class=\"demo-kicker\">{kicker}</div>
        <h1>{title}</h1><p>{description}</p></div>""",
        unsafe_allow_html=True,
    )


def section_label(title: str) -> None:
    st.markdown(f'<div class="section-label">{title}</div>', unsafe_allow_html=True)


def style_axis(axis) -> None:
    """Apply the quiet blue, presentation-oriented chart language used in the UI."""
    axis.set_facecolor("#ffffff")
    axis.figure.set_facecolor("#ffffff")
    axis.grid(axis="y", color="#dce7f5", linewidth=.75, alpha=.85)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#c7d8ec")
    axis.tick_params(colors="#42658f", labelsize=9)


def metric_cards(rows: list[dict]) -> None:
    columns = st.columns(4)
    for column, row in zip(columns, rows):
        with column:
            st.metric(f"{row['task']} · {row['head']}", f"R² {row['r2']:.3f}")
            st.caption(f"MAE {row['mae']:.4g} {row['unit']} · RMSE {row['rmse']:.4g} {row['unit']}")


def metrics_figure(rows: list[dict]):
    frame = pd.DataFrame(rows)
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    colors = {"Main": "#2563eb", "Explain": "#7c3aed"}
    for axis, metric, title in zip(axes, ("r2", "mae", "rmse"), ("R² ↑", "MAE ↓", "RMSE ↓")):
        groups = ["Conductivity", "Anion ratio"]
        positions = range(len(groups))
        width = 0.34
        for offset, head in ((-width / 2, "Main"), (width / 2, "Explain")):
            values = [
                float(frame[(frame["task"] == task) & (frame["head"] == head)][metric].iloc[0])
                for task in groups
            ]
            axis.bar([position + offset for position in positions], values, width,
                     label=head, color=colors[head])
        axis.set_title(title)
        axis.set_xticks(list(positions), groups, fontsize=9)
        axis.grid(axis="y", alpha=.25)
    axes[0].legend(frameon=False, fontsize=9)
    return figure


def show_model_evaluation() -> None:
    st.title("模型可信度")
    st.caption("Held-out 数据上的主预测头与解释头。新生成配方没有实验真值，因此不在该页计算 R²、MAE 或 RMSE。")
    try:
        rows = cached_metrics()
    except FileNotFoundError as error:
        st.error(str(error))
        return
    metric_cards(rows)
    st.pyplot(metrics_figure(rows), clear_figure=True)
    frame = pd.DataFrame(rows)
    frame["ΔMAE vs Main"] = frame.groupby("task")["mae"].transform(lambda values: values - values.iloc[0])
    st.dataframe(
        frame[["task", "head", "n", "r2", "mae", "rmse", "unit", "ΔMAE vs Main"]],
        width="stretch",
        hide_index=True,
        column_config={
            "r2": st.column_config.NumberColumn("R²", format="%.4f"),
            "mae": st.column_config.NumberColumn("MAE", format="%.5f"),
            "rmse": st.column_config.NumberColumn("RMSE", format="%.5f"),
            "ΔMAE vs Main": st.column_config.NumberColumn("ΔMAE", format="%+.5f"),
        },
    )
    st.info("解释头应同时满足实验误差可接受和与主头的一致性。贡献图用于解释模型判断，不等同于化学因果或真实相互作用能。")


def _vocabulary():
    _, solvents, salts = cached_generation_assets()
    return solvents, salts


def _chart_style(chart, *, width: int = 760, height: int = 300):
    """Presentation-oriented Vega-Lite styling; browser fonts render Chinese labels."""
    return (
        chart.properties(width=width, height=height)
        .configure_view(stroke=None, fill="#ffffff")
        .configure_axis(
            labelColor="#42658f", titleColor="#23466f", domainColor="#c7d8ec",
            gridColor="#dce7f5", tickColor="#c7d8ec", labelFontSize=11, titleFontSize=12,
        )
        .configure_title(color="#0a2e61", fontSize=15, fontWeight=700, anchor="start", offset=14)
    )


def contribution_figure(explain: dict):
    """A Chinese-labelled waterfall of terms with an explicit, interpretable meaning."""
    labels = ["条件基线", "溶剂物性项", "片段主项", "片段对项"]
    values = [
        explain.get("condition_reference", 0.0), explain.get("property_sum", 0.0),
        explain.get("main_sum", 0.0), explain.get("interaction_sum", 0.0),
    ]
    rows, running = [], 0.0
    for label, value in zip(labels, values):
        start = running
        running += value
        rows.append({
            "项目": label, "下界": min(start, running), "上界": max(start, running),
            "贡献": value, "数值": f"{value:+.3f}", "方向": "正向" if value >= 0 else "负向",
        })
    rows.append({
        "项目": "可解释项合计", "下界": min(0.0, running), "上界": max(0.0, running),
        "贡献": running, "数值": f"{running:+.3f}", "方向": "合计",
    })
    frame = pd.DataFrame(rows)
    order = labels + ["可解释项合计"]
    color = alt.Color(
        "方向:N", title=None,
        scale=alt.Scale(domain=["正向", "负向", "合计"], range=["#1677e8", "#f0646b", "#5b5bd6"]),
    )
    bars = alt.Chart(frame).mark_bar(size=42, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("项目:N", sort=order, title="解释组成", axis=alt.Axis(labelAngle=0, labelLimit=120)),
        y=alt.Y("下界:Q", title="归一化贡献"), y2="上界:Q", color=color,
        tooltip=[alt.Tooltip("项目:N"), alt.Tooltip("贡献:Q", format="+.4f", title="归一化贡献")],
    )
    labels_layer = alt.Chart(frame).mark_text(dy=-10, color="#23466f", fontWeight=600).encode(
        x=alt.X("项目:N", sort=order), y=alt.Y("上界:Q"), text="数值:N"
    )
    zero = alt.Chart(pd.DataFrame({"零线": [0]})).mark_rule(color="#526579", strokeWidth=1).encode(y="零线:Q")
    return _chart_style(
        bars + labels_layer + zero,
        width=800, height=330,
    ).properties(title="电导率解释分解")


def molecule_effect_figure(rows: list[dict]):
    display_rows = [
        {"分子": row["molecule_name"], "归一化贡献": row["total_property_effect_normalized"],
         "数值": f"{row['total_property_effect_normalized']:+.3f}",
         "方向": "正向" if row["total_property_effect_normalized"] >= 0 else "负向"}
        for row in rows
    ]
    return signed_effect_figure(
        display_rows, label_key="分子", value_key="归一化贡献", title="分子层面的物性贡献",
        category_title="配方分子",
    )


def signed_effect_figure(rows: list[dict], *, label_key: str, value_key: str, title: str, category_title: str):
    """Shared signed-bar view for property and fragment-pair terms."""
    frame = pd.DataFrame(rows)
    frame["类别"] = frame[label_key].astype(str)
    frame["归一化贡献"] = frame[value_key].astype(float)
    frame["数值"] = frame["归一化贡献"].map(lambda value: f"{value:+.3f}")
    frame["方向"] = frame["归一化贡献"].map(lambda value: "正向" if value >= 0 else "负向")
    order = frame["类别"].tolist()
    bars = alt.Chart(frame).mark_bar(size=22, cornerRadiusEnd=4).encode(
        x=alt.X("归一化贡献:Q", title="归一化贡献"),
        y=alt.Y("类别:N", sort=order, title=category_title),
        color=alt.Color(
            "方向:N", title=None,
            scale=alt.Scale(domain=["正向", "负向"], range=["#20b8a5", "#f0646b"]),
        ),
        tooltip=[alt.Tooltip("类别:N", title=category_title), alt.Tooltip("归一化贡献:Q", format="+.4f")],
    )
    text = alt.Chart(frame).mark_text(align="left", dx=6, color="#23466f", fontWeight=600).encode(
        x="归一化贡献:Q", y=alt.Y("类别:N", sort=order), text="数值:N"
    )
    zero = alt.Chart(pd.DataFrame({"零线": [0]})).mark_rule(color="#526579", strokeWidth=1).encode(x="零线:Q")
    return _chart_style(
        bars + text + zero, width=800, height=max(180, 42 * len(frame))
    ).properties(title=title)


def property_effect_figure(properties: dict):
    rows = [{"name": name, "effect": value} for name, value in properties.items()]
    return signed_effect_figure(rows, label_key="name", value_key="effect", title="溶剂物性贡献", category_title="物性项")


def fragment_pair_figure(rows: list[dict]):
    display_rows = []
    for row in rows[:10]:
        left, right = str(row.get("left", "fragment A")), str(row.get("right", "fragment B"))
        display_rows.append({"pair": f"{left}  ×  {right}", "effect": row["effect_normalized"]})
    return signed_effect_figure(display_rows, label_key="pair", value_key="effect", title="主要片段对交互贡献", category_title="片段对")


def motif_coverage_figure(counts: pd.Series):
    """Chinese-labelled coverage chart for the molecule-recommendation evidence tab."""
    frame = counts.rename_axis("片段类型").reset_index(name="候选分子数")
    return _chart_style(
        alt.Chart(frame).mark_bar(size=30, color="#1677e8", cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("片段类型:N", sort="-y", title="片段类型", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("候选分子数:Q", title="候选分子数"),
            tooltip=[alt.Tooltip("片段类型:N"), alt.Tooltip("候选分子数:Q")],
        ),
        width=800, height=260,
    ).properties(title="当前候选的片段覆盖")


def show_explanation_details(details: dict) -> None:
    """Render molecule, fragment, and pair terms from the explanation head."""
    if not details:
        return
    if details.get("rendering_warning"):
        st.warning(f"主预测已完成，但细粒度归因未能展示：{details['rendering_warning']}")
        return
    property_rows = details.get("molecule_property_rows", [])
    fragment_rows = details.get("fragment_rows", [])
    pair_rows = details.get("top_fragment_pairs", [])
    if not (property_rows or fragment_rows or pair_rows):
        return

    st.subheader("解释头分解：分子、物性与片段")
    st.caption("以下贡献均处于解释头的归一化空间；正值表示该模型条件下的正向归因，不代表独立实验因果效应。")
    molecule_tab, fragment_tab, pair_tab = st.tabs(["分子物性", "片段主项", "片段对"])
    with molecule_tab:
        if not property_rows:
            st.info("该 checkpoint 未返回逐分子物性项。")
        else:
            st.altair_chart(molecule_effect_figure(property_rows), width="stretch")
            molecule_names = [row["molecule_name"] for row in property_rows]
            selected_name = st.selectbox("查看分子结构", molecule_names, key="explain_molecule_structure")
            selected = next(row for row in property_rows if row["molecule_name"] == selected_name)
            image = molecule_png(selected["molecule_smiles"])
            if image:
                st.image(image, caption=f"{selected['molecule_name']} · 配方内比例 {selected['molar_ratio']:.3f}", width=300)
            with st.expander("查看逐分子物性数值"):
                property_frame = pd.DataFrame(property_rows).drop(columns=["molecule_smiles"], errors="ignore")
                st.dataframe(property_frame, width="stretch", hide_index=True)
    with fragment_tab:
        if not fragment_rows:
            st.info("该 checkpoint 未返回片段主项。")
        else:
            st.caption("按片段主项绝对值展示的结构；绿色为正向归因，红色为负向归因。")
            top_fragments = sorted(
                fragment_rows, key=lambda row: abs(row["main_effect_normalized"]), reverse=True
            )[:8]
            columns = st.columns(4)
            for column, fragment in zip(columns * 2, top_fragments):
                image = molecule_png(fragment["fragment_smiles"])
                with column:
                    if image:
                        st.image(image, width=160)
                    st.caption(
                        f"{fragment['molecule_name']} · f{fragment['fragment_id']}\n\n"
                        f"主项 {fragment['main_effect_normalized']:+.4f} · 注意力 {fragment['pooling_attention']:.3f}"
                    )
            with st.expander("查看片段归因数值"):
                fragment_frame = pd.DataFrame(fragment_rows).drop(columns=["molecule_smiles"], errors="ignore")
                st.dataframe(fragment_frame, width="stretch", hide_index=True)
    with pair_tab:
        if pair_rows:
            st.altair_chart(fragment_pair_figure(pair_rows), width="stretch")
            with st.expander("查看片段对数值"):
                st.dataframe(pd.DataFrame(pair_rows), width="stretch", hide_index=True)
        else:
            st.info("当前配方没有可展示的片段对项。")


def show_predict_explain() -> None:
    page_hero(
        "Frozen formulation oracle",
        "配方预测与解释",
        "在已覆盖的盐 / 溶剂范围内输入配方，获得预测电导率与可追溯的物性、片段和片段对归因。",
    )
    solvents, salts = _vocabulary()
    solvent_by_name = {item["name"]: item for item in solvents}
    salt_by_name = {item["name"]: item for item in salts}
    default_solvents = [name for name in ("EC", "DMC", "EMC") if name in solvent_by_name]
    section_label("01 · 配方输入")
    chosen_solvents = st.multiselect("溶剂", list(solvent_by_name), default=default_solvents or list(solvent_by_name)[:1])
    if not chosen_solvents:
        st.warning("请选择至少一个溶剂。")
        return
    ratio_columns = st.columns(min(4, len(chosen_solvents)))
    solvent_payload = []
    for index, name in enumerate(chosen_solvents):
        with ratio_columns[index % len(ratio_columns)]:
            ratio = st.number_input(f"{name} 溶剂内比例", min_value=0.0, value=1.0 / len(chosen_solvents), key=f"solvent_ratio_{name}")
        solvent_payload.append({**solvent_by_name[name], "ratio": ratio})
    chosen_salts = st.multiselect("盐（最多两种）", list(salt_by_name), default=list(salt_by_name)[:1], max_selections=2)
    if not chosen_salts:
        st.warning("请选择至少一种盐。")
        return
    salt_columns = st.columns(min(2, len(chosen_salts)))
    salt_payload = []
    for index, name in enumerate(chosen_salts):
        with salt_columns[index]:
            ratio = st.number_input(f"{name} 盐内摩尔份数", min_value=0.0, value=1.0 / len(chosen_salts), key=f"salt_ratio_{name}")
        salt_payload.append({**salt_by_name[name], "ratio": ratio})
    left, middle, right = st.columns(3)
    with left:
        temperature = st.number_input("温度 (°C)", value=25.0)
    with middle:
        concentration = st.number_input("总盐摩尔分数", min_value=0.001, max_value=0.95, value=0.10, step=0.01,
                                        help="n_salt / (n_salt + n_solvent)，不是 mol/L 或质量分数。")
    if st.button("运行冻结预测与解释", type="primary"):
        try:
            paths = default_paths()
            oracle = cached_oracle(str(paths.oracle_config), str(paths.oracle_checkpoint))
            st.session_state["prediction_result"] = oracle.predict(
                solvent_payload, salt_payload, temperature, concentration
            )
        except Exception as error:  # surface environment/checkpoint errors to the user without killing the app
            st.error(f"冻结 oracle 未能完成推理：{error}")
    result = st.session_state.get("prediction_result")
    if not result:
        st.info("请设置盐、溶剂、摩尔份数、温度与总盐摩尔分数后，点击“运行冻结预测与解释”。")
        return
    section_label("02 · 预测结果与贡献分解")
    result_column, note_column = st.columns((1, 2))
    with result_column:
        st.metric("预测电导率", f"{result['conductivity_ms_cm']:.3g} mS/cm")
    with note_column:
        st.info("这是冻结主预测头的结果。解释图用于说明该结果的模型归因，不额外展示解释头的电导率读数。")
    explain = result.get("conductivity_explain", {})
    if not explain:
        st.warning("当前 checkpoint 未返回 conductivity_explain 任务。请检查解释头配置。")
        return
    with st.container(border=True):
        st.caption("下图为解释头对同一配方的加性归因分解，不作为第二个需与主预测逐值一致的电导率读数展示。")
        st.altair_chart(contribution_figure(explain), width="stretch")
        st.caption("图中仅保留具有明确含义的可解释项：条件基线＝盐种、盐摩尔分数与温度的基线；溶剂物性项＝dc、vis、Tm 等物性归因；片段主项＝各片段的单独归因；片段对项＝不同片段共同出现的交互归因。")
    properties = explain.get("property_by_name", {})
    if properties:
        with st.container(border=True):
            st.altair_chart(property_effect_figure(properties), width="stretch")
    show_explanation_details(result.get("conductivity_explain_details", {}))
    st.caption("图中的片段与片段对均是当前条件下的模型归因；它们不是独立可相加的实验因果效应。")


def _range_default(constraints: list[dict], property_name: str) -> tuple[float, float]:
    row = next(row for row in constraints if row["property"] == property_name)
    return float(row["recommended_low"]), float(row["recommended_high"])


def candidate_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{
        "名称": row["name"], "SMILES": row["smiles"], "片段": ", ".join(row["motifs"]) or "other",
        "dc": row["dc"], "vis": row["vis"], "Tm": row["Tm"],
        "物性窗口分": row.get("property_window_score", math.nan),
        "片段风险": row["fragment_risk_score"], "综合分": row.get("total_score", math.nan),
        "覆盖步骤": row.get("selected_step", ""),
        "边际覆盖分": row.get("selection_marginal_score", math.nan),
        "家族": row["functional_family"], "骨架": row["murcko_scaffold"],
    } for row in rows])


def show_library() -> None:
    page_hero(
        "Molecule recommender",
        "分子推荐",
        "在 GDB17 严格中性候选中，联合片段、物性窗口、覆盖多样性与风险信号筛选可进一步评估的候选分子。",
    )
    paths = default_paths()
    try:
        catalog = cached_catalogue(str(paths.root), str(paths.gdb17_catalogue))
        constraints = property_constraints(paths)
    except Exception as error:
        st.error(f"无法载入 GDB17 候选库：{error}")
        return
    with st.sidebar:
        st.subheader("软约束筛选")
        motif_options = sorted({motif for row in catalog for motif in row["motifs"]})
        motifs = st.multiselect("片段", motif_options)
        dc_range = st.slider("介电常数 dc", 0.0, 100.0, _range_default(constraints, "dc"))
        vis_range = st.slider("黏度 vis", 0.0, 12.0, _range_default(constraints, "vis"))
        tm_range = st.slider("熔点 Tm (°C)", -100.0, 150.0, _range_default(constraints, "Tm"))
    filtered = filter_catalogue(catalog, motifs, dc_range, vis_range, tm_range)
    scored = presentation_rows(filtered, constraints, paths.root)
    filter_signature = (
        tuple(sorted(motifs)), tuple(dc_range), tuple(vis_range), tuple(tm_range)
    )
    if st.session_state.get("library_filter_signature") != filter_signature:
        for key in ("automatic_selection", "manual_ranking", "manual_selection"):
            st.session_state.pop(key, None)
        st.session_state["library_filter_signature"] = filter_signature
    first, second, third = st.columns(3)
    first.metric("筛选后候选", f"{len(scored):,}")
    second.metric("片段要求", ", ".join(motifs) if motifs else "全部")
    third.metric("默认建议词表", "62 个分子")
    section_label("候选选择与证据")
    automatic_tab, manual_tab, evidence_tab = st.tabs(["自动选择", "辅助人工选择", "片段/物性证据"])
    with automatic_tab:
        st.write("自动模式复用 `coverage_v1`：物性窗口、片段边际收益、家族/骨架覆盖、相似度冗余与 anion 倾向平衡。")
        max_size = min(62, len(scored))
        if max_size == 0:
            requested_size = 0
        elif max_size <= 8:
            st.caption(f"当前筛选结果仅有 {max_size} 个分子，将全部纳入候选词表。")
            requested_size = max_size
        else:
            requested_size = st.slider("候选词表大小", 8, max_size, max_size)
        selection_seed = st.number_input("选择随机种子", min_value=0, value=20260909, step=1)
        if st.button("运行自动选择", type="primary", disabled=not scored):
            try:
                st.session_state["automatic_selection"] = automatic_selection(
                    scored, constraints, paths.root, pool_size=requested_size, seed=int(selection_seed)
                )
            except Exception as error:
                st.error(str(error))
        selected = st.session_state.get("automatic_selection", [])
        if selected:
            st.success(f"已生成 {len(selected)} 个分子的词表提案。该提案不会改写当前 generator 词表。")
            st.dataframe(candidate_frame(selected), width="stretch", hide_index=True)
            st.download_button("下载自动词表 CSV", selection_csv(selected), "gdb17_auto_vocabulary.csv", "text/csv")
    with manual_tab:
        st.write("先用与自动模式相同的 `coverage_v1` 逻辑排序，再由用户锁定候选；不会自动替换你的选择。")
        manual_max = min(300, len(scored))
        if manual_max == 0:
            st.info("当前筛选条件下没有可排序候选。")
        else:
            if manual_max <= 8:
                manual_size = manual_max
                st.caption(f"当前仅有 {manual_size} 个候选，将全部给出人工排序。")
            else:
                manual_size = st.slider("待人工选择的排序候选数", 8, manual_max, min(62, manual_max))
            manual_seed = st.number_input("人工排序随机种子", min_value=0, value=20260909, step=1)
            if st.button("生成供人工锁定的系统排序", disabled=not scored):
                try:
                    st.session_state["manual_ranking"] = automatic_selection(
                        scored, constraints, paths.root, pool_size=manual_size, seed=int(manual_seed)
                    )
                except Exception as error:
                    st.error(str(error))
            ranked_manual_rows = st.session_state.get("manual_ranking", [])
            if ranked_manual_rows:
                st.dataframe(candidate_frame(ranked_manual_rows), width="stretch", hide_index=True)
                options = {
                    f"#{row['selected_step']:03d} · {row['name']} · {row['smiles']}": row
                    for row in ranked_manual_rows
                }
                chosen = st.multiselect("锁定候选分子", list(options), key="manual_selection")
                manual_rows = [options[item] for item in chosen]
                if manual_rows:
                    st.dataframe(candidate_frame(manual_rows), width="stretch", hide_index=True)
                    if len(manual_rows) != 62:
                        st.warning(f"当前人工词表为 {len(manual_rows)} 个分子；正式替换 generator 词表通常需要 62 个。")
                    st.download_button("下载人工词表 CSV", selection_csv(manual_rows), "gdb17_manual_vocabulary.csv", "text/csv")
            else:
                st.info("先生成系统排序，再从中锁定人工词表。")
    with evidence_tab:
        st.subheader("片段、物性与风险证据")
        if scored:
            motif_counts = pd.Series(
                motif for row in scored for motif in row["motifs"]
            ).value_counts().head(12)
            if not motif_counts.empty:
                st.caption("当前筛选候选中的片段覆盖；片段条件也会参与自动/人工词表排序。")
                st.altair_chart(motif_coverage_figure(motif_counts), width="stretch")
        visible_rows = scored[:100]
        if st.button("返回列表默认分子", help="清除当前表格行选择并回到第一条候选。"):
            st.session_state.pop("molecule_recommendation_table", None)
            st.rerun()
        selection = st.dataframe(
            candidate_frame(visible_rows), width="stretch", hide_index=True, height=480,
            on_select="rerun", selection_mode="single-row", key="molecule_recommendation_table",
        )
        if scored:
            try:
                selected_indices = list(selection.selection.rows)
            except AttributeError:
                selected_indices = list(selection.get("selection", {}).get("rows", []))
            sample = visible_rows[selected_indices[0]] if selected_indices else visible_rows[0]
            image = molecule_png(sample["smiles"])
            left, middle, right = st.columns((1, 1, 2))
            with left:
                if image:
                    st.image(image, caption=f"普通分子结构：{sample['name']}")
                else:
                    st.code(sample["smiles"], language=None)
            with middle:
                topology = fragment_topology_png(sample["smiles"])
                if topology:
                    st.image(topology, caption="片段拓扑：原子:片段编号；红键＝跨片段连接")
            with right:
                st.markdown(
                    f"**当前选中：** {sample['name']}\n\n"
                    f"**片段匹配：** {sample['matched_structure_terms']}\n\n"
                    f"**物性：** dc={sample['dc']:.2f}，vis={sample['vis']:.2f}，Tm={sample['Tm']:.2f} °C\n\n"
                    f"**片段风险分：** {sample['fragment_risk_score']:+.2f}\n\n"
                    f"**SMILES：** `{sample['smiles']}`"
                )
    st.info("GDB17 提供结构候选，不表示该分子可采购、可合成、稳定或适用于实际电池。新选出的词表若要进入 generator，仍需后续重建 embedding 与重训生成模型。")


def component_line(rows: list[dict]) -> str:
    return " + ".join(f"{row['name']} ({row['ratio']:.2f})" for row in rows) or "—"


def constrain_generation_rows(rows: list[dict], required: set[str], diversify: bool) -> list[dict]:
    """Apply UI-only constraints after ranking for service-version compatibility."""
    filtered = [
        row for row in rows
        if not required or required.issubset({component["name"] for component in row["solvents"]})
    ]
    if not diversify:
        return filtered
    unique, remainder, signatures = [], [], set()
    for row in filtered:
        signature = tuple(sorted(component["name"] for component in row["solvents"]))
        if signature in signatures:
            remainder.append(row)
        else:
            signatures.add(signature)
            unique.append(row)
    return unique + remainder


def show_generate_rank() -> None:
    page_hero(
        "Frozen G0-MA · Bamboo vocabulary",
        "Top-N 实验配方卡",
        "按目标电导率和 anion ratio 重排序已生成的配方样本；结果可直接导出为实验筛选卡。",
    )
    try:
        records, solvents, salts = cached_generation_assets()
    except FileNotFoundError as error:
        st.error(str(error))
        return
    section_label("01 · 目标与词表范围")
    left, middle, right, fourth = st.columns(4)
    with left:
        target_sigma = st.number_input("目标电导率 (mS/cm)", min_value=.1, value=20.0)
    with middle:
        target_ratio = st.number_input("目标 anion ratio", min_value=0.0, max_value=1.0, value=.5, step=.01)
    with right:
        sigma_tol = st.number_input("电导率容差", min_value=.1, value=2.0)
    with fourth:
        ratio_tol = st.number_input("ratio 容差", min_value=.01, max_value=1.0, value=.05, step=.01)
    allowed = st.multiselect("允许出现的 Bamboo 62 词表溶剂（留空 / 全选表示不限制）", [row["name"] for row in solvents])
    diversify = st.toggle("优先展示不同溶剂组成", value=True,
                          help="每种溶剂组合先保留联合目标最优的一张卡，避免同一 Glyme 配方占满 Top-N。")
    ranked = rank_frozen_samples(
        records, solvents, salts, target_sigma, target_ratio, sigma_tol, ratio_tol,
        set(allowed) if allowed else None,
    )
    ranked = constrain_generation_rows(ranked, set(), diversify)
    top_n = st.slider("显示 Top-N", 1, min(20, len(ranked)), min(5, len(ranked))) if ranked else 0
    st.metric("可排序候选", f"{len(ranked):,}")
    if not ranked:
        st.warning("当前溶剂限制下没有 G0-MA 样本。请扩大允许词表或取消限制；系统不会加入未选择的溶剂。")
        return
    section_label("02 · 优先实验配方")
    cards = ranked[:top_n]
    for index, card in enumerate(cards, start=1):
        with st.container(border=True):
            heading, status = st.columns((5, 1))
            heading.subheader(f"#{index} · 联合排序分 {card['joint_loss']:.3f}")
            status.success(card["prediction_status"])
            st.markdown(f"**盐：** {component_line(card['salts'])}  \n\n**溶剂：** {component_line(card['solvents'])}")
            metric_columns = st.columns(3)
            metric_columns[0].metric("预测电导率", f"{card['conductivity_ms_cm']:.3g} mS/cm", f"误差 {card['conductivity_error']:.3g}")
            metric_columns[1].metric("预测 anion ratio", f"{card['anion_ratio']:.3f}", f"误差 {card['anion_error']:.3f}")
            metric_columns[2].metric("温度 / 盐摩尔分数", f"{card['temperature_c']:.0f} °C / {card['salt_molar_ratio']:.2f}")
            st.caption("若需完整的物性、片段与片段对归因，可在“配方预测与解释”页用同一配方运行冻结 oracle。")
    st.download_button("导出 Top-N 实验配方卡 JSON", card_json(cards), "top_n_experiment_cards.json", "application/json")


def main() -> None:
    st.sidebar.title("⚗️ Electrolyte Designer")
    page = st.sidebar.radio("页面", ["配方预测与解释", "分子推荐", "Top-N 实验配方卡"])
    st.sidebar.divider()
    st.sidebar.caption("Demo v1 · 冻结模型推理 · 不执行训练")
    if page == "配方预测与解释":
        show_predict_explain()
    elif page == "分子推荐":
        show_library()
    else:
        show_generate_rank()


if __name__ == "__main__":
    main()
