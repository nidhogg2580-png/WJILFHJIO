# ============================================================
# 生存曲线分析工具 · Streamlit 版
# 功能：多组KM曲线 + Log-rank两两检验 + Cox HR + 协变量选择调整
# ============================================================

import io
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch
import streamlit as st

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.utils import median_survival_times

warnings.filterwarnings("ignore")

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="生存曲线分析工具",
    page_icon="📊",
    layout="wide",
)

# ============================================================
# 邀请码验证（最先执行）
# ============================================================
INVITE_CODE = "WHU2026"

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.markdown(
        """
        <div style="background:linear-gradient(135deg,#00558C,#003a63);
                    border-radius:10px;padding:18px 24px;margin-bottom:24px;">
          <h2 style="color:white;margin:0;font-family:Arial,sans-serif;">
            📊 生存曲线分析工具
          </h2>
          <p style="color:#cce4f7;margin:6px 0 0;font-size:13px;">
            支持多组 KM 曲线 · Log-rank 两两检验 · Cox 比例风险回归
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.subheader("🔐 请输入邀请码以访问系统")
    code_input = st.text_input("邀请码", type="password", placeholder="请输入邀请码")
    if st.button("确认"):
        if code_input == INVITE_CODE:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("❌ 邀请码错误，请重新输入。")
    st.stop()

# ============================================================
# 字体设置
# ============================================================
def _setup_font():
    available = {f.name for f in fm.fontManager.ttflist}
    for candidate in ["Arial", "Liberation Sans", "FreeSans", "DejaVu Sans"]:
        if candidate in available:
            plt.rcParams.update({
                "font.family": "sans-serif",
                "font.sans-serif": [candidate],
                "axes.unicode_minus": False,
            })
            return candidate
    plt.rcParams.update({"axes.unicode_minus": False})
    return "default"

_setup_font()

# ============================================================
# 配色
# ============================================================
PRESET_COLORS = [
    "#00558C", "#2E8B57", "#D4820A",
    "#6A0DAD", "#007272", "#C84B31", "#1B6CA8",
    "#8B008B", "#556B2F", "#B8860B",
]

def assign_colors(groups, reference):
    colors = {}
    non_ref = [g for g in groups if g != reference]
    for i, g in enumerate(non_ref):
        colors[g] = PRESET_COLORS[i] if i < len(PRESET_COLORS) \
            else "#{:06x}".format(random.randint(0x333333, 0xBBBBBB))
    colors[reference] = "#A6192E"
    return colors

# ============================================================
# 工具函数
# ============================================================
def get_median_ci(kmf):
    median = kmf.median_survival_time_
    if np.isinf(median) or pd.isna(median):
        return "NE", "NE"
    try:
        ci = median_survival_times(kmf.confidence_interval_)
        lower = float(ci.iloc[0, 0])
        upper = float(ci.iloc[0, 1])
        ls = "NE" if (np.isinf(lower) or pd.isna(lower)) else f"{lower:.1f}"
        us = "NE" if (np.isinf(upper) or pd.isna(upper)) else f"{upper:.1f}"
        return f"{median:.1f}", f"{ls}\u2013{us}"
    except:
        return f"{median:.1f}", "NE"

def fmt_p(p):
    if p < 0.0001:
        return "p<0.0001"
    elif p < 0.001:
        return f"p={p:.4f}"
    else:
        return f"p={p:.3f}"

# ============================================================
# 核心绘图函数
# ============================================================
def build_figure(analysis, text_x=0.02, text_y=0.42):
    """
    analysis : dict，由 run_analysis() 计算并缓存
    text_x/text_y : Median+HR 文字块在 ax 坐标中的左上角位置
    返回 img_bytes (PNG)
    """
    groups = analysis["groups"]
    kmf_dict = analysis["kmf_dict"]
    group_dfs = analysis["group_dfs"]
    colors = analysis["colors"]
    hr_texts = analysis["hr_texts"]
    x_max = analysis["x_max"]
    x_ticks = analysis["x_ticks"]
    tc = analysis["tc"]
    ec = analysis["ec"]
    x_label = analysis["x_label"]
    y_label = analysis["y_label"]
    n_groups = len(groups)

    FS_TICK = 12
    FS_LABEL = 13
    FS_LEGEND = 12
    FS_TEXT = 11
    FS_TABLE = 11
    FS_THEAD = 12

    ROW_H_INCH = 0.38
    tbl_h = (n_groups + 3.5) * ROW_H_INCH + 0.20
    main_h = 8.5 * 0.8
    fig_h = main_h + tbl_h

    fig = plt.figure(figsize=(14, fig_h), dpi=150)
    gs = fig.add_gridspec(2, 1,
                          height_ratios=[main_h, tbl_h],
                          hspace=0.05)
    ax = fig.add_subplot(gs[0])

    # KM 曲线
    for g in groups:
        kmf = kmf_dict[g]
        col = colors[g]
        sf = kmf.survival_function_ * 100
        ax.step(sf.index, sf.iloc[:, 0], where="post",
                color=col, lw=2.0, label=str(g))
        ci_df = kmf.confidence_interval_ * 100
        ax.fill_between(ci_df.index, ci_df.iloc[:, 0], ci_df.iloc[:, 1],
                        step="post", alpha=0.15, color=col)
        cens = group_dfs[g][group_dfs[g][ec] == 0]
        if len(cens):
            yvals = kmf.survival_function_at_times(cens[tc]) * 100
            ax.scatter(cens[tc], yvals, marker="|", color=col, s=60, lw=1.4, zorder=5)

    # 坐标轴
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 100)
    ax.set_xticks(x_ticks)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel(y_label, fontsize=FS_LABEL)
    ax.set_xlabel("")

    # 图例
    ax.legend(frameon=False, loc="upper right", fontsize=FS_LEGEND)

    # Median + HR 文字块
    line_h = 0.048 / 0.8
    line_h = min(line_h, 0.065)

    col_group = text_x + 0.008
    col_median = text_x + 0.008 + 0.18

    median_rows = []
    for g in groups:
        med, ci_s = get_median_ci(kmf_dict[g])
        median_rows.append((str(g), f"{med} ({ci_s})"))

    hr_lines = hr_texts

    n_lines = 1 + len(median_rows) + 1 + len(hr_lines)
    box_h = n_lines * line_h + 0.01
    box_w = 0.50

    ax.add_patch(FancyBboxPatch(
        (text_x - 0.005, text_y - box_h),
        box_w, box_h + 0.005,
        boxstyle="square,pad=0.005",
        transform=ax.transAxes,
        linewidth=0, edgecolor="none", facecolor="none",
        zorder=4,
    ))

    ax.text(col_group, text_y, "Median time (95% CI)",
            transform=ax.transAxes, fontsize=FS_TEXT,
            va="top", fontweight="bold", zorder=5)

    for i, (g_str, med_ci_str) in enumerate(median_rows):
        y_pos = text_y - (i + 1) * line_h
        ax.text(col_group, y_pos, g_str, transform=ax.transAxes,
                fontsize=FS_TEXT, va="top", zorder=5)
        ax.text(col_median, y_pos, med_ci_str, transform=ax.transAxes,
                fontsize=FS_TEXT, va="top", ha="left", zorder=5)

    hr_y_start = text_y - (len(median_rows) + 2) * line_h
    for i, ht in enumerate(hr_lines):
        ax.text(col_group, hr_y_start - i * line_h, ht,
                transform=ax.transAxes, fontsize=FS_TEXT, va="top", zorder=5)

    # 风险表
    ax_tbl = fig.add_subplot(gs[1])
    ax_tbl.axis("off")
    ax_tbl.set_xlim(0, x_max)
    ax_tbl.set_ylim(0, 1)

    row_unit = ROW_H_INCH / tbl_h
    top_pad = 0.5 * row_unit
    xlabel_y = 1.0 - top_pad
    gap_xlabel_to_header = 1.5 * row_unit - 1.0 * row_unit
    header_y = xlabel_y - row_unit - gap_xlabel_to_header
    gap_header_to_data = 1.5 * row_unit
    first_row_y = header_y - row_unit - gap_header_to_data

    ax_tbl.text(x_max / 2, xlabel_y, x_label,
                ha="center", va="top", fontsize=FS_THEAD)

    ax_tbl.text(-x_max * 0.065, header_y,
                "Number at risk\n(censored)",
                ha="right", va="top",
                fontsize=FS_THEAD, fontweight="bold")

    for i, g in enumerate(groups):
        gdf = group_dfs[g]
        row_y = first_row_y - i * row_unit
        ax_tbl.text(-x_max * 0.065, row_y, str(g),
                    ha="right", va="center",
                    fontsize=FS_THEAD, color="black")
        for t in x_ticks:
            n_risk = int((gdf[tc] >= t).sum())
            n_cens = int(((gdf[tc] <= t) & (gdf[ec] == 0)).sum())
            ax_tbl.text(t, row_y, f"{n_risk} ({n_cens})",
                        ha="center", va="center", fontsize=FS_TABLE)

    plt.subplots_adjust(left=0.16, right=0.95, top=0.96, bottom=0.02)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    img_bytes = buf.read()
    plt.close()
    return img_bytes

# ============================================================
# 核心分析
# ============================================================
def run_analysis(df, gc, tc, ec, groups, ref, colors, sel_cov, x_max, x_ticks, x_label, y_label):
    df = df.copy()
    df[tc] = pd.to_numeric(df[tc], errors="coerce")
    df[ec] = pd.to_numeric(df[ec], errors="coerce")
    df = df.dropna(subset=[tc, ec])

    non_ref_groups = [g for g in groups if g != ref]

    # KM 拟合
    kmf_dict = {}
    group_dfs = {}
    for g in groups:
        sub = df[df[gc] == g].copy()
        group_dfs[g] = sub
        kmf = KaplanMeierFitter()
        kmf.fit(sub[tc], sub[ec], label=str(g))
        kmf_dict[g] = kmf

    # Log-rank 两两
    pairwise_p = {}
    for g in non_ref_groups:
        lr = logrank_test(
            group_dfs[ref][tc], group_dfs[g][tc],
            event_observed_A=group_dfs[ref][ec],
            event_observed_B=group_dfs[g][ec],
        )
        pairwise_p[g] = lr.p_value

    # Cox HR
    hr_texts = []
    n_non_ref = len(non_ref_groups)
    for idx, g in enumerate(non_ref_groups):
        cox_sub = df[df[gc].isin([ref, g])].copy()
        cox_sub["_trt"] = (cox_sub[gc] == g).astype(int)
        fit_cols = [tc, ec, "_trt"]
        for col in sel_cov:
            if col in cox_sub.columns:
                cox_sub[col] = pd.to_numeric(cox_sub[col], errors="coerce")
                fit_cols.append(col)
        cox_sub = cox_sub[fit_cols].dropna()

        if n_non_ref == 1:
            hr_label = "HR"
        else:
            hr_label = f"HR{idx+1} ({g} vs {ref})"

        p_str = fmt_p(pairwise_p[g])
        if len(cox_sub) < 5:
            hr_texts.append(f"{hr_label}: 样本量不足")
            continue
        try:
            cph = CoxPHFitter()
            cph.fit(cox_sub, duration_col=tc, event_col=ec)
            hr = np.exp(cph.params_["_trt"])
            cil = np.exp(cph.confidence_intervals_.loc["_trt"].iloc[0])
            ciu = np.exp(cph.confidence_intervals_.loc["_trt"].iloc[1])
            hr_texts.append(
                f"{hr_label}: {hr:.2f} (95% CI {cil:.2f}\u2013{ciu:.2f}); {p_str}"
            )
        except Exception as e:
            hr_texts.append(f"{hr_label}: Cox拟合失败 ({e})")

    analysis = {
        "groups": groups, "ref": ref,
        "kmf_dict": kmf_dict, "group_dfs": group_dfs,
        "colors": colors, "hr_texts": hr_texts,
        "pairwise_p": pairwise_p, "non_ref_groups": non_ref_groups,
        "x_max": x_max, "x_ticks": x_ticks,
        "tc": tc, "ec": ec,
        "x_label": x_label, "y_label": y_label,
    }
    return analysis

# ============================================================
# Session state 初始化
# ============================================================
for key, default in {
    "df": None,
    "covariate_cols": [],
    "covariate_types": {},
    "selected_covariates": [],
    "analysis": None,
    "step": 1,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ============================================================
# 顶部标题
# ============================================================
st.markdown(
    """
    <div style="background:linear-gradient(135deg,#00558C,#003a63);
                border-radius:10px;padding:18px 24px;margin-bottom:16px;">
      <h2 style="color:white;margin:0;font-family:Arial,sans-serif;">
        📊 生存曲线分析工具
      </h2>
      <p style="color:#cce4f7;margin:6px 0 0;font-size:13px;">
        支持多组 KM 曲线 · Log-rank 两两检验 · Cox 比例风险回归
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 进度指示
# ============================================================
step = st.session_state["step"]
cols_prog = st.columns(4)
step_labels = ["第1步：上传数据", "第2步：协变量类型", "第3步：分析设置", "第4步：图形结果"]
for i, (col, label) in enumerate(zip(cols_prog, step_labels)):
    active = (i + 1) == step
    color = "#00558C" if active else ("#2E8B57" if (i + 1) < step else "#ccc")
    text_color = "white" if active else ("#2E8B57" if (i + 1) < step else "#888")
    bg = color if active else ("transparent" if (i + 1) < step else "#f0f0f0")
    with col:
        st.markdown(
            f"""<div style="background:{bg};border:2px solid {color};border-radius:6px;
            padding:8px;text-align:center;font-size:13px;font-weight:{'bold' if active else 'normal'};
            color:{text_color if active else color};">{label}</div>""",
            unsafe_allow_html=True,
        )

st.markdown("---")

# ============================================================
# 第 1 步：上传文件
# ============================================================
if st.session_state["step"] == 1:
    st.markdown(
        """<div style="background:#f0f4f8;border-left:4px solid #00558C;
        padding:8px 12px;font-weight:bold;font-size:15px;margin-bottom:10px;">
        第 1 步：上传数据文件</div>""",
        unsafe_allow_html=True,
    )
    st.info(
        "请上传 Excel 文件（.xlsx / .xls）。\n\n"
        "**前4列**依次为：受试者编号 · 组别 · 观测时长 · 结局状态（1=事件发生，0=删失）。\n\n"
        "第5列起为协变量（可选）。"
    )
    uploaded_file = st.file_uploader("选择 Excel 文件", type=["xlsx", "xls"])

    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            cols = list(df.columns)
            if len(cols) < 4:
                st.error("❌ 数据列数不足，至少需要4列（编号、组别、时间、事件）。")
            else:
                df.rename(columns={
                    cols[0]: "__id__",
                    cols[1]: "__group__",
                    cols[2]: "__time__",
                    cols[3]: "__event__",
                }, inplace=True)
                cov_cols = [c for c in df.columns
                            if c not in ("__id__", "__group__", "__time__", "__event__")]
                st.session_state["df"] = df
                st.session_state["covariate_cols"] = cov_cols
                st.session_state["covariate_types"] = {}
                st.session_state["selected_covariates"] = []
                st.session_state["analysis"] = None

                st.success(
                    f"✅ 成功读取：**{uploaded_file.name}**，共 **{len(df)}** 行，"
                    f"检测到 **{len(cov_cols)}** 个协变量列。"
                )
                st.dataframe(df.head(5), use_container_width=True)

                if st.button("下一步：设置协变量类型 ▶", type="primary"):
                    if cov_cols:
                        st.session_state["step"] = 2
                    else:
                        st.session_state["step"] = 3
                    st.rerun()
        except Exception as e:
            st.error(f"❌ 读取失败：{e}")

# ============================================================
# 第 2 步：协变量类型 + 协变量纳入模型
# ============================================================
elif st.session_state["step"] == 2:
    st.markdown(
        """<div style="background:#f0f4f8;border-left:4px solid #00558C;
        padding:8px 12px;font-weight:bold;font-size:15px;margin-bottom:10px;">
        第 2 步：指定协变量类型</div>""",
        unsafe_allow_html=True,
    )
    st.info("请为每个协变量指定类型。定性变量须已用 0、1、2… 整数编码。")

    cov_cols = st.session_state["covariate_cols"]
    cov_type_selections = {}

    for col in cov_cols:
        default_idx = 0
        prev = st.session_state["covariate_types"].get(col, "quantitative")
        default_idx = 0 if prev == "quantitative" else 1
        choice = st.selectbox(
            f"协变量：**{col}**",
            options=["定量变量（连续）", "定性变量（分类，需数字编码 0/1/2…）"],
            index=default_idx,
            key=f"cov_type_{col}",
        )
        cov_type_selections[col] = "quantitative" if choice.startswith("定量") else "qualitative"

    if st.button("确认协变量类型 ✔", type="primary"):
        issues = []
        for col, ctype in cov_type_selections.items():
            if ctype == "qualitative":
                try:
                    vals = st.session_state["df"][col].dropna().astype(float)
                    if not all(v == int(v) for v in vals):
                        issues.append(col)
                except:
                    issues.append(col)
        if issues:
            st.warning(f"⚠️ 以下定性变量含非整数值，请检查：{issues}")
        else:
            st.session_state["covariate_types"] = cov_type_selections
            st.success("✅ 类型已确认！")

    st.markdown("---")

    # 协变量纳入模型
    st.markdown(
        """<div style="background:#f0f4f8;border-left:4px solid #00558C;
        padding:8px 12px;font-weight:bold;font-size:15px;margin-bottom:10px;">
        第 2b 步：选择纳入 Cox 模型的协变量</div>""",
        unsafe_allow_html=True,
    )
    st.caption("请勾选需要纳入 Cox 比例风险回归模型进行调整的协变量（若不选则进行无调整的单变量Cox回归）。")

    selected_covs = []
    for col in cov_cols:
        ctype = st.session_state["covariate_types"].get(col, "quantitative")
        type_label = "定量" if ctype == "quantitative" else "定性"
        prev_checked = col in st.session_state.get("selected_covariates", [])
        checked = st.checkbox(f"{col}  [{type_label}]", value=prev_checked, key=f"sel_cov_{col}")
        if checked:
            selected_covs.append(col)

    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("◀ 返回上一步"):
            st.session_state["step"] = 1
            st.rerun()
    with col2:
        if st.button("确认并继续 ▶", type="primary"):
            st.session_state["selected_covariates"] = selected_covs
            if selected_covs:
                st.success(f"✅ 已选 {len(selected_covs)} 个协变量纳入Cox模型。")
            else:
                st.success("✅ 将进行无调整的单变量Cox回归。")
            st.session_state["step"] = 3
            st.rerun()

# ============================================================
# 第 3 步：分析设置
# ============================================================
elif st.session_state["step"] == 3:
    st.markdown(
        """<div style="background:#f0f4f8;border-left:4px solid #00558C;
        padding:8px 12px;font-weight:bold;font-size:15px;margin-bottom:10px;">
        第 3 步：分析设置</div>""",
        unsafe_allow_html=True,
    )

    df = st.session_state["df"]
    groups = sorted(df["__group__"].dropna().unique().tolist())
    st.info(f"检测到 **{len(groups)}** 个组别：{', '.join(str(g) for g in groups)}")

    ref_group = st.selectbox("选择对照组", options=groups, index=0)

    col_a, col_b = st.columns(2)
    with col_a:
        x_label = st.text_input("横轴标签", value="Time (months)")
        x_max = st.number_input("横轴最大值", min_value=1, max_value=9999, value=36, step=1)
    with col_b:
        y_label = st.text_input("纵轴标签", value="Survival probability (%)")
        x_ticks_str = st.text_input("横轴刻度（逗号分隔）", value="0,6,12,18,24,30,36")

    col1, col2 = st.columns([1, 4])
    with col1:
        back_step = 3 if st.session_state["covariate_cols"] else 1
        if st.button("◀ 返回上一步"):
            st.session_state["step"] = 2 if st.session_state["covariate_cols"] else 1
            st.rerun()
    with col2:
        if st.button("🔬 开始绘图分析", type="primary"):
            try:
                x_ticks = [float(x.strip()) for x in x_ticks_str.split(",") if x.strip()]
            except:
                x_ticks = list(range(0, int(x_max) + 1, 6))

            colors = assign_colors(groups, ref_group)

            with st.spinner("正在进行统计分析和绘图，请稍候…"):
                try:
                    analysis = run_analysis(
                        df=df,
                        gc="__group__",
                        tc="__time__",
                        ec="__event__",
                        groups=groups,
                        ref=ref_group,
                        colors=colors,
                        sel_cov=st.session_state["selected_covariates"],
                        x_max=x_max,
                        x_ticks=x_ticks,
                        x_label=x_label,
                        y_label=y_label,
                    )
                    st.session_state["analysis"] = analysis
                    st.session_state["step"] = 4
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ 分析失败：{e}")

# ============================================================
# 第 4 步：图形结果 + 位置调整 + 下载
# ============================================================
elif st.session_state["step"] == 4:
    st.markdown(
        """<div style="background:#f0f4f8;border-left:4px solid #00558C;
        padding:8px 12px;font-weight:bold;font-size:15px;margin-bottom:10px;">
        第 4 步：图形结果与下载</div>""",
        unsafe_allow_html=True,
    )

    analysis = st.session_state["analysis"]

    if analysis is None:
        st.warning("尚无分析结果，请返回第3步重新运行。")
    else:
        # 统计摘要
        with st.expander("📋 统计摘要", expanded=True):
            groups = analysis["groups"]
            ref = analysis["ref"]
            group_dfs = analysis["group_dfs"]
            pairwise_p = analysis["pairwise_p"]
            hr_texts = analysis["hr_texts"]
            ec = analysis["ec"]
            kmf_dict = analysis["kmf_dict"]
            sel_cov = st.session_state["selected_covariates"]

            st.markdown("**各组样本量与事件数：**")
            summary_rows = []
            for g in groups:
                gdf = group_dfs[g]
                n = len(gdf)
                events = int(gdf[ec].sum())
                med, ci_s = get_median_ci(kmf_dict[g])
                summary_rows.append({
                    "组别": str(g),
                    "样本量 (n)": n,
                    "事件数": events,
                    "中位生存时间": med,
                    "95% CI": ci_s,
                })
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

            st.markdown(f"**Log-rank 两两比较（对照组：{ref}）：**")
            lr_rows = []
            for g, pv in pairwise_p.items():
                lr_rows.append({"比较": f"{ref} vs {g}", "P值": fmt_p(pv)})
            st.dataframe(pd.DataFrame(lr_rows), use_container_width=True, hide_index=True)

            if sel_cov:
                st.markdown(f"**Cox调整协变量：** {', '.join(sel_cov)}")
            else:
                st.markdown("**Cox：** 单变量（无协变量调整）")

            st.markdown("**HR 结果（Cox比例风险回归）：**")
            for ht in hr_texts:
                st.markdown(f"- {ht}")

        # 位置调整
        st.markdown("---")
        st.markdown(
            """<div style="background:#fff3cd;border:1px solid #ffc107;
            border-radius:6px;padding:10px 14px;margin:10px 0;font-size:13px;">
            📌 <b>调整统计文字框位置</b>：拖动下方滑块移动图中的 Median / HR 文字框位置，
            满意后点击「确认并下载」。</div>""",
            unsafe_allow_html=True,
        )

        col_sl1, col_sl2 = st.columns(2)
        with col_sl1:
            text_x = st.slider("水平位置（X）", min_value=0.00, max_value=0.85,
                                value=0.02, step=0.01, format="%.2f")
        with col_sl2:
            text_y = st.slider("垂直位置（Y）", min_value=0.10, max_value=0.98,
                                value=0.42, step=0.01, format="%.2f")

        # 预览图
        with st.spinner("正在生成预览图…"):
            img_bytes = build_figure(analysis, text_x=text_x, text_y=text_y)

        st.image(img_bytes, caption="生存曲线预览（调整滑块后实时更新）", use_container_width=True)

        # 下载按钮
        st.download_button(
            label="⬇️ 下载图片（PNG）",
            data=img_bytes,
            file_name="survival_curve.png",
            mime="image/png",
            type="primary",
        )

        st.markdown("---")
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("◀ 返回修改设置"):
                st.session_state["step"] = 3
                st.rerun()
        with col2:
            if st.button("🔄 重新开始（上传新文件）"):
                for key in ["df", "covariate_cols", "covariate_types",
                            "selected_covariates", "analysis"]:
                    st.session_state[key] = [] if key in ("covariate_cols", "selected_covariates") \
                        else ({} if key == "covariate_types" else None)
                st.session_state["step"] = 1
                st.rerun()
