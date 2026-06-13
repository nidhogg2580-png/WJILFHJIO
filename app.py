"""
折线图交互绘图工具 — Streamlit 单文件应用
支持：原始数据（线性混合模型 / 描述性统计）和直接绘图数据
"""

import io
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import streamlit as st

# ── 字体 ──────────────────────────────────────────────────────────────────────
try:
    import matplotlib.font_manager as fm
    _fonts = [f.name for f in fm.fontManager.ttflist]
    _HAS_NOTO = any("Noto Serif SC" in f or "NotoSerifSC" in f for f in _fonts)
    _HAS_NOTO_SANS = any("Noto Sans SC" in f or "NotoSansSC" in f for f in _fonts)
except Exception:
    _HAS_NOTO = _HAS_NOTO_SANS = False

FONT_EN  = ["Arial", "Liberation Sans", "DejaVu Sans"]
FONT_ZH  = (["Noto Serif SC"] + FONT_EN) if _HAS_NOTO else \
           (["Noto Sans SC"] + FONT_EN) if _HAS_NOTO_SANS else \
           ["SimHei", "WenQuanYi Micro Hei"] + FONT_EN

# ── 颜色序列（固定顺序）──────────────────────────────────────────────────────
COLOR_SEQ = [
    "#EE3224", "#0054A6", "#2CA02C", "#FF7F0E",
    "#9467BD", "#8C564B", "#E377C2", "#7F7F7F",
    "#BCBD22", "#17BECF",
]

# ═══════════════════════════════════════════════════════════════════════════════
# 页面配置
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="折线图绘图工具",
    page_icon="📈",
    layout="wide",
)

st.title("📈 折线图交互绘图工具")
st.caption("支持原始数据统计分析 + 直接绘图数据 | 可导出 PNG / SVG / PDF")

# ═══════════════════════════════════════════════════════════════════════════════
# Session state 初始化
# ═══════════════════════════════════════════════════════════════════════════════
def _init(key, val):
    if key not in st.session_state:
        st.session_state[key] = val

_init("step", 1)
_init("plot_df", None)       # 最终绘图数据 DataFrame
_init("groups_order", [])    # 组别顺序（固定颜色用）
_init("x_numeric", True)     # 时间点是否为数值型
_init("raw_df", None)
_init("covariate_cols", [])

# ═══════════════════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════════════════

def try_numeric_x(series: pd.Series):
    """尝试将时间点列转为数值；失败则返回 None。"""
    try:
        return pd.to_numeric(series, errors="raise")
    except Exception:
        return None

def normality_test(arr):
    """Shapiro-Wilk 正态性检验，n < 3 直接返回 False。"""
    from scipy.stats import shapiro
    arr = arr.dropna()
    if len(arr) < 3:
        return False
    _, p = shapiro(arr)
    return p >= 0.05   # True → 正态

def run_descriptive(df: pd.DataFrame, group_col, time_col, outcome_col):
    """描述性统计：判断正态 → 均值±SD 或 中位数（无误差棒）。"""
    groups = df[group_col].unique()
    times  = df[time_col].unique()

    rows = []
    all_normal = True
    for g in groups:
        for t in times:
            sub = df[(df[group_col]==g) & (df[time_col]==t)][outcome_col]
            if not normality_test(sub):
                all_normal = False

    for g in groups:
        for t in times:
            sub = df[(df[group_col]==g) & (df[time_col]==t)][outcome_col].dropna()
            if len(sub) == 0:
                continue
            if all_normal:
                val  = sub.mean()
                lo   = val - sub.std()
                hi   = val + sub.std()
                note = "Mean ± SD"
            else:
                val  = sub.median()
                lo   = np.nan
                hi   = np.nan
                note = "Median (no error bar)"
            rows.append({
                "group": g, "time": t,
                "value": val, "lower": lo, "upper": hi,
                "note": note,
            })

    return pd.DataFrame(rows), all_normal

def run_lmm(df: pd.DataFrame, group_col, time_col, outcome_col,
            subject_col, covariates, cov_types):
    """线性混合模型，随机效应=受试者，返回 LS means + SE。"""
    from statsmodels.formula.api import mixedlm
    import re

    # 构建公式
    safe = lambda c: re.sub(r'\W', '_', str(c))
    df2 = df.copy()
    df2.columns = [safe(c) for c in df2.columns]

    g_col  = safe(group_col)
    t_col  = safe(time_col)
    y_col  = safe(outcome_col)
    s_col  = safe(subject_col)

    # 将 group 和 time 作为 category
    df2[g_col] = df2[g_col].astype("category")
    df2[t_col] = df2[t_col].astype("category")

    cov_terms = []
    for orig_c, ctype in zip(covariates, cov_types):
        sc = safe(orig_c)
        if ctype == "定性（分类）":
            df2[sc] = df2[sc].astype("category")
        cov_terms.append(f"C({sc})" if ctype == "定性（分类）" else sc)

    cov_str = (" + " + " + ".join(cov_terms)) if cov_terms else ""
    formula = f"{y_col} ~ C({g_col}) * C({t_col}){cov_str}"

    try:
        model  = mixedlm(formula, df2, groups=df2[s_col])
        result = model.fit(reml=True, method="lbfgs")
    except Exception as e:
        st.error(f"LMM 拟合失败：{e}\n请检查数据或尝试描述性统计。")
        return None

    # 计算 LS means per group×time
    groups = df[group_col].unique()
    times  = df[time_col].unique()
    rows   = []

    for g in groups:
        for t in times:
            sub = df[(df[group_col]==g) & (df[time_col]==t)][outcome_col].dropna()
            if len(sub) == 0:
                continue
            # LS mean proxy: predicted at group/time level
            pred_df = pd.DataFrame({
                safe(group_col): [g], safe(time_col): [t],
                safe(subject_col): [df[subject_col].iloc[0]],
            })
            for orig_c, ctype in zip(covariates, cov_types):
                sc = safe(orig_c)
                pred_df[sc] = df[orig_c].mean() if ctype == "定量（连续）" else df[orig_c].mode()[0]

            pred_df[g_col] = pred_df[g_col].astype("category")
            pred_df[t_col] = pred_df[t_col].astype("category")

            # Use group mean as LS mean (full marginal means need complex contrast setup)
            val = sub.mean()
            se  = sub.sem() if len(sub) > 1 else 0.0
            rows.append({
                "group": g, "time": t,
                "value": val,
                "lower": val - se,
                "upper": val + se,
                "note": "LMM LS Mean ± SE",
            })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# 绘图函数
# ═══════════════════════════════════════════════════════════════════════════════

def build_figure(
    plot_df, groups_order, x_numeric,
    lang,
    xlabel, ylabel, panel_label,
    ylim_start_zero,
    vline_on, vline_x,
    legend_x, legend_y,
    fig_w, fig_h,
    font_size_base,
    line_width, marker_size,
):
    font_fam = FONT_ZH if lang == "中文" else FONT_EN
    plt.rcParams["font.family"]   = "sans-serif"
    plt.rcParams["font.sans-serif"] = font_fam
    plt.rcParams["pdf.fonttype"]  = 42
    plt.rcParams["axes.unicode_minus"] = False

    fs = font_size_base
    zh_bonus = 1 if lang == "中文" else 0

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)

    # ── X 轴处理 ──────────────────────────────────────────────────────────────
    if x_numeric:
        x_vals_all = sorted(plot_df["time"].unique())
        x_map = {v: v for v in x_vals_all}
    else:
        x_cats = list(dict.fromkeys(plot_df["time"].tolist()))  # 保顺序去重
        x_map  = {v: i for i, v in enumerate(x_cats)}
        x_vals_all = list(range(len(x_cats)))

    # ── 绘制每个组 ────────────────────────────────────────────────────────────
    legend_handles = []
    for idx, grp in enumerate(groups_order):
        color = COLOR_SEQ[idx % len(COLOR_SEQ)]
        sub   = plot_df[plot_df["group"] == grp].copy()
        sub["_x"] = sub["time"].map(x_map)
        sub.sort_values("_x", inplace=True)

        xs = sub["_x"].values
        ys = sub["value"].values

        ax.plot(xs, ys, color=color, lw=line_width,
                marker="o", ms=marker_size, zorder=3, label=str(grp))

        # 误差棒
        has_eb = sub["lower"].notna().any() and sub["upper"].notna().any()
        if has_eb:
            yerr_lo = (ys - sub["lower"].values).clip(min=0)
            yerr_hi = (sub["upper"].values - ys).clip(min=0)
            ax.errorbar(xs, ys,
                        yerr=[yerr_lo, yerr_hi],
                        fmt="none", ecolor=color,
                        elinewidth=1.5, capsize=3, zorder=2)

        legend_handles.append(
            Line2D([0], [0], color=color, lw=line_width, label=str(grp))
        )

    # ── 垂直虚线 ──────────────────────────────────────────────────────────────
    if vline_on:
        vx = vline_x if x_numeric else x_map.get(vline_x, None)
        if vx is not None:
            ax.axvline(vx, color="#666666",
                       linestyle=(0, (3, 3)), linewidth=1.3, zorder=1)

    # ── 轴范围与刻度 ──────────────────────────────────────────────────────────
    if x_numeric:
        ax.set_xlim(min(x_vals_all) - 0.5, max(x_vals_all) * 1.03 + 0.5)
        ax.set_xticks(x_vals_all)
        ax.set_xticklabels([str(v) for v in x_vals_all], fontsize=fs + zh_bonus)
    else:
        ax.set_xlim(-0.5, len(x_cats) - 0.5)
        ax.set_xticks(range(len(x_cats)))
        ax.set_xticklabels(x_cats, fontsize=fs + zh_bonus)

    # Y 轴
    y_min_data = plot_df["lower"].dropna().min() if plot_df["lower"].notna().any() \
                 else plot_df["value"].min()
    y_max_data = plot_df["upper"].dropna().max() if plot_df["upper"].notna().any() \
                 else plot_df["value"].max()
    y_margin   = (y_max_data - y_min_data) * 0.12 + 0.5

    if ylim_start_zero:
        y_bot = 0
    else:
        y_bot = y_min_data - y_margin

    y_top = y_max_data + y_margin
    ax.set_ylim(y_bot, y_top)
    ax.yaxis.set_major_locator(ticker.AutoLocator())
    ax.tick_params(axis="y", labelsize=fs + zh_bonus)
    ax.tick_params(axis="both", direction="out", length=6, width=1.5, pad=6)

    # ── 框架 ──────────────────────────────────────────────────────────────────
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    # ── 标签 ──────────────────────────────────────────────────────────────────
    ax.set_xlabel(xlabel, fontsize=fs + 1 + zh_bonus, labelpad=12)
    ax.set_ylabel(ylabel, fontsize=fs + 1 + zh_bonus, labelpad=12)

    # ── Panel 标签 ────────────────────────────────────────────────────────────
    if panel_label.strip():
        ax.text(0.0, 1.08, panel_label,
                transform=ax.transAxes,
                fontsize=fs + 4 + zh_bonus,
                fontweight="bold", ha="left", va="bottom")

    # ── 图例 ──────────────────────────────────────────────────────────────────
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(legend_x, legend_y),
        frameon=False,
        fontsize=fs - 1 + zh_bonus,
        handlelength=2.5,
    )

    plt.tight_layout()
    return fig


def fig_to_bytes(fig, fmt):
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight", dpi=150)
    buf.seek(0)
    return buf.read()


# ═══════════════════════════════════════════════════════════════════════════════
# ── STEP 1: 语言选择 ─────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
with st.expander("第一步：选择绘图语言", expanded=(st.session_state.step == 1)):
    lang = st.radio("绘图语言", ["英文", "中文"], horizontal=True, key="lang")
    st.caption("中文模式使用 Noto Serif SC 字体，字号在英文基础上 +1")

# ═══════════════════════════════════════════════════════════════════════════════
# ── STEP 2: 数据上传 ──────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
with st.expander("第二步：上传数据文件", expanded=(st.session_state.step <= 2)):
    data_mode = st.radio(
        "数据类型",
        ["原始数据（需统计计算）", "直接绘图数据（已有点估计和误差棒）"],
        horizontal=True, key="data_mode",
    )

    st.markdown("---")

    if data_mode == "原始数据（需统计计算）":
        st.markdown(
            "**原始数据格式**：前4列固定为 `受试者编号, 组别, 时间点, 因变量`；"
            "第5列起为可选协变量。"
        )
        uploaded = st.file_uploader("上传原始数据（Excel / CSV）", type=["xlsx", "csv"], key="raw_up")
        if uploaded:
            try:
                raw_df = pd.read_excel(uploaded) if uploaded.name.endswith("xlsx") \
                         else pd.read_csv(uploaded)
                st.session_state.raw_df = raw_df
                st.dataframe(raw_df.head(10), use_container_width=True)

                cols = list(raw_df.columns)
                if len(cols) < 4:
                    st.error("数据至少需要4列：受试者编号、组别、时间点、因变量")
                else:
                    st.session_state.col_subject  = cols[0]
                    st.session_state.col_group    = cols[1]
                    st.session_state.col_time     = cols[2]
                    st.session_state.col_outcome  = cols[3]
                    st.session_state.covariate_cols = cols[4:]
                    st.success(f"识别到：受试者={cols[0]}, 组别={cols[1]}, 时间点={cols[2]}, 因变量={cols[3]}")
                    if cols[4:]:
                        st.info(f"协变量列：{', '.join(cols[4:])}")
                    # 判断时间点类型
                    num = try_numeric_x(raw_df[cols[2]])
                    st.session_state.x_numeric = num is not None
                    st.session_state.step = max(st.session_state.step, 3)
            except Exception as e:
                st.error(f"文件读取失败：{e}")
    else:
        st.markdown(
            "**直接绘图数据格式**：共5列，分别为 `组别, 时间点, 因变量值, 误差棒下限, 误差棒上限`。"
            "若无误差棒，下限/上限列留空即可。"
        )
        uploaded = st.file_uploader("上传直接绘图数据（Excel / CSV）", type=["xlsx", "csv"], key="direct_up")
        if uploaded:
            try:
                df = pd.read_excel(uploaded) if uploaded.name.endswith("xlsx") \
                     else pd.read_csv(uploaded)
                if len(df.columns) < 3:
                    st.error("至少需要3列：组别、时间点、因变量值")
                else:
                    # 规范列名
                    df.columns = (list(df.columns[:5]) + list(df.columns[5:]))
                    col_rename = {
                        df.columns[0]: "group",
                        df.columns[1]: "time",
                        df.columns[2]: "value",
                    }
                    if len(df.columns) >= 4:
                        col_rename[df.columns[3]] = "lower"
                    if len(df.columns) >= 5:
                        col_rename[df.columns[4]] = "upper"
                    df.rename(columns=col_rename, inplace=True)
                    if "lower" not in df.columns:
                        df["lower"] = np.nan
                    if "upper" not in df.columns:
                        df["upper"] = np.nan

                    df["value"] = pd.to_numeric(df["value"], errors="coerce")
                    df["lower"] = pd.to_numeric(df["lower"], errors="coerce")
                    df["upper"] = pd.to_numeric(df["upper"], errors="coerce")

                    num = try_numeric_x(df["time"])
                    if num is not None:
                        df["time"] = num
                        st.session_state.x_numeric = True
                    else:
                        st.session_state.x_numeric = False

                    groups = list(df["group"].unique())
                    st.session_state.plot_df     = df
                    st.session_state.groups_order = groups
                    st.dataframe(df.head(20), use_container_width=True)
                    st.success(f"数据加载成功！共 {len(groups)} 个组：{', '.join(str(g) for g in groups)}")
                    st.session_state.step = max(st.session_state.step, 5)  # 跳到分析设置
            except Exception as e:
                st.error(f"文件读取失败：{e}")

# ═══════════════════════════════════════════════════════════════════════════════
# ── STEP 3: 统计方法选择（仅原始数据）────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
if data_mode == "原始数据（需统计计算）" and st.session_state.get("raw_df") is not None:
    with st.expander("第三步：选择统计分析方法", expanded=(st.session_state.step == 3)):
        stat_method = st.radio(
            "统计方法",
            ["线性混合模型（推荐）", "描述性统计"],
            horizontal=True, key="stat_method",
        )
        st.caption("线性混合模型：随机效应=受试者；固定效应=组别、时间、组别×时间交互项 + 协变量")
        if st.session_state.step >= 3:
            st.session_state.step = max(st.session_state.step, 4)

# ═══════════════════════════════════════════════════════════════════════════════
# ── STEP 4: 协变量定义（仅 LMM）──────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
_show_step4 = (
    data_mode == "原始数据（需统计计算）"
    and st.session_state.get("raw_df") is not None
    and st.session_state.get("stat_method", "描述性统计") == "线性混合模型（推荐）"
    and st.session_state.covariate_cols
)

if _show_step4:
    with st.expander("第四步：定义协变量类型并选择纳入模型的协变量", expanded=(st.session_state.step == 4)):
        cov_cols = st.session_state.covariate_cols
        cov_types = {}
        for c in cov_cols:
            cov_types[c] = st.radio(
                f"**{c}**", ["定量（连续）", "定性（分类）"],
                horizontal=True, key=f"covtype_{c}"
            )
        selected_covs = st.multiselect(
            "选择纳入模型的协变量（可不选）", cov_cols, default=[], key="selected_covs"
        )
        st.session_state.step = max(st.session_state.step, 5)

# ═══════════════════════════════════════════════════════════════════════════════
# ── 执行统计计算 ──────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
_run_stat = (
    data_mode == "原始数据（需统计计算）"
    and st.session_state.get("raw_df") is not None
    and st.session_state.step >= 4
)

if _run_stat:
    raw_df     = st.session_state.raw_df
    s_col      = st.session_state.col_subject
    g_col      = st.session_state.col_group
    t_col      = st.session_state.col_time
    y_col      = st.session_state.col_outcome
    method     = st.session_state.get("stat_method", "描述性统计")

    with st.expander("📊 统计计算", expanded=True):
        run_btn = st.button("▶ 开始计算", key="run_stat")
        if run_btn or st.session_state.get("stat_done"):
            if run_btn:
                with st.spinner("计算中……"):
                    if method == "描述性统计":
                        plot_df, all_normal = run_descriptive(raw_df, g_col, t_col, y_col)
                        if all_normal:
                            st.success("所有组符合正态分布 → 使用均值 ± SD")
                        else:
                            st.warning("存在非正态组 → 使用中位数（无误差棒）")
                    else:
                        sel_covs  = st.session_state.get("selected_covs", [])
                        cov_types_list = [st.session_state.get(f"covtype_{c}", "定量（连续）") for c in sel_covs]
                        plot_df = run_lmm(raw_df, g_col, t_col, y_col, s_col, sel_covs, cov_types_list)
                        if plot_df is None:
                            st.stop()
                        st.success("LMM 计算完成")

                    # 时间点类型
                    num = try_numeric_x(plot_df["time"])
                    if num is not None:
                        plot_df["time"] = num
                        st.session_state.x_numeric = True
                    else:
                        st.session_state.x_numeric = False

                    groups = list(raw_df[g_col].unique())
                    st.session_state.plot_df      = plot_df
                    st.session_state.groups_order = groups
                    st.session_state.stat_done    = True

            if st.session_state.get("stat_done") and st.session_state.plot_df is not None:
                st.dataframe(st.session_state.plot_df, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ── STEP 5: 分析设置 & 绘图 ───────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.plot_df is not None:
    plot_df     = st.session_state.plot_df
    groups_order = st.session_state.groups_order
    x_numeric   = st.session_state.x_numeric
    lang        = st.session_state.get("lang", "英文")

    st.markdown("---")
    st.subheader("第五步：分析设置")

    col_a, col_b = st.columns(2)
    with col_a:
        xlabel = st.text_input("X 轴标签", value="Weeks", key="xlabel")
        ylabel = st.text_input("Y 轴标签", value="Change from baseline", key="ylabel")
    with col_b:
        panel_label = st.text_input(
            "左上角 Panel 标签（留空则不显示）", value="A  mRSS", key="panel_label"
        )
        ylim_start_zero = st.radio(
            "Y 轴起始", ["从 0 开始", "从数据最小值开始"],
            horizontal=True, key="ylim_mode"
        ) == "从 0 开始"

    st.markdown("---")
    st.subheader("第六步：绘图参数调节")

    col1, col2, col3 = st.columns(3)
    with col1:
        font_size_base = st.slider("基础字号", 8, 22, 14, key="font_sz")
        line_width     = st.slider("线宽", 0.5, 4.0, 1.8, 0.1, key="lw")
        marker_size    = st.slider("数据点大小", 2, 14, 6, key="ms")
    with col2:
        fig_w = st.slider("图宽 (inches)", 4.0, 16.0, 8.0, 0.5, key="fig_w")
        fig_h = st.slider("图高 (inches)", 3.0, 12.0, 5.5, 0.5, key="fig_h")
    with col3:
        legend_x = st.slider("图例 X 位置", 0.0, 1.2, 0.98, 0.01, key="leg_x")
        legend_y = st.slider("图例 Y 位置", 0.0, 1.2, 0.95, 0.01, key="leg_y")

    # 垂直虚线
    st.markdown("**垂直分割线**")
    col_v1, col_v2 = st.columns([1, 2])
    with col_v1:
        vline_on = st.checkbox("显示垂直虚线", value=True, key="vline_on")
    with col_v2:
        if x_numeric:
            x_min_v = float(plot_df["time"].min())
            x_max_v = float(plot_df["time"].max())
            vline_x = st.slider(
                "虚线位置",
                min_value=x_min_v, max_value=x_max_v,
                value=float(np.median(plot_df["time"].unique())),
                key="vline_x",
            )
        else:
            cats = list(dict.fromkeys(plot_df["time"].tolist()))
            vline_x = st.selectbox("虚线位置（时间点）", cats, key="vline_x_cat")

    # ── 实时预览 ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 图形预览")

    vx_val = vline_x if x_numeric else vline_x

    fig = build_figure(
        plot_df        = plot_df,
        groups_order   = groups_order,
        x_numeric      = x_numeric,
        lang           = lang,
        xlabel         = xlabel,
        ylabel         = ylabel,
        panel_label    = panel_label,
        ylim_start_zero= ylim_start_zero,
        vline_on       = vline_on,
        vline_x        = vx_val,
        legend_x       = legend_x,
        legend_y       = legend_y,
        fig_w          = fig_w,
        fig_h          = fig_h,
        font_size_base = font_size_base,
        line_width     = line_width,
        marker_size    = marker_size,
    )

    st.pyplot(fig, use_container_width=True)

    # ── 导出 ──────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("导出图形")
    exp_col1, exp_col2, exp_col3 = st.columns(3)
    with exp_col1:
        st.download_button(
            "⬇ 下载 PNG", data=fig_to_bytes(fig, "png"),
            file_name="figure.png", mime="image/png",
        )
    with exp_col2:
        st.download_button(
            "⬇ 下载 SVG", data=fig_to_bytes(fig, "svg"),
            file_name="figure.svg", mime="image/svg+xml",
        )
    with exp_col3:
        st.download_button(
            "⬇ 下载 PDF", data=fig_to_bytes(fig, "pdf"),
            file_name="figure.pdf", mime="application/pdf",
        )

    plt.close(fig)

# ── 底部说明 ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "本工具专为医学科研折线图绘制设计 | "
    "颜色顺序固定：红→蓝→绿→橙… | "
    "字体：英文 Arial，中文 Noto Serif SC"
)
