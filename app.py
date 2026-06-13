# ============================================================
# 生存曲线分析工具 · Streamlit 版
# 功能：多组KM曲线 + Log-rank两两检验 + Cox HR + 协变量选择调整
# ============================================================

import io
import re
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch
import streamlit as st
from itertools import combinations

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines.utils import median_survival_times
from sklearn.preprocessing import LabelEncoder

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
# 邀请码验证
# ============================================================
INVITE_CODE = "WHU2026"

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#00558C,#003a63);
                border-radius:10px;padding:18px 24px;margin-bottom:24px;">
      <h2 style="color:white;margin:0;font-family:Arial,sans-serif;">
        📊 生存曲线分析工具
      </h2>
      <p style="color:#cce4f7;margin:6px 0 0;font-size:13px;">
        支持多组 KM 曲线 · Log-rank 两两检验 · Cox 比例风险回归
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("🔐 请输入邀请码以继续")
    code_input = st.text_input("邀请码", type="password", placeholder="请输入邀请码")
    if st.button("验证", type="primary"):
        if code_input == INVITE_CODE:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("❌ 邀请码错误，请重新输入。")
    st.stop()

# ============================================================
# 语言标签
# ============================================================
LABELS = {
    "zh": {
        "median_header":   "中位时间（95% CI）",
        "risk_header":     "风险人数（删失数）",
        "overall_logrank": "总体 Log-rank 检验",
        "hr_label":        "HR",
    },
    "en": {
        "median_header":   "Median time (95% CI)",
        "risk_header":     "Number at risk\n(censored)",
        "overall_logrank": "Overall Log-rank test",
        "hr_label":        "HR",
    },
}

def L(key):
    return LABELS[st.session_state.get("lang", "en")][key]

# ============================================================
# 配色
# ============================================================
PRESET_COLORS = [
    "#00558C", "#D4820A", "#2E8B57", "#C84B31",
    "#6A0DAD", "#007272", "#1B6CA8", "#8B008B",
    "#556B2F", "#B8860B",
]

def assign_colors(groups):
    n = len(PRESET_COLORS)
    return {g: PRESET_COLORS[i % n] for i, g in enumerate(groups)}

# ============================================================
# 字体设置
# ============================================================
def _setup_font(lang="en"):
    available = {f.name for f in fm.fontManager.ttflist}
    _BASE_FONTSIZE = 14

    if lang == "zh":
    # 仅使用可自由商业使用或开源字体
     zh_candidates = [
        "Noto Serif SC",
        "Source Han Serif SC",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
    ]

    for candidate in zh_candidates:
        if candidate in available:
            plt.rcParams.update({
                "font.family": "sans-serif",
                "font.sans-serif": [candidate, "DejaVu Sans"],
                "axes.unicode_minus": False,
                "font.size": _BASE_FONTSIZE,
            })
            return candidate

    # 尝试自动下载 Noto Serif SC（推荐）
    try:
        import urllib.request
        import os

        font_url = (
            "https://github.com/notofonts/noto-cjk/raw/main/"
            "Serif/OTF/SimplifiedChinese/"
            "NotoSerifCJKsc-Regular.otf"
        )

        font_path = "/tmp/NotoSerifCJKsc-Regular.otf"

        if not os.path.exists(font_path):
            urllib.request.urlretrieve(font_url, font_path)

        fm.fontManager.addfont(font_path)

        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Serif SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": _BASE_FONTSIZE,
        })

        return "Noto Serif SC"

    except Exception:
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": _BASE_FONTSIZE,
        })

        return "default"

    else:
     _fs = {
        "font.size": _BASE_FONTSIZE - 1,
        "axes.titlesize": _BASE_FONTSIZE,
        "axes.labelsize": _BASE_FONTSIZE,
        "xtick.labelsize": _BASE_FONTSIZE - 1,
        "ytick.labelsize": _BASE_FONTSIZE - 1,
        "legend.fontsize": _BASE_FONTSIZE - 1,
    }

    for candidate in ["Arial", "Liberation Sans", "FreeSans", "DejaVu Sans"]:
        if candidate in available:
            plt.rcParams.update({
                "font.family": "sans-serif",
                "font.sans-serif": [candidate],
                "axes.unicode_minus": False,
                **_fs,
            })
            return candidate

    plt.rcParams.update({
        "axes.unicode_minus": False,
        **_fs,
    })

    return "default"
# ============================================================
# 工具函数
# ============================================================
def auto_xmax_ticks(max_time):
    def _nice_ceil(val, candidates):
        for c in candidates:
            if c >= val:
                return c
        mag = 10 ** math.ceil(math.log10(val))
        return math.ceil(val / mag) * mag

    if max_time <= 0:
        return 10, [0, 2, 4, 6, 8, 10]
    if max_time < 20:
        candidates = list(range(5, 105, 5))
        x_max = _nice_ceil(max_time, candidates)
        step  = max(1, x_max // 5)
    elif max_time < 60:
        candidates = [30, 36, 42, 48, 54, 60]
        x_max = _nice_ceil(max_time, candidates)
        step  = x_max // 6
    elif max_time < 120:
        candidates = [60, 72, 84, 90, 96, 108, 120]
        x_max = _nice_ceil(max_time, candidates)
        step  = x_max // 6
    elif max_time < 500:
        x_max = math.ceil(max_time / 50) * 50
        step  = x_max // 5
    elif max_time < 1000:
        x_max = math.ceil(max_time / 100) * 100
        step  = 100 if x_max / 100 <= 10 else 200
    elif max_time < 3000:
        x_max = math.ceil(max_time / 200) * 200
        step  = 200 if x_max / 200 <= 10 else 500
    else:
        x_max = math.ceil(max_time / 500) * 500
        step  = 500

    ticks = list(range(0, x_max + 1, step))
    if ticks[-1] != x_max:
        ticks.append(x_max)
    return x_max, ticks


def get_median_ci(kmf):
    median = kmf.median_survival_time_
    if np.isinf(median) or pd.isna(median):
        return "NE", "NE"
    try:
        ci    = median_survival_times(kmf.confidence_interval_)
        lower = float(ci.iloc[0, 0])
        upper = float(ci.iloc[0, 1])
        ls = "NE" if (np.isinf(lower) or pd.isna(lower)) else f"{lower:.1f}"
        us = "NE" if (np.isinf(upper) or pd.isna(upper)) else f"{upper:.1f}"
        return f"{median:.1f}", f"{ls}\u2013{us}"
    except:
        return f"{median:.1f}", "NE"


def fmt_p(p):
    if p < 0.0001:  return "P<0.0001"
    elif p < 0.001: return f"P={p:.4f}"
    else:           return f"P={p:.3f}"


def _italic_P(text):
    return re.sub(r'\bP([=<])', r'$\\mathit{P}$\1', text)


# ============================================================
# 核心分析
# ============================================================
def run_analysis(df, group_col, time_col, event_col, groups, colors,
                 selected_covariates, covariate_types):
    df = df.copy()
    df[time_col]  = pd.to_numeric(df[time_col],  errors="coerce")
    df[event_col] = pd.to_numeric(df[event_col], errors="coerce")
    df = df.dropna(subset=[time_col, event_col])

    # KM 拟合
    kmf_dict  = {}
    group_dfs = {}
    for g in groups:
        sub = df[df[group_col] == g].copy()
        group_dfs[g] = sub
        kmf = KaplanMeierFitter()
        kmf.fit(sub[time_col], sub[event_col], label=str(g))
        kmf_dict[g] = kmf

    # Log-rank 两两比较
    pairwise_p = {}
    for g1, g2 in combinations(groups, 2):
        lr = logrank_test(
            group_dfs[g1][time_col], group_dfs[g2][time_col],
            event_observed_A=group_dfs[g1][event_col],
            event_observed_B=group_dfs[g2][event_col],
        )
        pairwise_p[(g1, g2)] = lr.p_value

    # Log-rank 整体检验（≥3组）
    overall_p = None
    if len(groups) >= 3:
        try:
            overall_result = multivariate_logrank_test(df[time_col], df[group_col], df[event_col])
            overall_p = overall_result.p_value
        except Exception:
            pass

    # Cox HR（全两两）
    hr_texts = []
    n_pairs  = len(list(combinations(groups, 2)))
    for g1, g2 in combinations(groups, 2):
        cox_sub = df[df[group_col].isin([g1, g2])].copy()
        cox_sub["_trt"] = (cox_sub[group_col] == g2).astype(int)
        fit_cols = [time_col, event_col, "_trt"]
        for col in selected_covariates:
            if col in cox_sub.columns:
                ctype = covariate_types.get(col, "quantitative")
                if ctype == "qualitative":
                    le = LabelEncoder()
                    cox_sub[col] = le.fit_transform(cox_sub[col].astype(str))
                else:
                    cox_sub[col] = pd.to_numeric(cox_sub[col], errors="coerce")
                fit_cols.append(col)
        cox_sub = cox_sub[fit_cols].dropna()

        p_str = fmt_p(pairwise_p[(g1, g2)])
        label = f"HR ({g2} vs {g1})" if n_pairs > 1 else "HR"

        if len(cox_sub) < 5:
            hr_texts.append(f"{label}: 样本量不足")
            continue
        try:
            cph = CoxPHFitter()
            cph.fit(cox_sub, duration_col=time_col, event_col=event_col)
            hr  = np.exp(cph.params_["_trt"])
            cil = np.exp(cph.confidence_intervals_.loc["_trt"].iloc[0])
            ciu = np.exp(cph.confidence_intervals_.loc["_trt"].iloc[1])
            hr_texts.append(
                f"{label}: {hr:.2f} (95% CI {cil:.2f}\u2013{ciu:.2f}); {p_str}"
            )
        except Exception as e:
            hr_texts.append(f"{label}: Cox拟合失败 ({e})")

    return {
        "groups":     groups,
        "kmf_dict":   kmf_dict,
        "group_dfs":  group_dfs,
        "colors":     colors,
        "hr_texts":   hr_texts,
        "pairwise_p": pairwise_p,
        "overall_p":  overall_p,
        "time_col":   time_col,
        "event_col":  event_col,
    }


# ============================================================
# 绘图核心
# ============================================================
def build_figure(analysis, state,
                 text_x=0.02, text_y=0.42,
                 median_text_override=None,
                 hr_text_override=None,
                 logrank_text_override=None,
                 lr_x=0.98, lr_y=0.08,
                 show_ci=True,
                 legend_x=0.98, legend_y=0.98):

    groups    = analysis["groups"]
    kmf_dict  = analysis["kmf_dict"]
    group_dfs = analysis["group_dfs"]
    colors    = analysis["colors"]
    hr_texts  = hr_text_override if hr_text_override is not None else analysis["hr_texts"]
    overall_p = analysis.get("overall_p", None)
    x_max     = state["x_max"]
    x_ticks   = state["x_ticks"]
    tc        = analysis["time_col"]
    ec        = analysis["event_col"]
    n_groups  = len(groups)

    _en = (state.get("lang", "en") == "en")
    FS_TICK   = 13 if _en else 14
    FS_LABEL  = 14 if _en else 15
    FS_LEGEND = 13 if _en else 14
    FS_TEXT   = 12 if _en else 13
    FS_TABLE  = 12 if _en else 13
    FS_THEAD  = 13 if _en else 14
    plt.rcParams.update({
        "font.size":       FS_TICK,
        "axes.titlesize":  FS_LABEL,
        "axes.labelsize":  FS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEGEND,
    })

    ROW_H_INCH = 0.38
    tbl_h  = (n_groups + 3.5) * ROW_H_INCH + 0.20
    main_h = 8.5 * 0.8
    fig_h  = main_h + tbl_h

    fig = plt.figure(figsize=(14, fig_h), dpi=150)
    gs  = fig.add_gridspec(2, 1, height_ratios=[main_h, tbl_h], hspace=0.05)
    ax  = fig.add_subplot(gs[0])

    # KM 曲线
    for g in groups:
        kmf = kmf_dict[g]
        col = colors[g]
        sf  = kmf.survival_function_ * 100
        ax.step(sf.index, sf.iloc[:, 0], where="post", color=col, lw=2.0, label=str(g))
        if show_ci:
            ci_df = kmf.confidence_interval_ * 100
            ax.fill_between(ci_df.index, ci_df.iloc[:, 0], ci_df.iloc[:, 1],
                            step="post", alpha=0.15, color=col)
        cens = group_dfs[g][group_dfs[g][ec] == 0]
        if len(cens):
            yvals = kmf.survival_function_at_times(cens[tc]) * 100
            ax.scatter(cens[tc], yvals, marker="|", color=col, s=60, lw=1.4, zorder=5)

    # 坐标轴
    _data_max = state.get("x_data_max") or x_max
    x_right   = _data_max * 1.05
    ax.set_xlim(0, x_right)
    ax.set_ylim(0, 100)
    ax.set_xticks(x_ticks)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel(state["y_label"], fontsize=FS_LABEL)
    ax.set_xlabel("")

    # 图例
    ax.legend(frameon=False,
              bbox_to_anchor=(legend_x, legend_y),
              loc="upper right",
              bbox_transform=ax.transAxes,
              fontsize=FS_LEGEND)

    # Median + HR 文字块
    line_h     = 0.048 / 0.8
    line_h     = min(line_h, 0.065)
    col_group  = text_x + 0.008
    col_median = text_x + 0.008 + 0.18

    median_header = L("median_header")
    if median_text_override is not None:
        median_override_lines = median_text_override
        use_override_median   = True
    else:
        use_override_median = False
        median_rows = []
        for g in groups:
            med, ci_s = get_median_ci(kmf_dict[g])
            median_rows.append((str(g), f"{med} ({ci_s})"))

    hr_lines = hr_texts

    if use_override_median:
        n_median_lines = len(median_override_lines)
        n_lines = n_median_lines + 1 + len(hr_lines)
    else:
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

    if use_override_median:
        for i, line in enumerate(median_override_lines):
            y_pos = text_y - i * line_h
            fw = "bold" if i == 0 else "normal"
            ax.text(col_group, y_pos, line,
                    transform=ax.transAxes, fontsize=FS_TEXT,
                    va="top", fontweight=fw, zorder=5)
        hr_y_start = text_y - (n_median_lines + 1) * line_h
    else:
        ax.text(col_group, text_y, median_header,
                transform=ax.transAxes, fontsize=FS_TEXT,
                va="top", fontweight="bold", zorder=5)
        for i, (g_str, med_ci_str) in enumerate(median_rows):
            y_pos = text_y - (i + 1) * line_h
            ax.text(col_group,  y_pos, g_str,      transform=ax.transAxes,
                    fontsize=FS_TEXT, va="top", zorder=5)
            ax.text(col_median, y_pos, med_ci_str, transform=ax.transAxes,
                    fontsize=FS_TEXT, va="top", ha="left", zorder=5)
        hr_y_start = text_y - (len(median_rows) + 2) * line_h

    for i, ht in enumerate(hr_lines):
        ht_display = _italic_P(ht)
        ax.text(col_group, hr_y_start - i * line_h, ht_display,
                transform=ax.transAxes, fontsize=FS_TEXT, va="top", zorder=5)

    # Log-rank 整体值文字框（≥3组）
    if n_groups >= 3:
        if logrank_text_override is not None:
            lr_text = logrank_text_override
        else:
            if overall_p is not None:
                lr_text = f"{L('overall_logrank')}: {fmt_p(overall_p)}"
            else:
                lr_text = f"{L('overall_logrank')}: N/A"
        if lr_text.strip():
            lr_display = _italic_P(lr_text)
            ax.text(lr_x, lr_y, lr_display,
                    transform=ax.transAxes, fontsize=FS_TEXT,
                    va="bottom", ha="right",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="none", edgecolor="none", alpha=0.0),
                    zorder=5)

    # 风险表
    ax_tbl = fig.add_subplot(gs[1])
    ax_tbl.axis("off")
    ax_tbl.set_xlim(0, x_right)
    ax_tbl.set_ylim(0, 1)

    row_unit = ROW_H_INCH / tbl_h
    top_pad  = 0.5 * row_unit
    xlabel_y = 1.0 - top_pad

    gap_xlabel_to_header = 1.5 * row_unit - 1.0 * row_unit
    header_y = xlabel_y - row_unit - gap_xlabel_to_header

    gap_header_to_data = 1.5 * row_unit
    first_row_y = header_y - row_unit - gap_header_to_data

    ax_tbl.text(x_right / 2, xlabel_y, state["x_label"],
                ha="center", va="top", fontsize=FS_THEAD)
    ax_tbl.text(-x_right * 0.065, header_y,
                L("risk_header"),
                ha="right", va="top",
                fontsize=FS_THEAD, fontweight="bold")

    for i, g in enumerate(groups):
        gdf   = group_dfs[g]
        row_y = first_row_y - i * row_unit
        ax_tbl.text(-x_right * 0.065, row_y, str(g),
                    ha="right", va="center", fontsize=FS_THEAD, color="black")
        for t in x_ticks:
            n_risk = int((gdf[tc] >= t).sum())
            n_cens = int(((gdf[tc] <= t) & (gdf[ec] == 0)).sum())
            ax_tbl.text(t, row_y, f"{n_risk} ({n_cens})",
                        ha="center", va="center", fontsize=FS_TABLE)

    plt.subplots_adjust(left=0.16, right=0.95, top=0.96, bottom=0.02)

    buf_png = io.BytesIO()
    plt.savefig(buf_png, format="png", dpi=150, bbox_inches="tight")
    buf_png.seek(0)
    png_bytes = buf_png.read()

    buf_pdf = io.BytesIO()
    plt.savefig(buf_pdf, format="pdf", bbox_inches="tight")
    buf_pdf.seek(0)
    pdf_bytes = buf_pdf.read()

    plt.close()
    return {"png": png_bytes, "pdf": pdf_bytes}


# ============================================================
# Session state 初始化
# ============================================================
def init_state():
    defaults = {
        "lang":               "en",
        "font_name":          "Arial",
        "df":                 None,
        "groups":             [],
        "group_col":          None,
        "time_col":           None,
        "event_col":          None,
        "id_col":             None,
        "covariate_cols":     [],
        "covariate_types":    {},
        "selected_covariates": [],
        "group_colors":       {},
        "x_label":            "Time (months)",
        "y_label":            "Survival probability (%)",
        "x_max":              36,
        "x_ticks":            [0, 6, 12, 18, 24, 30, 36],
        "x_data_max":         None,
        "show_ci":            True,
        "analysis":           None,
        "step":               0,
        # 位置调整参数
        "text_x":             0.02,
        "text_y":             0.42,
        "leg_x":              0.98,
        "leg_y":              0.98,
        "lr_x":               0.98,
        "lr_y":               0.08,
        "median_text":        "",
        "hr_text":            "",
        "lr_text":            "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ============================================================
# 页面标题
# ============================================================
st.markdown("""
<div style="background:linear-gradient(135deg,#00558C,#003a63);
            border-radius:10px;padding:18px 24px;margin-bottom:16px;">
  <h2 style="color:white;margin:0;font-family:Arial,sans-serif;">
    📊 生存曲线分析工具
  </h2>
  <p style="color:#cce4f7;margin:6px 0 0;font-size:13px;">
    支持多组 KM 曲线 · Log-rank 两两检验 · Cox 比例风险回归
  </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 进度指示
# ============================================================
step = st.session_state["step"]
steps_labels = ["语言", "上传数据", "协变量类型", "选择协变量", "分析设置", "结果与下载"]
cols_prog = st.columns(len(steps_labels))
for i, (col, label) in enumerate(zip(cols_prog, steps_labels)):
    with col:
        if i < step:
            st.markdown(f"<div style='text-align:center;color:#2E8B57;font-size:12px;'>✅ {label}</div>", unsafe_allow_html=True)
        elif i == step:
            st.markdown(f"<div style='text-align:center;color:#00558C;font-weight:bold;font-size:12px;'>▶ {label}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='text-align:center;color:#aaa;font-size:12px;'>○ {label}</div>", unsafe_allow_html=True)

st.markdown("---")

# ============================================================
# STEP 0：选择绘图语言
# ============================================================
if step == 0:
    st.subheader("第 0 步：选择绘图语言")
    st.markdown("请选择图表中文字的显示语言。中文模式自动使用中文字体；英文模式使用 Sans-serif（Arial / DejaVu Sans）。")

    lang_choice = st.radio(
        "绘图语言",
        options=["English", "中文"],
        index=0 if st.session_state["lang"] == "en" else 1,
        horizontal=True,
    )

    if st.button("确认并继续 →", type="primary", key="btn_step0"):
        lang = "zh" if lang_choice == "中文" else "en"
        st.session_state["lang"] = lang
        font_name = _setup_font(lang=lang)
        st.session_state["font_name"] = font_name
        if lang == "zh":
            st.session_state["x_label"] = "时间（月）"
            st.session_state["y_label"] = "生存概率（%）"
        else:
            st.session_state["x_label"] = "Time (months)"
            st.session_state["y_label"] = "Survival probability (%)"
        st.session_state["step"] = 1
        st.rerun()

# ============================================================
# STEP 1：上传数据文件
# ============================================================
elif step == 1:
    st.subheader("第 1 步：上传数据文件")
    st.markdown("""
    请上传 Excel 文件（`.xlsx` / `.xls`）。  
    **前4列依次为：受试者编号 · 组别 · 观测时长 · 结局状态**（1=事件发生，0=删失）  
    第5列起为可选协变量。
    """)

    uploaded_file = st.file_uploader("选择 Excel 文件", type=["xlsx", "xls"])

    if uploaded_file is not None:
        try:
            df = pd.read_excel(io.BytesIO(uploaded_file.read()))
            cols = list(df.columns)
            if len(cols) < 4:
                st.error("❌ 文件列数不足，至少需要4列（编号、组别、时间、结局）。")
            else:
                df.rename(columns={
                    cols[0]: "__id__",
                    cols[1]: "__group__",
                    cols[2]: "__time__",
                    cols[3]: "__event__",
                }, inplace=True)

                st.session_state["df"]            = df
                st.session_state["id_col"]        = "__id__"
                st.session_state["group_col"]     = "__group__"
                st.session_state["time_col"]      = "__time__"
                st.session_state["event_col"]     = "__event__"
                st.session_state["covariate_cols"] = [
                    c for c in df.columns
                    if c not in ("__id__", "__group__", "__time__", "__event__")
                ]

                st.success(f"✅ 成功读取：{uploaded_file.name}，共 {len(df)} 行，"
                           f"检测到 {len(st.session_state['covariate_cols'])} 个协变量列")

                st.markdown("**数据预览（前5行）：**")
                preview_df = df.head(5).copy()
                preview_df.columns = [cols[i] if i < len(cols) else c for i, c in enumerate(preview_df.columns)]
                st.dataframe(preview_df)

                col_back, col_next = st.columns([1, 5])
                with col_back:
                    if st.button("← 返回", key="back_step1"):
                        st.session_state["step"] = 0
                        st.rerun()
                with col_next:
                    if st.button("确认并继续 →", type="primary", key="btn_step1"):
                        if st.session_state["covariate_cols"]:
                            st.session_state["step"] = 2
                        else:
                            st.session_state["step"] = 3
                        st.rerun()
        except Exception as e:
            st.error(f"❌ 读取失败：{e}")

    if st.button("← 返回语言设置", key="back_step1b"):
        st.session_state["step"] = 0
        st.rerun()

# ============================================================
# STEP 2：指定协变量类型
# ============================================================
elif step == 2:
    st.subheader("第 2 步：指定协变量类型")
    st.markdown("请为每个协变量指定类型。定性变量支持数字编码（0/1/2…）或文字（如男/女）。")

    cov_cols = st.session_state["covariate_cols"]
    cov_types_input = {}
    for col in cov_cols:
        current = st.session_state["covariate_types"].get(col, "quantitative")
        idx = 0 if current == "quantitative" else 1
        choice = st.selectbox(
            f"**{col}**",
            options=["定量变量（连续）", "定性变量（分类）"],
            index=idx,
            key=f"cov_type_{col}",
        )
        cov_types_input[col] = "quantitative" if "定量" in choice else "qualitative"

    col_back, col_next = st.columns([1, 5])
    with col_back:
        if st.button("← 返回", key="back_step2"):
            st.session_state["step"] = 1
            st.rerun()
    with col_next:
        if st.button("确认协变量类型 →", type="primary", key="btn_step2"):
            st.session_state["covariate_types"] = cov_types_input
            st.session_state["step"] = 3
            st.rerun()

# ============================================================
# STEP 3：选择纳入 Cox 的协变量（原 Step 2b）
# ============================================================
elif step == 3:
    st.subheader("第 2b 步：选择纳入 Cox 模型的协变量")
    st.markdown("请勾选需要纳入 Cox 比例风险回归模型进行调整的协变量。若不选择任何协变量，则进行无调整的单变量 Cox 回归。")

    cov_cols = st.session_state["covariate_cols"]

    if not cov_cols:
        st.info("没有检测到协变量列，将进行单变量 Cox 回归。")
        st.session_state["selected_covariates"] = []
        col_back, col_next = st.columns([1, 5])
        with col_back:
            if st.button("← 返回", key="back_step3_no_cov"):
                st.session_state["step"] = 1
                st.rerun()
        with col_next:
            if st.button("确认并继续 →", type="primary", key="btn_step3_no_cov"):
                st.session_state["step"] = 4
                st.rerun()
    else:
        selected_covs = []
        for col in cov_cols:
            ctype = st.session_state["covariate_types"].get(col, "quantitative")
            type_label = "定量" if ctype == "quantitative" else "定性"
            checked = st.checkbox(f"{col}  [{type_label}]", key=f"cov_sel_{col}")
            if checked:
                selected_covs.append(col)

        st.caption("若不选择任何协变量，则进行无调整的单变量Cox回归。")

        col_back, col_next = st.columns([1, 5])
        with col_back:
            if st.button("← 返回", key="back_step3"):
                st.session_state["step"] = 2
                st.rerun()
        with col_next:
            if st.button("确认并继续 →", type="primary", key="btn_step3"):
                st.session_state["selected_covariates"] = selected_covs
                st.session_state["step"] = 4
                st.rerun()

# ============================================================
# STEP 4：分析设置
# ============================================================
elif step == 4:
    st.subheader("第 3 步：分析设置")

    df = st.session_state["df"]
    groups = sorted(df["__group__"].dropna().unique().tolist(), key=lambda x: str(x))
    st.session_state["groups"] = groups

    try:
        max_t = pd.to_numeric(df["__time__"], errors="coerce").dropna().max()
        _auto_xmax, _auto_ticks = auto_xmax_ticks(float(max_t))
    except Exception:
        _auto_xmax, _auto_ticks = 36, [0, 6, 12, 18, 24, 30, 36]

    st.info(f"检测到 **{len(groups)}** 个组别：{', '.join(str(g) for g in groups)}  \n"
            f"HR 与 P 值将对所有组别进行两两比较")

    col1, col2 = st.columns(2)
    with col1:
        x_label = st.text_input("横轴标签", value=st.session_state["x_label"])
        x_max   = st.number_input("横轴最大值", value=int(_auto_xmax), min_value=1, step=1)
    with col2:
        y_label  = st.text_input("纵轴标签", value=st.session_state["y_label"])
        xticks_s = st.text_input(
            "横轴刻度（逗号分隔）",
            value=",".join(str(int(t)) for t in _auto_ticks),
        )

    col_back, col_next = st.columns([1, 5])
    with col_back:
        if st.button("← 返回", key="back_step4"):
            st.session_state["step"] = 3
            st.rerun()
    with col_next:
        if st.button("🚀 开始绘图分析", type="primary", key="btn_step4"):
            _data_max = float(pd.to_numeric(df["__time__"], errors="coerce").dropna().max())
            try:
                raw_ticks = [float(x.strip()) for x in xticks_s.split(",") if x.strip()]
                filtered_ticks = [t for t in raw_ticks if t <= _data_max]
                if not filtered_ticks:
                    filtered_ticks = raw_ticks
            except:
                filtered_ticks = list(range(0, int(x_max) + 1, 6))

            st.session_state["x_label"]    = x_label or "Time"
            st.session_state["y_label"]    = y_label or "Survival (%)"
            st.session_state["x_max"]      = int(x_max)
            st.session_state["x_ticks"]    = filtered_ticks
            st.session_state["x_data_max"] = _data_max
            st.session_state["group_colors"] = assign_colors(groups)

            with st.spinner("正在进行生存分析……"):
                analysis = run_analysis(
                    df=df,
                    group_col="__group__",
                    time_col="__time__",
                    event_col="__event__",
                    groups=groups,
                    colors=st.session_state["group_colors"],
                    selected_covariates=st.session_state["selected_covariates"],
                    covariate_types=st.session_state["covariate_types"],
                )
                st.session_state["analysis"] = analysis

                # 预设文字内容
                kmf_dict = analysis["kmf_dict"]
                lang = st.session_state["lang"]
                median_header = LABELS[lang]["median_header"]
                median_lines = [median_header]
                for g in groups:
                    med, ci_s = get_median_ci(kmf_dict[g])
                    median_lines.append(f"{g}    {med} ({ci_s})")
                st.session_state["median_text"] = "\n".join(median_lines)
                st.session_state["hr_text"]     = "\n".join(analysis["hr_texts"])

                overall_p = analysis.get("overall_p", None)
                if len(groups) >= 3:
                    overall_logrank_label = LABELS[lang]["overall_logrank"]
                    lr_default = (f"{overall_logrank_label}: {fmt_p(overall_p)}"
                                  if overall_p is not None
                                  else f"{overall_logrank_label}: N/A")
                    st.session_state["lr_text"] = lr_default
                else:
                    st.session_state["lr_text"] = ""

            st.session_state["step"] = 5
            st.rerun()

# ============================================================
# STEP 5：结果与下载
# ============================================================
elif step == 5:
    analysis = st.session_state.get("analysis")
    if analysis is None:
        st.error("未找到分析结果，请返回重新运行。")
        if st.button("← 返回"):
            st.session_state["step"] = 4
            st.rerun()
        st.stop()

    groups    = analysis["groups"]
    group_dfs = analysis["group_dfs"]

    # 统计摘要
    with st.expander("📋 统计摘要", expanded=True):
        sum_data = []
        for g in groups:
            gdf = group_dfs[g]
            sum_data.append({
                "组别": str(g),
                "样本量 n": len(gdf),
                "事件数": int(gdf["__event__"].sum()),
            })
        st.table(pd.DataFrame(sum_data))

        st.markdown("**Log-rank 两两比较：**")
        lr_data = []
        for (g1, g2), pv in analysis["pairwise_p"].items():
            lr_data.append({"比较": f"{g1} vs {g2}", "P 值": fmt_p(pv)})
        st.table(pd.DataFrame(lr_data))

        if analysis.get("overall_p") is not None:
            st.markdown(f"**整体 Log-rank：** {fmt_p(analysis['overall_p'])}")

        st.markdown("**Cox HR：**")
        for ht in analysis["hr_texts"]:
            st.markdown(f"- {ht}")

        sel_cov = st.session_state.get("selected_covariates", [])
        if sel_cov:
            st.markdown(f"**Cox 调整协变量：** {', '.join(sel_cov)}")
        else:
            st.markdown("**Cox：** 单变量（无协变量调整）")

    st.markdown("---")
    st.subheader("🎨 图形实时调整与下载")
    st.caption("调整下方任意参数后，预览图将自动更新。")

    col_back, _ = st.columns([1, 5])
    with col_back:
        if st.button("← 返回设置", key="back_step5"):
            st.session_state["step"] = 4
            st.rerun()

    # ── 左右分栏：左侧控件，右侧实时预览 ──
    col_ctrl, col_prev = st.columns([1, 2], gap="large")

    with col_ctrl:
        st.markdown("#### 🔧 位置与显示")
        # 使用 key= 直接绑定 session_state，每次拖动自动触发 rerun
        st.slider("中位时间框 X", 0.0, 0.85, value=st.session_state["text_x"], step=0.01, key="text_x")
        st.slider("中位时间框 Y", 0.10, 0.98, value=st.session_state["text_y"], step=0.01, key="text_y")
        st.slider("图例位置 X",   0.0,  1.0,  value=st.session_state["leg_x"],  step=0.01, key="leg_x")
        st.slider("图例位置 Y",   0.0,  1.0,  value=st.session_state["leg_y"],  step=0.01, key="leg_y")
        st.checkbox("显示 95% 置信区间色带", key="show_ci")

        if len(groups) >= 3:
            st.markdown("**整体 Log-rank 框位置**")
            c3, c4 = st.columns(2)
            with c3:
                st.slider("X", 0.01, 0.99, value=st.session_state["lr_x"], step=0.01, key="lr_x", label_visibility="collapsed")
            with c4:
                st.slider("Y", 0.01, 0.97, value=st.session_state["lr_y"], step=0.01, key="lr_y", label_visibility="collapsed")
            st.caption("↑ 整体检验框 X / Y")

        st.markdown("#### 📝 文字内容")
        st.text_area(
            "中位时间文字（第1行加粗标题，其余行各组数据）",
            height=110,
            key="median_text",
        )
        st.text_area(
            "HR / P 值文字（每行一条，清空则不显示）",
            height=110,
            key="hr_text",
        )
        if len(groups) >= 3:
            st.text_area(
                "Log-rank 整体检验文字（可编辑，清空则不显示）",
                height=55,
                key="lr_text",
            )

    # ── 实时渲染（每次 rerun 都重绘，无需按钮）──
    with col_prev:
        st.markdown("#### 👁 实时预览")
        _setup_font(lang=st.session_state["lang"])

        median_lines = [l for l in st.session_state["median_text"].split("\n")]
        hr_lines     = [l for l in st.session_state["hr_text"].split("\n") if l.strip()]
        lr_text_val  = st.session_state.get("lr_text", "") if len(groups) >= 3 else ""

        state_snap = {
            "x_max":      st.session_state["x_max"],
            "x_ticks":    st.session_state["x_ticks"],
            "x_data_max": st.session_state["x_data_max"],
            "x_label":    st.session_state["x_label"],
            "y_label":    st.session_state["y_label"],
            "lang":       st.session_state["lang"],
            "show_ci":    st.session_state["show_ci"],
        }

        with st.spinner("渲染中…"):
            result = build_figure(
                analysis=analysis,
                state=state_snap,
                text_x=st.session_state["text_x"],
                text_y=st.session_state["text_y"],
                median_text_override=median_lines if median_lines else None,
                hr_text_override=hr_lines,
                logrank_text_override=lr_text_val if len(groups) >= 3 else None,
                lr_x=st.session_state["lr_x"],
                lr_y=st.session_state["lr_y"],
                show_ci=st.session_state["show_ci"],
                legend_x=st.session_state["leg_x"],
                legend_y=st.session_state["leg_y"],
            )

        st.image(result["png"], use_container_width=True)

        st.markdown("#### ⬇️ 下载")
        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                label="⬇️ 下载 PNG",
                data=result["png"],
                file_name="survival_curve.png",
                mime="image/png",
                type="primary",
            )
        with dl_col2:
            st.download_button(
                label="⬇️ 下载 PDF",
                data=result["pdf"],
                file_name="survival_curve.pdf",
                mime="application/pdf",
            )

# ============================================================
# 页脚
# ============================================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#aaa;font-size:12px;'>"
    "生存曲线分析工具 · 基于 lifelines · matplotlib · Streamlit</p>",
    unsafe_allow_html=True,
)
