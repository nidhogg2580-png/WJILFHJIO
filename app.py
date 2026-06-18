# ============================================================
# 折线图交互绘图工具 · Streamlit 版
# 功能：原始数据统计分析（LMM / 描述性统计）+ 直接绘图数据
# 风格与生存曲线分析工具保持一致
# ============================================================

import io
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import streamlit as st

warnings.filterwarnings("ignore")

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="折线图绘图工具",
    page_icon="📈",
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
        📈 折线图绘图工具
      </h2>
      <p style="color:#cce4f7;margin:6px 0 0;font-size:13px;">
        支持原始数据统计分析 · 线性混合模型 · 描述性统计 · 直接绘图数据
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
# 固定颜色序列（与生存曲线工具一致）
# ============================================================
PRESET_COLORS = [
    "#00558C", "#D4820A", "#2E8B57", "#C84B31",
    "#6A0DAD", "#007272", "#1B6CA8", "#8B008B",
    "#556B2F", "#B8860B",
]

# ============================================================
# 字体设置（与生存曲线工具保持一致）
# ============================================================
def _setup_font(lang="en"):
    available = {f.name for f in fm.fontManager.ttflist}
    _BASE_FONTSIZE = 14

    if lang == "zh":
        zh_candidates = [
            "SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei",
            "Noto Sans CJK SC", "AR PL UMing CN", "Source Han Sans SC",
            "PingFang SC", "STSong", "STHeiti",
        ]
        for candidate in zh_candidates:
            if candidate in available:
                plt.rcParams.update({
                    "font.family":        "sans-serif",
                    "font.sans-serif":    [candidate, "DejaVu Sans"],
                    "axes.unicode_minus": False,
                    "font.size":          _BASE_FONTSIZE + 1,
                })
                return candidate
        try:
            import urllib.request, os
            font_url  = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf"
            font_path = "/tmp/NotoSansCJKsc-Regular.otf"
            if not os.path.exists(font_path):
                urllib.request.urlretrieve(font_url, font_path)
            fm.fontManager.addfont(font_path)
            plt.rcParams.update({
                "font.family":        "sans-serif",
                "font.sans-serif":    ["Noto Sans CJK SC", "DejaVu Sans"],
                "axes.unicode_minus": False,
                "font.size":          _BASE_FONTSIZE + 1,
            })
            return "Noto Sans CJK SC"
        except Exception:
            plt.rcParams.update({"axes.unicode_minus": False, "font.size": _BASE_FONTSIZE + 1})
            return "default"
    else:
        _fs = {
            "font.size":       _BASE_FONTSIZE - 1,
            "axes.titlesize":  _BASE_FONTSIZE,
            "axes.labelsize":  _BASE_FONTSIZE,
            "xtick.labelsize": _BASE_FONTSIZE - 1,
            "ytick.labelsize": _BASE_FONTSIZE - 1,
            "legend.fontsize": _BASE_FONTSIZE - 1,
        }
        for candidate in ["Arial", "Liberation Sans", "FreeSans", "DejaVu Sans"]:
            if candidate in available:
                plt.rcParams.update({
                    "font.family":        "sans-serif",
                    "font.sans-serif":    [candidate],
                    "axes.unicode_minus": False,
                    **_fs,
                })
                return candidate
        plt.rcParams.update({"axes.unicode_minus": False, **_fs})
        return "default"

# ============================================================
# 工具函数
# ============================================================
def try_numeric(series):
    """尝试转数值，成功返回 numeric Series，失败返回 None"""
    try:
        return pd.to_numeric(series, errors="raise")
    except Exception:
        return None


def normality_test(arr):
    from scipy.stats import shapiro
    arr = arr.dropna()
    if len(arr) < 3:
        return False
    _, p = shapiro(arr)
    return p >= 0.05


def run_descriptive(df, group_col, time_col, outcome_col):
    """描述性统计：正态→均值±SD；非正态→中位数（无误差棒）"""
    groups = df[group_col].unique()
    times  = df[time_col].unique()

    # 先全局判断正态性
    all_normal = True
    for g in groups:
        for t in times:
            sub = df[(df[group_col] == g) & (df[time_col] == t)][outcome_col]
            if not normality_test(sub):
                all_normal = False

    rows = []
    for g in groups:
        for t in times:
            sub = df[(df[group_col] == g) & (df[time_col] == t)][outcome_col].dropna()
            if len(sub) == 0:
                continue
            if all_normal:
                val  = sub.mean()
                lo   = val - sub.std(ddof=1)
                hi   = val + sub.std(ddof=1)
            else:
                val  = sub.median()
                lo   = np.nan
                hi   = np.nan
            rows.append({"group": g, "time": t, "value": val, "lower": lo, "upper": hi})

    return pd.DataFrame(rows), all_normal


def run_lmm(df, group_col, time_col, outcome_col, subject_col, covariates, cov_types):
    """
    线性混合模型：随机效应=受试者，固定效应=组别*时间+协变量
    自动尝试多个优化器；返回真正的 LS Mean（最小二乘均值）± SE
    """
    import re as _re
    from statsmodels.formula.api import mixedlm

    # ── 列名安全化 ──────────────────────────────────────────
    safe = lambda c: _re.sub(r'\W+', '_', str(c)).strip('_')
    col_map = {col: safe(col) for col in df.columns}
    df2 = df.rename(columns=col_map).copy()

    g_col = col_map[group_col]
    t_col = col_map[time_col]
    y_col = col_map[outcome_col]
    s_col = col_map[subject_col]

    # ── 类型转换 ─────────────────────────────────────────────
    df2[g_col] = df2[g_col].astype("category")
    df2[t_col] = df2[t_col].astype("category")
    df2[y_col] = pd.to_numeric(df2[y_col], errors="coerce")
    df2 = df2.dropna(subset=[y_col])

    # ── 协变量处理 ───────────────────────────────────────────
    cov_terms = []
    safe_covs = []
    for orig_c, ctype in zip(covariates, cov_types):
        sc = col_map[orig_c]
        safe_covs.append(sc)
        if ctype == "定性（分类）":
            df2[sc] = df2[sc].astype("category")
            cov_terms.append(f"C({sc})")
        else:
            df2[sc] = pd.to_numeric(df2[sc], errors="coerce")
            cov_terms.append(sc)

    cov_str = (" + " + " + ".join(cov_terms)) if cov_terms else ""
    formula = f"{y_col} ~ C({g_col}) * C({t_col}){cov_str}"

    # ── 拟合：依次尝试多个优化器 ────────────────────────────
    METHODS = ["lbfgs", "bfgs", "powell", "cg", "nm"]
    result = None
    last_err = None
    for method in METHODS:
        try:
            model  = mixedlm(formula, df2, groups=df2[s_col])
            result = model.fit(reml=True, method=method,
                               warn_convergence=False, disp=False)
            break   # 成功即跳出
        except Exception as e:
            last_err = e
            continue

    if result is None:
        return None, f"所有优化器均拟合失败，最后错误：{last_err}"

    # ── 计算真正的 LS Means（最小二乘均值）± SE ─────────────
    groups_vals = sorted(df[group_col].unique())
    times_vals  = sorted(df[time_col].unique(), key=lambda x: (str(x)))

    # 协变量取均值（连续）或众数（分类）作为边际化基准
    cov_anchor = {}
    for orig_c, ctype in zip(covariates, cov_types):
        sc = col_map[orig_c]
        if ctype == "定量（连续）":
            cov_anchor[sc] = float(df2[sc].mean())
        else:
            cov_anchor[sc] = df2[sc].mode()[0]

    rows = []
    params = result.params
    cov_matrix = result.cov_params()

    # 参考水平（pandas category 的第一个）
    ref_g = df2[g_col].cat.categories[0]
    ref_t = df2[t_col].cat.categories[0]

    for g in groups_vals:
        for t in times_vals:
            sub = df[(df[group_col] == g) & (df[time_col] == t)][outcome_col].dropna()
            if len(sub) == 0:
                continue

            # 构建对比向量 c（与 params 等长）
            c_vec = np.zeros(len(params))
            idx = list(params.index)

            # 截距
            if "Intercept" in idx:
                c_vec[idx.index("Intercept")] = 1.0

            # 组效应
            g_term = f"C({g_col})[T.{g}]"
            if g_term in idx:
                c_vec[idx.index(g_term)] = 1.0

            # 时间效应
            t_term = f"C({t_col})[T.{t}]"
            if t_term in idx:
                c_vec[idx.index(t_term)] = 1.0

            # 交互效应
            inter = f"C({g_col})[T.{g}]:C({t_col})[T.{t}]"
            if inter in idx:
                c_vec[idx.index(inter)] = 1.0

            # 协变量贡献
            for orig_c, ctype in zip(covariates, cov_types):
                sc = col_map[orig_c]
                if ctype == "定量（连续）":
                    k = sc
                    if k in idx:
                        c_vec[idx.index(k)] = cov_anchor[sc]
                else:
                    k = f"C({sc})[T.{cov_anchor[sc]}]"
                    if k in idx:
                        c_vec[idx.index(k)] = 1.0

            ls_mean = float(c_vec @ params.values)
            ls_se   = float(np.sqrt(c_vec @ cov_matrix.values @ c_vec))
            rows.append({
                "group": g, "time": t,
                "value": ls_mean,
                "lower": ls_mean - ls_se,
                "upper": ls_mean + ls_se,
            })

    return pd.DataFrame(rows), None   # (plot_df, error_msg)


def sort_time_categories(cats):
    """
    对定性时间点列表进行智能升序排列。
    规则：Baseline/基线排第一，其余提取数字后按数值升序。
    例：["Week 48","Week 4","Baseline","Week 12"] →
        ["Baseline","Week 4","Week 12","Week 48"]
    """
    import re as _re

    BASELINE_KEYWORDS = {
        "baseline", "基线", "base", "screening",
        "筛选", "run-in", "run_in", "week0", "week 0",
        "month0", "month 0", "day0", "day 0", "visit0", "visit 0",
    }

    def _sort_key(s):
        s_lower = str(s).lower().strip()
        # 基线类关键词排最前
        if s_lower in BASELINE_KEYWORDS or s_lower.replace(" ", "") in BASELINE_KEYWORDS:
            return (0, 0, s)
        # 提取第一个数字，按数值排序
        nums = _re.findall(r'\d+(?:\.\d+)?', s_lower)
        if nums:
            return (1, float(nums[0]), s)
        # 无数字：字母序兜底
        return (2, 0, s)

    return sorted(cats, key=_sort_key)


# ============================================================
# 绘图核心（风格与生存曲线工具一致）
# ============================================================
def build_figure(
    plot_df, groups_order, x_numeric, x_cats,
    lang,
    xlabel, ylabel, panel_label,
    ylim_start_zero,
    vline_on, vline_x,
    legend_x, legend_y,
    fig_width,
    show_errbar=True,       # Fix-1: 控制是否显示误差棒
):
    """
    关键设计原则：
    1. 图形物理尺寸大（12×7 in）+ 低 DPI(100) = 与生存曲线工具等比，
       使字号/线宽在屏幕和 PDF 中视觉一致。
    2. 图例始终用 ax.transAxes 定位，且保存时用固定 subplots_adjust
       而非 tight_layout + bbox_inches="tight"，
       这样移动图例不会改变 axes 的物理比例。
    """
    _zh = (lang == "zh")

    # ── 与生存曲线工具对齐的字号体系 ─────────────────────────
    FS_TICK   = 11 if not _zh else 12
    FS_LABEL  = 12 if not _zh else 13
    FS_LEGEND = 11 if not _zh else 12
    # Panel 标签需明显大于轴标签，作为图形左上角的"小标题"
    # 此前 16/17 在 12×7in 大画布上视觉仍偏小，这里直接采用 20/21
    FS_PANEL  = 20 if not _zh else 21

    FIG_H = 7.0
    DPI   = 100

    plt.rcParams.update({
        "font.size":          FS_TICK,
        "axes.titlesize":     FS_LABEL,
        "axes.labelsize":     FS_LABEL,
        "xtick.labelsize":    FS_TICK,
        "ytick.labelsize":    FS_TICK,
        "legend.fontsize":    FS_LEGEND,
        "pdf.fonttype":       42,
        "axes.unicode_minus": False,
        "lines.linewidth":    1.8,
    })

    fig = plt.figure(figsize=(fig_width, FIG_H), dpi=DPI)

    L, B = 0.10, 0.12
    R, T = 0.92, 0.88
    ax = fig.add_axes([L, B, R - L, T - B])

    # ── X 轴映射（Fix-3: 定性变量按智能升序排列）────────────
    if x_numeric:
        x_vals_sorted = sorted(plot_df["time"].unique())
        x_map      = {v: float(v) for v in x_vals_sorted}
        x_plot_min = float(min(x_vals_sorted))
        x_plot_max = float(max(x_vals_sorted))
    else:
        # 对 x_cats 进行智能时间升序排列
        x_cats_sorted = sort_time_categories(x_cats)
        x_map      = {v: i for i, v in enumerate(x_cats_sorted)}
        x_plot_min = 0
        x_plot_max = len(x_cats_sorted) - 1

    # ── 绘制每组曲线 ──────────────────────────────────────────
    legend_handles = []
    for idx, grp in enumerate(groups_order):
        color = PRESET_COLORS[idx % len(PRESET_COLORS)]
        sub   = plot_df[plot_df["group"] == grp].copy()
        sub["_x"] = sub["time"].map(x_map)
        sub.sort_values("_x", inplace=True)

        xs = sub["_x"].values
        ys = sub["value"].values

        ax.plot(xs, ys, color=color, lw=1.8,
                marker="o", ms=5, zorder=3)

        has_eb = (show_errbar and
                  sub["lower"].notna().any() and sub["upper"].notna().any())
        if has_eb:
            yerr_lo = np.clip(ys - sub["lower"].values, 0, None)
            yerr_hi = np.clip(sub["upper"].values - ys, 0, None)
            ax.errorbar(xs, ys, yerr=[yerr_lo, yerr_hi],
                        fmt="none", ecolor=color,
                        elinewidth=1.2, capsize=3, zorder=2)

        legend_handles.append(
            Line2D([0], [0], color=color, lw=1.8, label=str(grp))
        )

    # ── 垂直虚线 ──────────────────────────────────────────────
    if vline_on:
        vx = vline_x if x_numeric else x_map.get(vline_x, None)
        if vx is not None:
            ax.axvline(float(vx), color="#888888",
                       linestyle=(0, (4, 4)), linewidth=1.0, zorder=1)

    # ── X 轴范围与刻度 ────────────────────────────────────────
    if x_numeric:
        x_range = x_plot_max - x_plot_min if x_plot_max != x_plot_min else 1.0
        ax.set_xlim(x_plot_min, x_plot_max + x_range * 0.02)
        ax.set_xticks(x_vals_sorted)
        ax.set_xticklabels([str(v) for v in x_vals_sorted], fontsize=FS_TICK)
    else:
        ax.set_xlim(-0.5, x_plot_max + 0.5)
        ax.set_xticks(range(len(x_cats_sorted)))
        ax.set_xticklabels(x_cats_sorted, fontsize=FS_TICK)

    # ── Y 轴范围与刻度（稀疏，nbins=4）──────────────────────
    y_vals    = plot_df["value"].dropna()
    y_lo_data = plot_df["lower"].dropna().min() if plot_df["lower"].notna().any() else y_vals.min()
    y_hi_data = plot_df["upper"].dropna().max() if plot_df["upper"].notna().any() else y_vals.max()
    y_range   = max(y_hi_data - y_lo_data, 1e-6)
    y_margin  = y_range * 0.15

    y_bot = 0 if ylim_start_zero else y_lo_data - y_margin
    y_top = y_hi_data + y_margin
    ax.set_ylim(y_bot, y_top)

    # Y 轴刻度：保证固定数量（约5~6个），同时尽量落在整齐数值上
    # 策略：在 [y_bot, y_top] 范围内，尝试若干"美观步长"，
    # 选择能产生 4~7 个刻度中最接近 6 的步长；若都不满足才退化为等分。
    y_span = y_top - y_bot
    candidate_steps = []
    for base in [1, 2, 2.5, 5]:
        for mag in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
            candidate_steps.append(base * mag)

    best_step = None
    best_count = None
    for step in sorted(set(candidate_steps)):
        n = y_span / step
        if 4 <= n <= 7:
            if best_step is None or abs(n - 6) < abs(best_count - 6):
                best_step = step
                best_count = n

    if best_step is not None:
        # 用美观步长生成刻度，起点对齐到 step 的整数倍
        start_tick = np.ceil(y_bot / best_step) * best_step
        y_ticks = list(np.arange(start_tick, y_top + best_step * 0.5, best_step))
        # 确保至少有4个刻度，否则补充等分
        if len(y_ticks) < 4:
            y_ticks = list(np.linspace(y_bot, y_top, 6))
    else:
        # 退化方案：固定6个等分刻度
        y_ticks = list(np.linspace(y_bot, y_top, 6))

    # 根据 step 决定小数位数
    if best_step is not None and best_step >= 1:
        decimals = 0
    elif best_step is not None and best_step >= 0.1:
        decimals = 1
    else:
        decimals = 2

    y_ticks = sorted(set(round(v, decimals) for v in y_ticks))
    # 过滤超出范围的刻度（保留视觉边界内的）
    y_ticks = [v for v in y_ticks if y_bot - 1e-9 <= v <= y_top + 1e-9]

    ax.set_yticks(y_ticks)
    if decimals == 0:
        ax.set_yticklabels([f"{int(round(v))}" for v in y_ticks], fontsize=FS_TICK)
    else:
        ax.set_yticklabels([f"{v:.{decimals}f}" for v in y_ticks], fontsize=FS_TICK)
    ax.tick_params(axis="y", labelsize=FS_TICK)

    # ── 坐标轴样式 ────────────────────────────────────────────
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(axis="both", direction="out", length=5, width=1.2, pad=5)

    # ── 轴标签 ────────────────────────────────────────────────
    ax.set_xlabel(xlabel, fontsize=FS_LABEL, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL, labelpad=10)

    # ── Panel 标签（左上角，固定在 axes 上方）────────────────
    if panel_label.strip():
        ax.text(0.0, 1.04, panel_label,
                transform=ax.transAxes,
                fontsize=FS_PANEL,
                fontweight="bold",
                ha="left", va="bottom")

    # ── 图例 ─────────────────────────────────────────────────
    # 关键：bbox_transform=ax.transAxes，始终在 axes 坐标系内定位
    # 保存时不用 bbox_inches="tight"，所以图例超出 axes 也不会压缩图形
    ax.legend(
        handles=legend_handles,
        frameon=False,
        bbox_to_anchor=(legend_x, legend_y),
        loc="upper right",
        bbox_transform=ax.transAxes,
        fontsize=FS_LEGEND,
        handlelength=2.0,
    )

    # ── 保存：bbox_inches=None 保持 figure 固定尺寸 ──────────
    buf_png = io.BytesIO()
    fig.savefig(buf_png, format="png", dpi=DPI, bbox_inches=None)
    buf_png.seek(0)
    png_bytes = buf_png.read()

    buf_pdf = io.BytesIO()
    fig.savefig(buf_pdf, format="pdf", bbox_inches=None)
    buf_pdf.seek(0)
    pdf_bytes = buf_pdf.read()

    plt.close(fig)
    return {"png": png_bytes, "pdf": pdf_bytes}


# ============================================================
# Session state 初始化
# ============================================================
def init_state():
    defaults = {
        "step":             0,
        "lang":             "en",
        "data_mode":        None,     # "raw" | "direct"
        "raw_df":           None,
        "plot_df":          None,
        "groups_order":     [],
        "x_numeric":        True,
        "x_cats":           [],       # 定性时间点顺序
        "col_subject":      None,
        "col_group":        None,
        "col_time":         None,
        "col_outcome":      None,
        "covariate_cols":   [],
        "stat_method":      None,
        "cov_types_map":    {},
        "selected_covs":    [],
        "stat_note":        "",
        # 分析设置
        "xlabel":           "Timepoint",
        "ylabel":           "Change from baseline",
        "panel_label":      "A  Figure",
        "ylim_start_zero":  False,
        "vline_on":         False,
        "vline_x":          0,
        # 预览调节
        "legend_x":         0.98,
        "legend_y":         0.98,
        "fig_width":        12.0,
        "show_errbar":      True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ============================================================
# 页面标题横幅
# ============================================================
st.markdown("""
<div style="background:linear-gradient(135deg,#00558C,#003a63);
            border-radius:10px;padding:18px 24px;margin-bottom:16px;">
  <h2 style="color:white;margin:0;font-family:Arial,sans-serif;">
    📈 折线图绘图工具
  </h2>
  <p style="color:#cce4f7;margin:6px 0 0;font-size:13px;">
    支持原始数据统计分析（线性混合模型 / 描述性统计）· 直接绘图数据
  </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 进度条
# ============================================================
step = st.session_state["step"]

# 将 step 统一映射为数值，用于进度条着色
# "run_stat" 和 3.5 均视为第4步（协变量/计算中）
def _step_to_num(s):
    if s in ("run_stat", "cov_setup"):
        return 4
    try:
        return int(float(s))
    except Exception:
        return 0

_step_num = _step_to_num(step)

STEP_LABELS_RAW    = ["语言", "数据类型", "上传数据", "统计方法", "协变量/计算", "分析设置", "结果与下载"]
STEP_LABELS_DIRECT = ["语言", "数据类型", "上传数据", "分析设置", "结果与下载"]

# 根据数据模式决定标签
if st.session_state["data_mode"] == "direct":
    _labels = STEP_LABELS_DIRECT
else:
    _labels = STEP_LABELS_RAW

cols_prog = st.columns(len(_labels))
for i, (col, label) in enumerate(zip(cols_prog, _labels)):
    with col:
        if i < _step_num:
            st.markdown(f"<div style='text-align:center;color:#2E8B57;font-size:12px;'>✅ {label}</div>", unsafe_allow_html=True)
        elif i == _step_num:
            st.markdown(f"<div style='text-align:center;color:#00558C;font-weight:bold;font-size:12px;'>▶ {label}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='text-align:center;color:#aaa;font-size:12px;'>○ {label}</div>", unsafe_allow_html=True)

st.markdown("---")

# ============================================================
# STEP 0：语言选择
# ============================================================
if step == 0:
    st.subheader("第 0 步：选择绘图语言")
    st.markdown("请选择图表中文字的显示语言。中文模式自动使用中文字体，字号在英文基础上增大 1 号；英文模式使用 Arial / DejaVu Sans。")

    lang_choice = st.radio(
        "绘图语言",
        options=["English", "中文"],
        index=0 if st.session_state["lang"] == "en" else 1,
        horizontal=True,
    )

    if st.button("确认并继续 →", type="primary", key="btn_step0"):
        lang = "zh" if lang_choice == "中文" else "en"
        st.session_state["lang"] = lang
        _setup_font(lang=lang)
        if lang == "zh":
            st.session_state["xlabel"] = "时间点"
            st.session_state["ylabel"] = "较基线的变化量"
        else:
            st.session_state["xlabel"] = "Timepoint"
            st.session_state["ylabel"] = "Change from baseline"
        st.session_state["step"] = 1
        st.rerun()

# ============================================================
# STEP 1：数据类型选择
# ============================================================
elif step == 1:
    st.subheader("第 1 步：选择数据类型")
    st.markdown("""
请选择您的数据来源：

- **原始数据**：包含每位受试者的个体测量值，本工具将帮助您完成统计分析（线性混合模型或描述性统计），计算点估计值与误差棒。
- **直接绘图数据**：您已有整理好的均值/中位数及误差棒数据（如从文献、GBD 数据库下载），可直接用于绘图。
    """)

    data_choice = st.radio(
        "数据类型",
        ["原始数据（需统计计算）", "直接绘图数据（已有点估计 + 误差棒）"],
        horizontal=False,
    )

    col_back, col_next = st.columns([1, 5])
    with col_back:
        if st.button("← 返回", key="back_s1"):
            st.session_state["step"] = 0
            st.rerun()
    with col_next:
        if st.button("确认并继续 →", type="primary", key="btn_step1"):
            if "原始" in data_choice:
                st.session_state["data_mode"] = "raw"
            else:
                st.session_state["data_mode"] = "direct"
            st.session_state["step"] = 2
            st.rerun()

# ============================================================
# STEP 2：上传数据文件
# ============================================================
elif step == 2:
    mode = st.session_state["data_mode"]

    if mode == "raw":
        st.subheader("第 2 步：上传原始数据")
        st.markdown("""
**文件格式（Excel / CSV）：**

| 列序 | 含义 | 说明 |
|------|------|------|
| 第 1 列 | 受试者编号 | 每位受试者的唯一 ID |
| 第 2 列 | 组别 | 如治疗组、对照组 |
| 第 3 列 | 时间点 | 如 0, 4, 12, 24（数值）或 Baseline, Week1（文字） |
| 第 4 列 | 因变量 | 每次测量的数值（如 mRSS, eGFR） |
| 第 5 列起 | 协变量（可选） | 任意数量，可不提供 |
        """)

        uploaded = st.file_uploader("选择文件（.xlsx 或 .csv）", type=["xlsx", "xls", "csv"], key="raw_up")

        if uploaded:
            try:
                if uploaded.name.lower().endswith("csv"):
                    df = pd.read_csv(uploaded)
                else:
                    df = pd.read_excel(io.BytesIO(uploaded.read()))

                cols = list(df.columns)
                if len(cols) < 4:
                    st.error("❌ 文件至少需要 4 列（受试者编号、组别、时间点、因变量）")
                else:
                    st.success(f"✅ 读取成功：{uploaded.name}，共 {len(df)} 行，{len(cols)} 列")
                    st.dataframe(df.head(6), use_container_width=True)

                    st.session_state["raw_df"]       = df
                    st.session_state["col_subject"]  = cols[0]
                    st.session_state["col_group"]    = cols[1]
                    st.session_state["col_time"]     = cols[2]
                    st.session_state["col_outcome"]  = cols[3]
                    st.session_state["covariate_cols"] = cols[4:]

                    # 自动检测时间点类型
                    num_t = try_numeric(df[cols[2]])
                    st.session_state["x_numeric"] = (num_t is not None)
                    if num_t is None:
                        cats = list(dict.fromkeys(df[cols[2]].astype(str).tolist()))
                        st.session_state["x_cats"] = cats
                        st.info(f"ℹ️ 时间点为**定性变量**，将按等间距排列：{' → '.join(cats)}")
                    else:
                        st.info(f"ℹ️ 时间点为**定量变量**，将按数值比例排列坐标轴")

                    col_back, col_next = st.columns([1, 5])
                    with col_back:
                        if st.button("← 返回", key="back_s2r"):
                            st.session_state["step"] = 1
                            st.rerun()
                    with col_next:
                        if st.button("确认并继续 →", type="primary", key="btn_s2r"):
                            st.session_state["step"] = 3
                            st.rerun()
            except Exception as e:
                st.error(f"❌ 读取失败：{e}")
        else:
            if st.button("← 返回", key="back_s2r_empty"):
                st.session_state["step"] = 1
                st.rerun()

    else:  # direct
        st.subheader("第 2 步：上传直接绘图数据")
        st.markdown("""
**文件格式（Excel / CSV）— 共 5 列：**

| 列序 | 含义 | 说明 |
|------|------|------|
| 第 1 列 | 组别 | 如 Rituximab, Placebo |
| 第 2 列 | 时间点 | 如 0, 4, 12, 24（数值）或文字 |
| 第 3 列 | 点估计值 | 均值或中位数 |
| 第 4 列 | 误差棒下限 | 均值 - SD / 95%CI 下界（无则留空） |
| 第 5 列 | 误差棒上限 | 均值 + SD / 95%CI 上界（无则留空） |
        """)

        uploaded = st.file_uploader("选择文件（.xlsx 或 .csv）", type=["xlsx", "xls", "csv"], key="direct_up")

        if uploaded:
            try:
                if uploaded.name.lower().endswith("csv"):
                    df = pd.read_csv(uploaded)
                else:
                    df = pd.read_excel(io.BytesIO(uploaded.read()))

                if len(df.columns) < 3:
                    st.error("❌ 至少需要 3 列（组别、时间点、点估计值）")
                else:
                    rename_map = {
                        df.columns[0]: "group",
                        df.columns[1]: "time",
                        df.columns[2]: "value",
                    }
                    if len(df.columns) >= 4:
                        rename_map[df.columns[3]] = "lower"
                    if len(df.columns) >= 5:
                        rename_map[df.columns[4]] = "upper"
                    df.rename(columns=rename_map, inplace=True)
                    if "lower" not in df.columns:
                        df["lower"] = np.nan
                    if "upper" not in df.columns:
                        df["upper"] = np.nan

                    df["value"] = pd.to_numeric(df["value"], errors="coerce")
                    df["lower"] = pd.to_numeric(df["lower"], errors="coerce")
                    df["upper"] = pd.to_numeric(df["upper"], errors="coerce")

                    num_t = try_numeric(df["time"])
                    if num_t is not None:
                        df["time"] = num_t
                        st.session_state["x_numeric"] = True
                        st.info("ℹ️ 时间点为**定量变量**，将按数值比例排列坐标轴")
                    else:
                        st.session_state["x_numeric"] = False
                        cats = list(dict.fromkeys(df["time"].astype(str).tolist()))
                        st.session_state["x_cats"] = cats
                        df["time"] = df["time"].astype(str)
                        st.info(f"ℹ️ 时间点为**定性变量**，将按等间距排列：{' → '.join(cats)}")

                    groups = list(dict.fromkeys(df["group"].astype(str).tolist()))
                    df["group"] = df["group"].astype(str)
                    st.session_state["plot_df"]      = df
                    st.session_state["groups_order"] = groups

                    st.success(f"✅ 数据加载成功！{len(groups)} 个组：{', '.join(groups)}")
                    st.dataframe(df.head(10), use_container_width=True)

                    col_back, col_next = st.columns([1, 5])
                    with col_back:
                        if st.button("← 返回", key="back_s2d"):
                            st.session_state["step"] = 1
                            st.rerun()
                    with col_next:
                        if st.button("确认并继续 →", type="primary", key="btn_s2d"):
                            st.session_state["step"] = 4   # 直接跳分析设置
                            st.rerun()
            except Exception as e:
                st.error(f"❌ 读取失败：{e}")
        else:
            if st.button("← 返回", key="back_s2d_empty"):
                st.session_state["step"] = 1
                st.rerun()

# ============================================================
# STEP 3（仅原始数据）：选择统计方法
# ============================================================
elif step == 3:
    st.subheader("第 3 步：选择统计分析方法")
    st.markdown("""
请选择用于计算**点估计值和误差棒**的统计方法：

- **线性混合模型（推荐）**：考虑受试者间的个体差异（随机效应），同时控制组别、时间及其交互作用（固定效应）。适用于重复测量临床试验数据，计算真正的最小二乘均值（LS Mean）± 标准误（SE）。
- **描述性统计**：先检验正态性（Shapiro-Wilk）。若所有组均正态，则输出均值 ± SD；若存在非正态组，则输出中位数（不提供误差棒）。
    """)

    method_choice = st.radio(
        "统计方法",
        ["线性混合模型（推荐）", "描述性统计"],
        horizontal=True,
    )
    st.session_state["stat_method"] = method_choice

    col_back, col_next = st.columns([1, 5])
    with col_back:
        if st.button("← 返回", key="back_s3"):
            st.session_state["step"] = 2
            st.rerun()
    with col_next:
        if st.button("确认并继续 →", type="primary", key="btn_s3"):
            cov_cols = st.session_state["covariate_cols"]
            if "线性混合" in method_choice and cov_cols:
                st.session_state["step"] = "cov_setup"
                st.rerun()
            else:
                # 描述性统计 或 LMM但无协变量 → 直接执行
                st.session_state["selected_covs"] = []
                st.session_state["step"] = "run_stat"
                st.rerun()

# ============================================================
# STEP cov_setup（仅 LMM + 有协变量）：协变量设置
# ============================================================
elif step == "cov_setup":
    st.subheader("第 3b 步：协变量设置")
    st.markdown("请为每个协变量指定类型，并勾选需要纳入线性混合模型进行调整的协变量。")

    cov_cols = st.session_state["covariate_cols"]
    cov_types_map = {}
    for c in cov_cols:
        current = st.session_state["cov_types_map"].get(c, "定量（连续）")
        idx_c = 0 if current == "定量（连续）" else 1
        choice = st.selectbox(
            f"**{c}** 的变量类型",
            ["定量（连续）", "定性（分类）"],
            index=idx_c,
            key=f"ctype_{c}",
        )
        cov_types_map[c] = choice

    st.session_state["cov_types_map"] = cov_types_map

    selected_covs = []
    st.markdown("**选择纳入模型的协变量（可不选）：**")
    for c in cov_cols:
        checked = st.checkbox(f"{c}  [{cov_types_map.get(c,'定量（连续）')}]", key=f"cov_sel_{c}")
        if checked:
            selected_covs.append(c)

    col_back, col_next = st.columns([1, 5])
    with col_back:
        if st.button("← 返回", key="back_cov"):
            st.session_state["step"] = 3
            st.rerun()
    with col_next:
        if st.button("确认并开始计算 →", type="primary", key="btn_cov"):
            st.session_state["selected_covs"] = selected_covs
            st.session_state["step"] = "run_stat"
            st.rerun()

# ============================================================
# STEP run_stat：执行统计计算
# ============================================================
elif step == "run_stat":
    st.subheader("正在执行统计分析……")

    raw_df  = st.session_state["raw_df"]
    s_col   = st.session_state["col_subject"]
    g_col   = st.session_state["col_group"]
    t_col   = st.session_state["col_time"]
    y_col   = st.session_state["col_outcome"]
    method  = st.session_state["stat_method"]
    sel_cov = st.session_state["selected_covs"]
    cov_types_map = st.session_state["cov_types_map"]

    with st.spinner("计算中，请稍候……"):
        if "描述性" in method:
            plot_df, all_normal = run_descriptive(raw_df, g_col, t_col, y_col)
            if all_normal:
                note = "✅ 所有组数据符合正态分布 → 使用均值 ± SD"
            else:
                note = "⚠️ 存在非正态组 → 使用中位数（不提供误差棒）"
            err = None
        else:
            cov_types_list = [cov_types_map.get(c, "定量（连续）") for c in sel_cov]
            plot_df, err = run_lmm(raw_df, g_col, t_col, y_col, s_col, sel_cov, cov_types_list)
            note = "✅ 线性混合模型计算完成（LS Mean ± SE）" if err is None else ""

    if err:
        st.error(f"❌ 统计计算失败：{err}")
        if st.button("← 返回重新设置", key="back_rstat"):
            st.session_state["step"] = 3
            st.rerun()
        st.stop()

    st.success(note)
    st.session_state["stat_note"] = note

    # 处理时间点类型
    num_t = try_numeric(plot_df["time"])
    if num_t is not None:
        plot_df["time"] = num_t
        st.session_state["x_numeric"] = True
    else:
        st.session_state["x_numeric"] = False
        cats = list(dict.fromkeys(plot_df["time"].astype(str).tolist()))
        st.session_state["x_cats"] = cats

    groups = list(dict.fromkeys(
        [str(g) for g in raw_df[g_col].unique()]
    ))
    plot_df["group"] = plot_df["group"].astype(str)

    st.session_state["plot_df"]      = plot_df
    st.session_state["groups_order"] = groups

    st.dataframe(plot_df.round(4), use_container_width=True)

    col_back, col_next = st.columns([1, 5])
    with col_back:
        if st.button("← 返回重新设置", key="back_rstat2"):
            st.session_state["step"] = 3
            st.rerun()
    with col_next:
        if st.button("继续进行分析设置 →", type="primary", key="btn_rstat"):
            st.session_state["step"] = 4
            st.rerun()

# ============================================================
# STEP 4：分析设置（坐标轴标签、标题等）
# ============================================================
elif step == 4:
    st.subheader("第 4 步：分析设置")
    st.markdown("请设置图形的基本要素，包括坐标轴标签、左上角标注和纵轴起始。")

    lang = st.session_state["lang"]

    col1, col2 = st.columns(2)
    with col1:
        xlabel = st.text_input(
            "横轴标签",
            value=st.session_state["xlabel"],
            key="xlabel_input",
        )
        ylabel = st.text_input(
            "纵轴标签",
            value=st.session_state["ylabel"],
            key="ylabel_input",
        )
    with col2:
        panel_label = st.text_input(
            "左上角 Panel 标签（留空则不显示）",
            value=st.session_state["panel_label"],
            key="panel_input",
        )
        ylim_mode = st.radio(
            "纵轴 Y 起始点",
            ["从 0 开始（适合比例/率类数据）", "从数据最小值开始（适合变化量数据）"],
            index=1,
            horizontal=False,
        )
        ylim_start_zero = "从 0 开始" in ylim_mode

    # 垂直虚线设置
    st.markdown("---")
    st.markdown("**垂直分割线**（可选）")
    vline_on = st.checkbox("在图中添加垂直虚线（如标记某时间节点）", value=st.session_state["vline_on"])

    vline_x = st.session_state["vline_x"]
    if vline_on:
        plot_df   = st.session_state["plot_df"]
        x_numeric = st.session_state["x_numeric"]
        if x_numeric:
            t_min = float(plot_df["time"].min())
            t_max = float(plot_df["time"].max())
            vline_x = st.number_input(
                "虚线位置（X 轴数值）",
                min_value=float(t_min),
                max_value=float(t_max),
                value=float(np.median(plot_df["time"].unique())),
                step=float((t_max - t_min) / 20) or 1.0,
            )
        else:
            x_cats  = st.session_state["x_cats"]
            vline_x = st.selectbox("虚线位置（时间点名称）", x_cats)

    col_back, col_next = st.columns([1, 5])
    with col_back:
        if st.button("← 返回", key="back_s4"):
            if st.session_state["data_mode"] == "direct":
                st.session_state["step"] = 2
            else:
                st.session_state["step"] = "run_stat"
            st.rerun()
    with col_next:
        if st.button("🚀 生成图形", type="primary", key="btn_s4"):
            st.session_state["xlabel"]         = xlabel
            st.session_state["ylabel"]         = ylabel
            st.session_state["panel_label"]    = panel_label
            st.session_state["ylim_start_zero"] = ylim_start_zero
            st.session_state["vline_on"]       = vline_on
            st.session_state["vline_x"]        = vline_x
            st.session_state["step"]           = 5
            st.rerun()

# ============================================================
# STEP 5：结果预览与下载
# ============================================================
elif step == 5:
    plot_df     = st.session_state.get("plot_df")
    groups_order = st.session_state.get("groups_order", [])

    if plot_df is None or len(groups_order) == 0:
        st.error("未找到绘图数据，请返回重新操作。")
        if st.button("← 返回", key="back_s5_err"):
            st.session_state["step"] = 4
            st.rerun()
        st.stop()

    _setup_font(lang=st.session_state["lang"])

    st.markdown("---")
    st.subheader("🎨 图形实时预览与下载")
    st.caption("调整下方参数后，预览图将**自动更新**，无需点击任何按钮。")

    col_back, _ = st.columns([1, 5])
    with col_back:
        if st.button("← 返回设置", key="back_s5"):
            st.session_state["step"] = 4
            st.rerun()

    # ── 左侧控件 | 右侧预览 ────────────────────────────────
    col_ctrl, col_prev = st.columns([1, 2], gap="large")

    with col_ctrl:
        st.markdown("#### 🔧 调整参数")

        st.markdown("**图例位置**")
        st.slider("图例 X 位置", 0.0, 1.2, value=st.session_state["legend_x"],
                  step=0.01, key="legend_x")
        st.slider("图例 Y 位置", 0.0, 1.2, value=st.session_state["legend_y"],
                  step=0.01, key="legend_y")

        st.markdown("**横轴宽度**")
        st.slider("图形宽度 (inches)", 8.0, 20.0,
                  value=st.session_state["fig_width"],
                  step=0.5, key="fig_width")
        st.caption("拖动可避免横轴刻度标签重叠；图例相对位置不变。")

        st.markdown("**误差棒**")
        st.checkbox(
            "显示误差棒",
            value=st.session_state["show_errbar"],
            key="show_errbar",
            help="分组较多时可关闭误差棒，避免相互遮挡影响可视化",
        )

    with col_prev:
        st.markdown("#### 👁 实时预览")

        result = build_figure(
            plot_df        = plot_df,
            groups_order   = groups_order,
            x_numeric      = st.session_state["x_numeric"],
            x_cats         = st.session_state["x_cats"],
            lang           = st.session_state["lang"],
            xlabel         = st.session_state["xlabel"],
            ylabel         = st.session_state["ylabel"],
            panel_label    = st.session_state["panel_label"],
            ylim_start_zero= st.session_state["ylim_start_zero"],
            vline_on       = st.session_state["vline_on"],
            vline_x        = st.session_state["vline_x"],
            legend_x       = st.session_state["legend_x"],
            legend_y       = st.session_state["legend_y"],
            fig_width      = st.session_state["fig_width"],
            show_errbar    = st.session_state["show_errbar"],
        )

        st.image(result["png"], use_container_width=True)

        st.markdown("#### ⬇️ 下载")
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "⬇️ 下载 PNG",
                data=result["png"],
                file_name="figure.png",
                mime="image/png",
                type="primary",
            )
        with dl2:
            st.download_button(
                "⬇️ 下载 PDF",
                data=result["pdf"],
                file_name="figure.pdf",
                mime="application/pdf",
            )

# ============================================================
# 页脚
# ============================================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#aaa;font-size:12px;'>"
    "折线图绘图工具 · 基于 statsmodels · matplotlib · Streamlit</p>",
    unsafe_allow_html=True,
)
