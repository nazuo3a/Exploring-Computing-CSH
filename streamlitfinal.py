import re
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# ===== Matplotlib (헤드리스 + 한글 폰트 설정) =====
import matplotlib
matplotlib.use("Agg")  # Streamlit Cloud/서버 환경에서 안전
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def setup_korean_font():
    """
    1) 레포에 포함한 폰트(권장): ./fonts/NanumGothic.ttf 또는 ./fonts/Pretendard-Regular.ttf 등
    2) 시스템 폰트 fallback: Windows/Mac/Linux 대표 한글 폰트들
    """
    repo_dir = Path(__file__).parent
    candidates = [
        repo_dir / "fonts" / "NanumGothic.ttf",
        repo_dir / "fonts" / "NanumGothic-Regular.ttf",
        repo_dir / "fonts" / "Pretendard-Regular.ttf",
        repo_dir / "fonts" / "NotoSansCJKkr-Regular.otf",
        repo_dir / "fonts" / "NotoSansKR-Regular.otf",
    ]

    chosen_name = None
    chosen_path = None

    for fp in candidates:
        if fp.exists():
            try:
                fm.fontManager.addfont(str(fp))
                chosen_name = fm.FontProperties(fname=str(fp)).get_name()
                chosen_path = str(fp)
                break
            except Exception:
                pass

    if chosen_name is not None:
        mpl.rcParams["font.family"] = chosen_name
        mpl.rcParams["axes.unicode_minus"] = False
        return True, f"Repo font: {chosen_name} ({chosen_path})"

    # 시스템 폰트 fallback (환경에 따라 있을 수도/없을 수도)
    fallback = ["NanumGothic", "Noto Sans CJK KR", "Noto Sans KR", "Malgun Gothic", "AppleGothic", "DejaVu Sans"]
    mpl.rcParams["font.family"] = fallback
    mpl.rcParams["axes.unicode_minus"] = False
    return False, f"System fallback: {fallback}"


FONT_OK, FONT_MSG = setup_korean_font()


# ===== Streamlit 기본 설정 =====
st.set_page_config(page_title="노인 건강 데이터 상관/시각화", layout="wide")
st.title("노인 건강 데이터: 상관관계 · 시각화 · 상관계수(앱)")


# ===== 파일 기본 경로(로컬/채점 환경에서 존재할 때 자동 로드) =====
DEFAULT_CHRONIC = "노인의_성별_만성질병_종류별_유병률_의사진단_기준__및_현_치료율_20251214174945.xlsx"
DEFAULT_SUBJ    = "/노인의_주관적_건강상태_20251214230000.xlsx"


# =========================
# 유틸
# =========================
def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace("%", "", regex=False).str.strip(), errors="coerce")


def safe_corr_and_pvalues(x, y):
    """
    scipy가 있으면 p-value까지, 없으면 r만 반환.
    """
    x = pd.Series(x)
    y = pd.Series(y)
    mask = x.notna() & y.notna()
    x = x[mask].astype(float)
    y = y[mask].astype(float)

    out = {"n": int(len(x))}
    if len(x) < 3:
        out.update({"pearson_r": np.nan, "pearson_p": np.nan, "spearman_rho": np.nan, "spearman_p": np.nan})
        return out

    out["pearson_r"] = float(x.corr(y, method="pearson"))
    out["spearman_rho"] = float(x.corr(y, method="spearman"))

    try:
        from scipy.stats import pearsonr, spearmanr
        pr, pp = pearsonr(x, y)
        sr, sp = spearmanr(x, y)
        out["pearson_r"], out["pearson_p"] = float(pr), float(pp)
        out["spearman_rho"], out["spearman_p"] = float(sr), float(sp)
    except Exception:
        out["pearson_p"], out["spearman_p"] = np.nan, np.nan

    return out


def scatter_with_regline(x, y, title, xlabel, ylabel):
    x = pd.Series(x)
    y = pd.Series(y)
    mask = x.notna() & y.notna()
    x = x[mask].astype(float)
    y = y[mask].astype(float)

    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if len(x) >= 2:
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ys = m * xs + b
        ax.plot(xs, ys)

    fig.tight_layout()
    return fig


def corr_heatmap(df_numeric: pd.DataFrame, title="Correlation (Pearson)"):
    C = df_numeric.corr(method="pearson")
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(C.values, aspect="auto")
    ax.set_xticks(range(C.shape[1]))
    ax.set_yticks(range(C.shape[0]))
    ax.set_xticklabels(C.columns, rotation=45, ha="right")
    ax.set_yticklabels(C.index)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            v = C.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9)

    fig.tight_layout()
    return fig


def age_mid(s):
    nums = list(map(int, re.findall(r"\d+", str(s))))
    if len(nums) >= 2:
        return (nums[0] + nums[1]) / 2
    if len(nums) == 1:
        return float(nums[0])
    return np.nan


# =========================
# 데이터 로더/전처리
# =========================
@st.cache_data(show_spinner=False)
def read_excel(file_or_path, sheet="데이터"):
    return pd.read_excel(file_or_path, sheet_name=sheet)


@st.cache_data(show_spinner=False)
def load_chronic_tidy(file_or_path):
    raw = read_excel(file_or_path, sheet="데이터")

    # 첫 2행을 헤더로 재구성
    header0 = raw.iloc[0].tolist()  # 전체/남자/여자
    header1 = raw.iloc[1].tolist()  # 유병률/치료율

    cols = ["질환"] + [f"{a}_{b}" for a, b in zip(header0[1:], header1[1:])]
    df = raw.iloc[2:].copy()

    # 첫 컬럼명 보정 + 컬럼 재지정
    df = df.rename(columns={df.columns[0]: "질환"})
    df.columns = cols

    for c in df.columns[1:]:
        df[c] = to_numeric(df[c])

    long = df.melt(id_vars=["질환"], var_name="집단_지표", value_name="값")
    long[["집단", "지표"]] = long["집단_지표"].str.split("_", expand=True)
    long = long.drop(columns=["집단_지표"])
    return df, long


@st.cache_data(show_spinner=False)
def load_subjective_tidy(file_or_path):
    raw = read_excel(file_or_path, sheet="데이터")

    # 0행(첫 줄)에 라벨이 들어있는 형태 처리
    measures = raw.iloc[0].tolist()
    cols = list(raw.columns[:2]) + [str(m) for m in measures[2:]]

    df = raw.iloc[1:].copy()
    df.columns = cols
    df = df.rename(columns={df.columns[0]: "특성1", df.columns[1]: "특성2"})

    for c in df.columns[2:]:
        df[c] = to_numeric(df[c])

    df["특성1"] = df["특성1"].ffill()
    return df


# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("데이터 업로드")
    chronic_file = st.file_uploader("만성질환(유병률/치료율) 엑셀", type=["xlsx"], key="chronic")
    subj_file = st.file_uploader("주관적 건강상태 엑셀", type=["xlsx"], key="subj")

    st.divider()
    use_defaults = st.checkbox("기본 경로 파일 사용(가능할 때)", value=True)

    st.divider()
    st.subheader("한글 폰트 상태")
    st.write(FONT_MSG)
    if not FONT_OK:
        st.caption("Streamlit Cloud에서 확실히 하려면 레포에 폰트 파일을 포함하세요: ./fonts/NanumGothic.ttf")


def resolve_source(uploaded, default_path):
    if uploaded is not None:
        return uploaded
    if use_defaults and Path(default_path).exists():
        return default_path
    return None


chronic_src = resolve_source(chronic_file, DEFAULT_CHRONIC)
subj_src = resolve_source(subj_file, DEFAULT_SUBJ)

if chronic_src is None and subj_src is None:
    st.info("왼쪽 사이드바에서 엑셀 파일을 업로드해줘.")
    st.stop()


# =========================
# Tabs
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "데이터 미리보기",
    "만성질환 상관/시각화",
    "주관적 건강 상관/시각화",
    "히트맵(요약)",
    "내보내기"
])

# TAB 1
with tab1:
    c1, c2 = st.columns(2)

    if chronic_src is not None:
        chronic_wide, chronic_long = load_chronic_tidy(chronic_src)
        with c1:
            st.subheader("만성질환(2020) - wide")
            st.dataframe(chronic_wide, use_container_width=True, height=420)
            st.caption("집계표(질환 단위) 상관이므로 '질환 간 관계'를 의미")

    if subj_src is not None:
        subj = load_subjective_tidy(subj_src)
        with c2:
            st.subheader("주관적 건강상태(2023)")
            st.dataframe(subj, use_container_width=True, height=420)

# TAB 2
with tab2:
    if chronic_src is None:
        st.warning("만성질환 파일이 필요해.")
        st.stop()

    chronic_wide, chronic_long = load_chronic_tidy(chronic_src)

    st.subheader("질환별 유병률 ↔ 치료율 (전체/남자/여자)")
    group = st.selectbox("집단 선택", ["전체", "남자", "여자"], index=0)

    prev_col = f"{group}_유병률 (%)"
    trt_col  = f"{group}_치료율 (%)"

    if prev_col not in chronic_wide.columns or trt_col not in chronic_wide.columns:
        st.error("예상한 컬럼명이 데이터에 없습니다. 현재 컬럼명을 확인하세요.")
        st.write(list(chronic_wide.columns))
        st.stop()

    stats = safe_corr_and_pvalues(chronic_wide[prev_col], chronic_wide[trt_col])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("n(질환 수)", stats["n"])
    m2.metric("Pearson r", f"{stats['pearson_r']:.4f}" if np.isfinite(stats["pearson_r"]) else "NA")
    m3.metric("Spearman ρ", f"{stats['spearman_rho']:.4f}" if np.isfinite(stats["spearman_rho"]) else "NA")
    m4.metric("p-value", "OK" if np.isfinite(stats.get("pearson_p", np.nan)) else "scipy 없음/미계산")

    if np.isfinite(stats.get("pearson_p", np.nan)):
        st.write({"Pearson p": float(stats["pearson_p"]), "Spearman p": float(stats["spearman_p"])})

    fig = scatter_with_regline(
        chronic_wide[prev_col], chronic_wide[trt_col],
        title=f"{group}: 질환별 유병률 vs 치료율",
        xlabel="유병률(%)", ylabel="치료율(%)"
    )
    st.pyplot(fig, use_container_width=True)

    st.divider()
    st.subheader("만성질환 wide 컬럼 자유 선택 상관")
    numeric_cols = [c for c in chronic_wide.columns if c != "질환"]
    xcol = st.selectbox("X", numeric_cols, index=0)
    ycol = st.selectbox("Y", numeric_cols, index=min(1, len(numeric_cols)-1))

    stats2 = safe_corr_and_pvalues(chronic_wide[xcol], chronic_wide[ycol])
    st.write(stats2)

    fig2 = scatter_with_regline(
        chronic_wide[xcol], chronic_wide[ycol],
        title=f"만성질환: {xcol} vs {ycol}",
        xlabel=xcol, ylabel=ycol
    )
    st.pyplot(fig2, use_container_width=True)

# TAB 3
with tab3:
    if subj_src is None:
        st.warning("주관적 건강상태 파일이 필요해.")
        st.stop()

    subj = load_subjective_tidy(subj_src)

    st.subheader("특성 선택 후 상관/시각화")
    features = sorted(subj["특성1"].dropna().unique().tolist())
    if not features:
        st.info("특성(1) 값이 없습니다.")
        st.stop()

    feature = st.selectbox("특성(1)", features, index=0)
    df = subj[subj["특성1"] == feature].copy()

    pct_cols = [c for c in df.columns if str(c).endswith("(%)")]
    left, right = st.columns(2)

    with left:
        st.dataframe(df, use_container_width=True, height=380)

    with right:
        if len(pct_cols) >= 2:
            xcol = st.selectbox("X(%)", pct_cols, index=0, key="subj_x")
            ycol = st.selectbox("Y(%)", pct_cols, index=1, key="subj_y")

            stats = safe_corr_and_pvalues(df[xcol], df[ycol])
            st.write(stats)

            fig = scatter_with_regline(
                df[xcol], df[ycol],
                title=f"{feature}: {xcol} vs {ycol}",
                xlabel=xcol, ylabel=ycol
            )
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("(% )컬럼이 2개 이상 있어야 상관을 계산할 수 있습니다.")

    st.divider()
    st.subheader("연령대: 연령 ↔ '나쁨(poor%)' 상관 + 추이")

    age_df = subj[subj["특성1"].astype(str).str.contains("연령", na=False)].copy()
    if len(age_df) == 0:
        st.info("'연령' 행을 찾지 못했습니다.")
    else:
        poor1 = "건강하지 않음 (%)"
        poor2 = "전혀 건강하지 않음 (%)"

        if poor1 in age_df.columns and poor2 in age_df.columns:
            age_df["poor_pct"] = age_df[poor1] + age_df[poor2]
            age_df["age_mid"] = age_df["특성2"].apply(age_mid)
            age_df = age_df.sort_values("age_mid")

            stats = safe_corr_and_pvalues(age_df["age_mid"], age_df["poor_pct"])
            st.write(stats)

            fig, ax = plt.subplots()
            ax.plot(age_df["age_mid"], age_df["poor_pct"], marker="o")
            ax.set_title("연령대별 '나쁨(poor%)' 추이")
            ax.set_xlabel("연령대 중앙값")
            ax.set_ylabel("poor% (=건강하지 않음+전혀 건강하지 않음)")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
        else:
            st.warn
