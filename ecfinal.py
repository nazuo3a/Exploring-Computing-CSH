import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="노인 건강 데이터 상관/시각화", layout="wide")

# =========================
# 기본 설정(로컬에 파일이 있을 때 자동 로드)
# =========================
DEFAULT_CHRONIC = "노인의_성별_만성질병_종류별_유병률_의사진단_기준__및_현_치료율_20251214174945.xlsx"
DEFAULT_SUBJ    = "노인의_주관적_건강상태_20251214230000.xlsx"

# =========================
# 유틸
# =========================
def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace("%", "", regex=False).str.strip(), errors="coerce")

def safe_corr_and_pvalues(x, y):
    """scipy 있으면 p-value까지, 없으면 r만"""
    x = pd.Series(x)
    y = pd.Series(y)
    mask = x.notna() & y.notna()
    x = x[mask].astype(float)
    y = y[mask].astype(float)

    out = {"n": len(x)}
    if len(x) < 3:
        out.update({"pearson_r": np.nan, "pearson_p": np.nan, "spearman_rho": np.nan, "spearman_p": np.nan})
        return out

    # r
    out["pearson_r"] = float(x.corr(y, method="pearson"))
    out["spearman_rho"] = float(x.corr(y, method="spearman"))

    # p-value (optional)
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

    # 첫 컬럼명 보정
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

    measures = raw.iloc[0].tolist()
    cols = list(raw.columns[:2]) + [str(m) for m in measures[2:]]
    df = raw.iloc[1:].copy()
    df.columns = cols
    df = df.rename(columns={df.columns[0]: "특성1", df.columns[1]: "특성2"})

    for c in df.columns[2:]:
        df[c] = to_numeric(df[c])

    df["특성1"] = df["특성1"].ffill()
    return df

def age_mid(s):
    nums = list(map(int, re.findall(r"\d+", str(s))))
    if len(nums) >= 2:
        return (nums[0] + nums[1]) / 2
    if len(nums) == 1:
        return float(nums[0])
    return np.nan

# =========================
# UI
# =========================
st.title("노인 건강 데이터: 상관관계 · 시각화 · 상관계수(앱)")

with st.sidebar:
    st.header("데이터 업로드")
    chronic_file = st.file_uploader("만성질환(유병률/치료율) 엑셀", type=["xlsx"], key="chronic")
    subj_file = st.file_uploader("주관적 건강상태 엑셀", type=["xlsx"], key="subj")

    st.divider()
    st.caption("업로드하지 않으면, 로컬 기본 경로가 있으면 자동 로드합니다.")
    use_defaults = st.checkbox("기본 경로 파일 사용(가능할 때)", value=True)

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
# 탭 구성
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "데이터 미리보기",
    "만성질환 상관/시각화",
    "주관적 건강 상관/시각화",
    "히트맵(요약)",
    "내보내기"
])

# =========================
# TAB 1: 미리보기
# =========================
with tab1:
    c1, c2 = st.columns(2)

    if chronic_src is not None:
        chronic_wide, chronic_long = load_chronic_tidy(chronic_src)
        with c1:
            st.subheader("만성질환(2020) - wide")
            st.dataframe(chronic_wide, use_container_width=True, height=420)
            st.caption("질환 단위 집계표이므로 상관은 '질환 간' 관계를 의미")

    if subj_src is not None:
        subj = load_subjective_tidy(subj_src)
        with c2:
            st.subheader("주관적 건강상태(2023)")
            st.dataframe(subj, use_container_width=True, height=420)
            st.caption("특성(예: 성별/연령 등) 단위 집계표")

# =========================
# TAB 2: 만성질환 상관/시각화
# =========================
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
        st.error("예상한 컬럼명이 데이터에 없어서 실행할 수 없어. (헤더 라벨이 다른 경우 컬럼명 확인 필요)")
        st.write("현재 컬럼:", list(chronic_wide.columns))
        st.stop()

    stats = safe_corr_and_pvalues(chronic_wide[prev_col], chronic_wide[trt_col])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("n(질환 수)", stats["n"])
    m2.metric("Pearson r", f"{stats['pearson_r']:.4f}" if np.isfinite(stats["pearson_r"]) else "NA")
    m3.metric("Spearman ρ", f"{stats['spearman_rho']:.4f}" if np.isfinite(stats["spearman_rho"]) else "NA")
    if np.isfinite(stats.get("pearson_p", np.nan)):
        m4.metric("Pearson p", f"{stats['pearson_p']:.3g}")
    else:
        m4.metric("p-value", "scipy 없음/미계산")

    fig = scatter_with_regline(
        chronic_wide[prev_col], chronic_wide[trt_col],
        title=f"{group}: 질환별 유병률 vs 치료율",
        xlabel="유병률(%)", ylabel="치료율(%)"
    )
    st.pyplot(fig, use_container_width=True)

    st.divider()
    st.subheader("컬럼 자유 선택 상관 (만성질환 wide)")
    numeric_cols = [c for c in chronic_wide.columns if c != "질환"]
    xcol = st.selectbox("X", numeric_cols, index=0)
    ycol = st.selectbox("Y", numeric_cols, index=min(1, len(numeric_cols)-1))

    stats2 = safe_corr_and_pvalues(chronic_wide[xcol], chronic_wide[ycol])
    st.write({
        "n": stats2["n"],
        "pearson_r": stats2["pearson_r"],
        "pearson_p": stats2.get("pearson_p", None),
        "spearman_rho": stats2["spearman_rho"],
        "spearman_p": stats2.get("spearman_p", None),
    })

    fig2 = scatter_with_regline(
        chronic_wide[xcol], chronic_wide[ycol],
        title=f"만성질환: {xcol} vs {ycol}",
        xlabel=xcol, ylabel=ycol
    )
    st.pyplot(fig2, use_container_width=True)

# =========================
# TAB 3: 주관적 건강 상관/시각화
# =========================
with tab3:
    if subj_src is None:
        st.warning("주관적 건강상태 파일이 필요해.")
        st.stop()

    subj = load_subjective_tidy(subj_src)

    st.subheader("특성 선택 후, 상관/시각화")
    features = sorted(subj["특성1"].dropna().unique().tolist())
    feature = st.selectbox("특성(1)", features, index=0 if features else None)

    df = subj[subj["특성1"] == feature].copy()

    # 퍼센트 컬럼
    pct_cols = [c for c in df.columns if str(c).endswith("(%)")]
    st.caption("퍼센트 컬럼(%) 중 2개를 골라 상관/산점도를 볼 수 있어.")

    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(df, use_container_width=True, height=360)

    with c2:
        if len(pct_cols) >= 2:
            xcol = st.selectbox("X(%)", pct_cols, index=0, key="subj_x")
            ycol = st.selectbox("Y(%)", pct_cols, index=1, key="subj_y")

            stats = safe_corr_and_pvalues(df[xcol], df[ycol])
            st.write({
                "n": stats["n"],
                "pearson_r": stats["pearson_r"],
                "pearson_p": stats.get("pearson_p", None),
                "spearman_rho": stats["spearman_rho"],
                "spearman_p": stats.get("spearman_p", None),
            })

            fig = scatter_with_regline(
                df[xcol], df[ycol],
                title=f"{feature}: {xcol} vs {ycol}",
                xlabel=xcol, ylabel=ycol
            )
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("(% )컬럼이 2개 이상 있어야 상관을 계산할 수 있어.")

    st.divider()
    st.subheader("연령대(있다면): 연령 ↔ '나쁨(poor%)' 상관 + 추이/분포")

    age_df = subj[subj["특성1"].astype(str).str.contains("연령", na=False)].copy()
    if len(age_df) == 0:
        st.info("이 파일에서 '연령' 행을 찾지 못했어.")
    else:
        poor1 = "건강하지 않음 (%)"
        poor2 = "전혀 건강하지 않음 (%)"

        if poor1 in age_df.columns and poor2 in age_df.columns:
            age_df["poor_pct"] = age_df[poor1] + age_df[poor2]
            age_df["age_mid"] = age_df["특성2"].apply(age_mid)
            age_df = age_df.sort_values("age_mid")

            stats = safe_corr_and_pvalues(age_df["age_mid"], age_df["poor_pct"])
            st.write({
                "n": stats["n"],
                "pearson_r": stats["pearson_r"],
                "pearson_p": stats.get("pearson_p", None),
                "spearman_rho": stats["spearman_rho"],
                "spearman_p": stats.get("spearman_p", None),
            })

            # line plot
            fig, ax = plt.subplots()
            ax.plot(age_df["age_mid"], age_df["poor_pct"], marker="o")
            ax.set_title("연령대별 '나쁨(poor%)' 추이")
            ax.set_xlabel("연령대 중앙값")
            ax.set_ylabel("poor% (=건강하지 않음+전혀 건강하지 않음)")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)

            # stacked bar (건강단계 분포)
            health_cols = [c for c in subj.columns if str(c).endswith("(%)") and c != "비율 (%)"]
            if len(health_cols) > 0:
                xlabels = age_df["특성2"].astype(str).tolist()
                data = age_df[health_cols].to_numpy(dtype=float)

                fig, ax = plt.subplots(figsize=(9, 5))
                bottom = np.zeros(len(age_df))
                for j, col in enumerate(health_cols):
                    ax.bar(xlabels, data[:, j], bottom=bottom, label=col)
                    bottom += np.nan_to_num(data[:, j])

                ax.set_title("연령대별 주관적 건강상태 분포(%)")
                ax.set_xlabel("연령대")
                ax.set_ylabel("비율(%)")
                ax.set_xticklabels(xlabels, rotation=45, ha="right")
                ax.legend()
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
        else:
            st.warning("poor% 계산에 필요한 컬럼명이 다름. 현재 컬럼명을 확인해줘.")
            st.write("현재 컬럼:", list(age_df.columns))

# =========================
# TAB 4: 히트맵(요약)
# =========================
with tab4:
    st.subheader("상관 히트맵 (요약)")

    if chronic_src is not None:
        chronic_wide, _ = load_chronic_tidy(chronic_src)
        numeric_cols = [c for c in chronic_wide.columns if c != "질환"]
        fig = corr_heatmap(chronic_wide[numeric_cols], title="만성질환 지표 간 상관(질환단위, Pearson)")
        st.pyplot(fig, use_container_width=True)

    if subj_src is not None:
        subj = load_subjective_tidy(subj_src)
        # 퍼센트 컬럼만 뽑아 히트맵(전체를 섞으면 특성행들이 함께 있어 해석이 애매할 수 있음)
        pct_cols = [c for c in subj.columns if str(c).endswith("(%)")]
        if len(pct_cols) >= 2:
            # 전체 데이터에서 퍼센트 컬럼들 간 상관(행=특성 조합 단위)
            fig = corr_heatmap(subj[pct_cols], title="주관적 건강상태 퍼센트 컬럼 간 상관(Pearson)")
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("주관적 건강상태에서 (% )컬럼이 2개 이상이어야 히트맵 가능")

# =========================
# TAB 5: 내보내기
# =========================
with tab5:
    st.subheader("전처리된 데이터 다운로드")

    if chronic_src is not None:
        chronic_wide, chronic_long = load_chronic_tidy(chronic_src)
        st.download_button(
            "만성질환 wide CSV 다운로드",
            data=chronic_wide.to_csv(index=False).encode("utf-8-sig"),
            file_name="chronic_wide.csv",
            mime="text/csv"
        )
        st.download_button(
            "만성질환 long CSV 다운로드",
            data=chronic_long.to_csv(index=False).encode("utf-8-sig"),
            file_name="chronic_long.csv",
            mime="text/csv"
        )

    if subj_src is not None:
        subj = load_subjective_tidy(subj_src)
        st.download_button(
            "주관적 건강상태 CSV 다운로드",
            data=subj.to_csv(index=False).encode("utf-8-sig"),
            file_name="subjective_health.csv",
            mime="text/csv"
        )

    st.caption("Streamlit Cloud 배포 시, requirements.txt에 openpyxl/pandas/matplotlib/numpy(그리고 p-value 원하면 scipy) 포함 권장.")
