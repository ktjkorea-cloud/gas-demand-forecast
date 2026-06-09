import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import io

REGIONS = ["안동", "영주", "예천", "봉화", "의성", "군위"]
MENU = REGIONS + ["📊 전체 합계"]

st.set_page_config(page_title="대성청정에너지 도시가스 수요량 예측", page_icon="🔥", layout="wide")

st.markdown("""
<style>
/* 사이드바 너비 확장 */
[data-testid="stSidebar"] {
    min-width: 260px;
    max-width: 260px;
}

/* 라디오 버튼 목록 */
[data-testid="stSidebar"] .stRadio {
    margin-top: -40px;
}
[data-testid="stSidebar"] .stRadio > div {
    display: flex;
    flex-direction: column;
    gap: 0px;
}
[data-testid="stSidebar"] .stRadio label {
    display: flex !important;
    align-items: center;
    padding: 10px 14px !important;
    font-size: 2rem !important;
    font-weight: 600 !important;
    color: rgb(49, 51, 63) !important;
    border-radius: 10px;
    cursor: pointer;
    border: 2px solid transparent;
    transition: background 0.2s, border 0.2s;
    line-height: 1 !important;
    min-height: 52px;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255, 107, 53, 0.12);
    border: 2px solid #FF6B35;
}
[data-testid="stSidebar"] .stRadio label:has(input:checked) {
    background: rgba(255, 107, 53, 0.18);
    border: 2px solid #FF6B35;
    color: #FF6B35 !important;
}
/* 라디오 원형 아이콘 숨기기 */
[data-testid="stSidebar"] .stRadio input[type="radio"] {
    display: none;
}
</style>
""", unsafe_allow_html=True)

# ── session_state 초기화 ──────────────────────────────────────────
for r in REGIONS:
    if f"df_{r}" not in st.session_state:
        st.session_state[f"df_{r}"] = None
    if f"cfg_{r}" not in st.session_state:
        st.session_state[f"cfg_{r}"] = {"forecast_days": 28, "n_estimators": 100}
    if f"result_{r}" not in st.session_state:
        st.session_state[f"result_{r}"] = None

# ── 사이드바: 지역 선택 ───────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
        border-radius: 12px;
        padding: 20px 14px;
        margin-bottom: 8px;
        text-align: center;
        border: 1px solid rgba(255,107,53,0.3);
        box-shadow: 0 4px 15px rgba(255,107,53,0.15);
    ">
        <img src="https://www.daesungcleanenergy.co.kr/images/common/ci.svg"
             style="width:75%; max-width:160px; filter: brightness(0) invert(1);"
             onerror="this.style.display='none'">
        <div style="
            color: #ffffff;
            font-size: 26px;
            font-weight: 700;
            margin-top: 8px;
            opacity: 0.85;
        ">도시가스<br>수요량 예측</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<p style='font-size:1.1rem; font-weight:700; margin:0; padding:0;'>🗺️ 지역 선택</p>", unsafe_allow_html=True)
    selected = st.radio(
        "지역 선택",
        MENU,
        label_visibility="hidden"
    )

# ── 공통 함수 ─────────────────────────────────────────────────────
ORIGIN_DATE = pd.Timestamp("2023-01-01")
DOW_LABELS  = ["월", "화", "수", "목", "금", "토", "일"]

def apply_weights(df_fut, weights):
    """날짜 컬럼 기준으로 요일별 가중치를 예측값에 곱함"""
    df_fut = df_fut.copy()
    dow = pd.to_datetime(df_fut["날짜"]).dt.weekday  # 0=월 ~ 6=일
    df_fut["요일"]   = dow.map(lambda x: DOW_LABELS[x])
    df_fut["가중치"] = dow.map(lambda x: weights[x])
    df_fut["예측_수요량(MJ)"] = (df_fut["예측_수요량(MJ)"] * df_fut["가중치"]).round(1)
    return df_fut

def add_features(df, temp_col=None):
    df = df.copy()
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["월"]     = df["날짜"].dt.month
    df["일"]     = df["날짜"].dt.day
    df["요일번호"] = df["날짜"].dt.weekday
    df["주말여부"] = (df["요일번호"] >= 5).astype(int)
    df["분기"]   = df["날짜"].dt.quarter
    df["월_sin"] = np.sin(2 * np.pi * df["월"] / 12)
    df["월_cos"] = np.cos(2 * np.pi * df["월"] / 12)
    # 3차 다항식 트렌드 특성 (기준일로부터 경과 일수)
    t = (df["날짜"] - ORIGIN_DATE).dt.days / 365.0
    df["추세_t"]  = t
    df["추세_t2"] = t ** 2
    df["추세_t3"] = t ** 3
    feat_cols = ["월", "일", "요일번호", "주말여부", "분기", "월_sin", "월_cos",
                 "추세_t", "추세_t2", "추세_t3"]
    if temp_col and temp_col in df.columns:
        feat_cols.append(temp_col)
    return df, feat_cols

def run_model(df, demand_col, temp_col, n_estimators, forecast_days):
    df_f, feat_cols = add_features(df, temp_col)
    X, y = df_f[feat_cols], df_f[demand_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    future_dates = pd.date_range(
        start=df["날짜"].max() + pd.Timedelta(days=1), periods=forecast_days)
    fut = pd.DataFrame({"날짜": future_dates})
    fut_f, _ = add_features(fut, temp_col)
    if temp_col and temp_col in df_f.columns:
        monthly_temp = df_f.groupby("월")[temp_col].mean()
        fut_f[temp_col] = fut_f["월"].map(monthly_temp)
    fut["예측_수요량(MJ)"] = model.predict(fut_f[feat_cols]).round(1)

    return y_pred, y_test, df_f, fut, mae, r2

def load_uploaded(file):
    df = pd.read_csv(file, encoding="utf-8-sig")
    df["날짜"] = pd.to_datetime(df["날짜"])
    return df.sort_values("날짜").reset_index(drop=True)

def make_sample(region):
    SEASONAL = {1:2.5,2:2.3,3:1.6,4:1.1,5:0.8,
                6:0.5,7:0.4,8:0.5,9:0.8,10:1.2,11:1.8,12:2.4}
    BASES = {"안동":1200,"영주":900,"군위":400,"의성":500,"예천":550,"봉화":350}
    SCALE = {"안동":1.0,"영주":0.85,"군위":0.6,"의성":0.65,"예천":0.7,"봉화":0.55}
    np.random.seed(hash(region) % (2**31))
    dates = pd.date_range("2023-01-01", "2024-12-31")
    rows = []
    for d in dates:
        weekend = 0.9 if d.weekday() >= 5 else 1.0
        temp    = round(20 - 15*np.cos(2*np.pi*(d.month-1)/12) + np.random.normal(0,2), 1)
        demand  = BASES[region] * SEASONAL[d.month] * weekend * SCALE[region] * np.random.normal(1,0.05)
        rows.append({"날짜": d.strftime("%Y-%m-%d"),
                     "수요량(MJ)": round(demand,1),
                     "기온(°C)": temp,
                     "요일": ["월","화","수","목","금","토","일"][d.weekday()]})
    df = pd.DataFrame(rows)
    df["날짜"] = pd.to_datetime(df["날짜"])
    return df

# ── 지역 페이지 ───────────────────────────────────────────────────
def page_region(region):
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #56CCF2 0%, #2F80ED 100%);
        border-radius: 14px;
        padding: 18px 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(47,128,237,0.3);
        display: flex;
        align-items: center;
        gap: 12px;
    ">
        <img src="https://www.daesungcleanenergy.co.kr/images/common/ci.svg"
             style="height:33px; filter: brightness(0) invert(1); display:none;"
             onload="this.style.display='inline'">
        <div style="color:#ffffff; font-size:33px; font-weight:800; white-space:nowrap;">
            {region} 도시가스 수요량 예측
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 사이드바 내 지역별 설정
    with st.sidebar:
        st.markdown(f"### ⚙️ {region} 설정")
        forecast_days = st.slider("예측 기간 (일)", 7, 360,
                                   st.session_state[f"cfg_{region}"]["forecast_days"], 7,
                                   key=f"fd_{region}")
        n_estimators  = st.slider("모델 복잡도", 50, 300,
                                   st.session_state[f"cfg_{region}"]["n_estimators"], 50,
                                   key=f"ne_{region}")
        st.session_state[f"cfg_{region}"] = {"forecast_days": forecast_days,
                                              "n_estimators": n_estimators}

        st.markdown("### ⚖️ 요일별 가중치")
        st.caption("1.0 = 기본, 1.1 = 10% 증가, 0.9 = 10% 감소")
        DOW_LABELS = ["월", "화", "수", "목", "금", "토", "일"]
        DOW_DEFAULTS = [1.0, 1.0, 1.0, 1.0, 1.0, 0.95, 0.95]
        weights = []
        for i, (label, default) in enumerate(zip(DOW_LABELS, DOW_DEFAULTS)):
            w = st.number_input(
                f"{label}요일",
                min_value=0.1, max_value=3.0,
                value=float(st.session_state.get(f"w_{region}_{i}", default)),
                step=0.05, format="%.2f",
                key=f"w_{region}_{i}"
            )
            weights.append(w)

        st.markdown("### 📂 데이터 업로드")
        uploaded = st.file_uploader(f"{region} CSV 업로드",
                                     type=["csv"], key=f"up_{region}",
                                     help="날짜, 수요량(MJ), 기온(°C) 컬럼 필요")
        if uploaded:
            st.session_state[f"df_{region}"] = load_uploaded(uploaded)
            st.success("업로드 완료!")


    df = st.session_state[f"df_{region}"]
    if df is None:
        st.info("👈 왼쪽에서 데이터를 업로드하거나 샘플 데이터를 사용하세요.")
        st.markdown("""
        **CSV 형식 예시**
        ```
        날짜,수요량(MJ),기온(°C),요일
        2023-01-01,2305.9,3.0,일
        2023-01-02,2482.7,5.8,월
        ```
        """)
        return

    demand_col = next((c for c in df.columns if "수요량" in c), None)
    temp_col   = next((c for c in df.columns if "기온" in c), None)

    tab1, tab2, tab3, tab4 = st.tabs(["📊 데이터 탐색", "🤖 예측 모델", "📐 다항식 예측", "📥 다운로드"])

    # ── 데이터 탐색 ──────────────────────────────────────────────
    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("데이터 기간", f"{len(df)}일")
        c2.metric("평균 수요량", f"{df[demand_col].mean():,.0f} MJ")
        c3.metric("최대 수요량", f"{df[demand_col].max():,.0f} MJ")
        c4.metric("최소 수요량", f"{df[demand_col].min():,.0f} MJ")

        fig1 = px.line(df, x="날짜", y=demand_col,
                       title=f"{region} 일별 수요량",
                       labels={demand_col: "수요량 (MJ)"},
                       color_discrete_sequence=["#FF6B35"])
        fig1.update_layout(hovermode="x unified")
        st.plotly_chart(fig1, use_container_width=True)

        ca, cb = st.columns(2)
        with ca:
            m_avg = df.copy()
            m_avg["월"] = m_avg["날짜"].dt.month
            m_avg = m_avg.groupby("월")[demand_col].mean().reset_index()
            m_avg["월"] = m_avg["월"].astype(str) + "월"
            fig2 = px.bar(m_avg, x="월", y=demand_col, title="월별 평균 수요량",
                          color=demand_col, color_continuous_scale="Oranges",
                          labels={demand_col: "평균 수요량 (MJ)"})
            st.plotly_chart(fig2, use_container_width=True)
        with cb:
            if temp_col:
                fig3 = px.scatter(df, x=temp_col, y=demand_col,
                                  title="기온 vs 수요량",
                                  labels={temp_col: "기온 (°C)", demand_col: "수요량 (MJ)"},
                                  color_discrete_sequence=["#4A90D9"])
                st.plotly_chart(fig3, use_container_width=True)

        st.dataframe(df, use_container_width=True, height=280)

    # ── 예측 모델 ────────────────────────────────────────────────
    with tab2:
        with st.spinner("모델 학습 중..."):
            y_pred, y_test, df_f, fut, mae, r2 = run_model(
                df, demand_col, temp_col, n_estimators, forecast_days)

        cm1, cm2 = st.columns(2)
        cm1.metric("평균 절대 오차 (MAE)", f"{mae:,.1f} MJ")
        cm2.metric("결정계수 (R²)", f"{r2:.3f}")

        test_dates = df_f["날짜"].iloc[-len(y_test):]
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=test_dates, y=y_test.values,
                                   name="실제값", line=dict(color="#FF6B35")))
        fig4.add_trace(go.Scatter(x=test_dates, y=y_pred,
                                   name="예측값", line=dict(color="#4A90D9", dash="dash")))
        fig4.update_layout(title="실제값 vs 예측값",
                           xaxis_title="날짜", yaxis_title="수요량 (MJ)",
                           hovermode="x unified")
        st.plotly_chart(fig4, use_container_width=True)

        # 요일 가중치 적용
        fut_w = apply_weights(fut, weights)

        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=df["날짜"].iloc[-60:], y=df[demand_col].iloc[-60:],
                                   name="과거 실제값", line=dict(color="#FF6B35")))
        fig5.add_trace(go.Scatter(x=fut_w["날짜"], y=fut["예측_수요량(MJ)"],
                                   name="가중치 적용 전",
                                   line=dict(color="#A0A0A0", dash="dot")))
        fig5.add_trace(go.Scatter(x=fut_w["날짜"], y=fut_w["예측_수요량(MJ)"],
                                   name=f"가중치 적용 후 ({forecast_days}일)",
                                   line=dict(color="#4A90D9", dash="dash"),
                                   fill="tozeroy", fillcolor="rgba(74,144,217,0.1)"))
        fig5.update_layout(title=f"향후 {forecast_days}일 예측 (요일 가중치 적용)",
                           xaxis_title="날짜", yaxis_title="수요량 (MJ)",
                           hovermode="x unified")
        st.plotly_chart(fig5, use_container_width=True)

        # 요일별 가중치 적용 결과 테이블
        st.markdown("##### 📋 요일별 가중치 적용 결과")
        show_cols = ["날짜", "요일", "가중치", "예측_수요량(MJ)"]
        st.dataframe(
            fut_w[show_cols].assign(날짜=fut_w["날짜"].dt.strftime("%Y-%m-%d")),
            use_container_width=True, height=250
        )

        fut_w["지역"] = region
        st.session_state[f"result_{region}"] = fut_w

    # ── 다항식 예측 ──────────────────────────────────────────────
    with tab3:
        st.markdown("### 📐 3차 다항식 회귀 예측")
        st.caption("시간 흐름에 따른 장기 트렌드를 3차 곡선으로 모델링합니다.")

        # 날짜 → 숫자(경과일) 변환
        t_all = (df["날짜"] - ORIGIN_DATE).dt.days.values
        y_all = df[demand_col].values

        # 자동 피팅으로 초기 계수 계산
        coeffs_auto = np.polyfit(t_all, y_all, 3)
        a0, b0, c0, d0 = coeffs_auto

        # 계수 편집 UI
        st.markdown("#### ✏️ 계수 직접 편집")
        st.caption("슬라이더로 조정하거나 숫자를 직접 입력하세요. 차트가 실시간으로 바뀝니다.")

        col_reset, _ = st.columns([1, 3])
        reset = col_reset.button("↺ 자동 피팅값으로 초기화", key=f"reset_poly_{region}")

        ea, eb, ec, ed = st.columns(4)
        a = ea.number_input("3차 계수 (a)", value=float(a0), format="%.6f",
                             step=float(abs(a0) * 0.1) or 1e-6,
                             key=f"pa_{region}" if not reset else f"pa_{region}_r")
        b = eb.number_input("2차 계수 (b)", value=float(b0), format="%.4f",
                             step=float(abs(b0) * 0.1) or 1e-4,
                             key=f"pb_{region}" if not reset else f"pb_{region}_r")
        c = ec.number_input("1차 계수 (c)", value=float(c0), format="%.2f",
                             step=float(abs(c0) * 0.1) or 0.1,
                             key=f"pc_{region}" if not reset else f"pc_{region}_r")
        d = ed.number_input("상수항 (d)", value=float(d0), format="%.1f",
                             step=float(abs(d0) * 0.01) or 1.0,
                             key=f"pd_{region}" if not reset else f"pd_{region}_r")

        # 편집된 계수로 계산
        poly_edit    = np.poly1d([a, b, c, d])
        y_poly_fit   = poly_edit(t_all)
        r2_poly      = r2_score(y_all, y_poly_fit)
        mae_poly     = mean_absolute_error(y_all, y_poly_fit)

        # 수식 표시
        st.latex(rf"f(t) = {a:.4e}\,t^3 + {b:.4f}\,t^2 + {c:.2f}\,t + {d:,.1f}")

        # 정확도
        pm1, pm2 = st.columns(2)
        pm1.metric("결정계수 (R²)", f"{r2_poly:.3f}")
        pm2.metric("평균 절대 오차 (MAE)", f"{mae_poly:,.1f} MJ")

        # 실제값 vs 다항식 피팅 차트
        fig_p1 = go.Figure()
        fig_p1.add_trace(go.Scatter(x=df["날짜"], y=y_all,
                                     name="실제값", line=dict(color="#FF6B35")))
        fig_p1.add_trace(go.Scatter(x=df["날짜"], y=y_poly_fit,
                                     name="3차 다항식 피팅",
                                     line=dict(color="#9B59B6", dash="dash", width=2)))
        fig_p1.update_layout(title="실제값 vs 3차 다항식 피팅",
                              xaxis_title="날짜", yaxis_title="수요량 (MJ)",
                              hovermode="x unified")
        st.plotly_chart(fig_p1, use_container_width=True)

        # 미래 예측 + 요일 가중치 적용
        st.markdown("#### 🔮 다항식 기반 미래 예측")
        future_dates = pd.date_range(
            start=df["날짜"].max() + pd.Timedelta(days=1), periods=forecast_days)
        t_future = (future_dates - ORIGIN_DATE).days.values
        y_future_poly = poly_edit(t_future)

        fut_poly = pd.DataFrame({"날짜": future_dates,
                                  "예측_수요량(MJ)": y_future_poly.round(1)})
        fut_poly_w = apply_weights(fut_poly, weights)

        fig_p2 = go.Figure()
        fig_p2.add_trace(go.Scatter(x=df["날짜"].iloc[-60:], y=df[demand_col].iloc[-60:],
                                     name="과거 실제값", line=dict(color="#FF6B35")))
        fig_p2.add_trace(go.Scatter(x=future_dates, y=y_future_poly,
                                     name="가중치 적용 전",
                                     line=dict(color="#A0A0A0", dash="dot")))
        fig_p2.add_trace(go.Scatter(x=future_dates, y=fut_poly_w["예측_수요량(MJ)"],
                                     name=f"가중치 적용 후 ({forecast_days}일)",
                                     line=dict(color="#9B59B6", dash="dash", width=2),
                                     fill="tozeroy", fillcolor="rgba(155,89,182,0.1)"))
        fig_p2.update_layout(title=f"3차 다항식 향후 {forecast_days}일 예측 (요일 가중치 적용)",
                              xaxis_title="날짜", yaxis_title="수요량 (MJ)",
                              hovermode="x unified")
        st.plotly_chart(fig_p2, use_container_width=True)

        st.markdown("##### 📋 요일별 가중치 적용 결과")
        st.dataframe(
            fut_poly_w[["날짜","요일","가중치","예측_수요량(MJ)"]].assign(
                날짜=fut_poly_w["날짜"].dt.strftime("%Y-%m-%d")),
            use_container_width=True, height=250
        )

        # RandomForest vs 다항식 비교 차트
        with st.spinner("RandomForest 모델 비교 중..."):
            _, _, df_f2, fut2, mae_rf, r2_rf = run_model(
                df, demand_col, temp_col, n_estimators, forecast_days)

        st.markdown("#### ⚖️ RandomForest vs 3차 다항식 비교")
        comp1, comp2 = st.columns(2)
        comp1.metric("RandomForest R²", f"{r2_rf:.3f}")
        comp2.metric("3차 다항식 R²", f"{r2_poly:.3f}",
                     delta=f"{r2_poly - r2_rf:+.3f} vs RF")

        fig_p3 = go.Figure()
        fig_p3.add_trace(go.Scatter(x=future_dates, y=fut2["예측_수요량(MJ)"],
                                     name="RandomForest 예측",
                                     line=dict(color="#4A90D9", dash="dash")))
        fig_p3.add_trace(go.Scatter(x=future_dates, y=y_future_poly,
                                     name="3차 다항식 예측",
                                     line=dict(color="#9B59B6", dash="dot")))
        fig_p3.update_layout(title="두 모델 예측 비교",
                              xaxis_title="날짜", yaxis_title="수요량 (MJ)",
                              hovermode="x unified")
        st.plotly_chart(fig_p3, use_container_width=True)

    # ── 다운로드 ─────────────────────────────────────────────────
    with tab4:
        result = st.session_state[f"result_{region}"]
        if result is None:
            st.info("먼저 '예측 모델' 탭을 실행하세요.")
        else:
            st.dataframe(result[["지역","날짜","예측_수요량(MJ)"]],
                         use_container_width=True)
            buf = io.BytesIO()
            result[["지역","날짜","예측_수요량(MJ)"]].to_csv(
                buf, index=False, encoding="utf-8-sig")
            st.download_button(
                label=f"⬇️ {region} 예측 결과 CSV 다운로드",
                data=buf.getvalue(),
                file_name=f"{region}_수요량예측.csv",
                mime="text/csv",
                use_container_width=True
            )

# ── 전체 합계 페이지 ──────────────────────────────────────────────
def page_total():
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #56CCF2 0%, #2F80ED 100%);
        border-radius: 14px;
        padding: 26px 32px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(47,128,237,0.3);
        display: flex;
        align-items: center;
        gap: 20px;
    ">
        <img src="https://www.daesungcleanenergy.co.kr/images/common/ci.svg"
             style="height:33px; filter: brightness(0) invert(1); display:none;"
             onload="this.style.display='inline'">
        <div style="color:#ffffff; font-size:33px; font-weight:800; white-space:nowrap;">
            전체 지역 도시가스 수요량 예측
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 데이터가 있는 지역 수집
    loaded = {r: st.session_state[f"df_{r}"] for r in REGIONS
              if st.session_state[f"df_{r}"] is not None}
    results = {r: st.session_state[f"result_{r}"] for r in REGIONS
               if st.session_state[f"result_{r}"] is not None}

    if not loaded:
        st.info("👈 왼쪽 메뉴에서 각 지역을 선택하고 데이터를 업로드해주세요.")
        return

    # 현황 요약 카드
    cols = st.columns(len(loaded))
    for col, (region, df) in zip(cols, loaded.items()):
        demand_col = next((c for c in df.columns if "수요량" in c), None)
        col.metric(f"🏙️ {region}", f"{df[demand_col].mean():,.0f} MJ",
                   "일 평균")

    st.markdown("---")

    # 지역별 일별 수요량 비교 차트
    st.subheader("📈 지역별 수요량 추이 비교")
    frames = []
    for region, df in loaded.items():
        demand_col = next((c for c in df.columns if "수요량" in c), None)
        tmp = df[["날짜", demand_col]].copy()
        tmp["지역"] = region
        tmp = tmp.rename(columns={demand_col: "수요량(MJ)"})
        frames.append(tmp)
    df_all = pd.concat(frames)

    # 월별 집계
    df_all["월"] = df_all["날짜"].dt.to_period("M").astype(str)
    monthly = df_all.groupby(["월","지역"])["수요량(MJ)"].mean().reset_index()
    fig1 = px.line(monthly, x="월", y="수요량(MJ)", color="지역",
                   title="지역별 월평균 수요량 추이",
                   category_orders={"지역": REGIONS})
    fig1.update_xaxes(tickangle=45)
    st.plotly_chart(fig1, use_container_width=True)

    # 지역별 합산 파이 차트
    st.subheader("🥧 지역별 수요 비중")
    share = df_all.groupby("지역")["수요량(MJ)"].mean().reindex(REGIONS).dropna()
    fig2 = px.pie(values=share.values, names=share.index,
                  title="지역별 평균 수요량 비중",
                  category_orders={"지역": REGIONS})
    st.plotly_chart(fig2, use_container_width=True)

    # 예측 결과 합계
    if results:
        st.markdown("---")
        st.subheader("🔮 전체 지역 예측 합계")

        pred_frames = []
        for region, res in results.items():
            tmp = res[["날짜","예측_수요량(MJ)"]].copy()
            tmp["지역"] = region
            pred_frames.append(tmp)
        df_pred = pd.concat(pred_frames)

        # 지역별 예측 차트
        fig3 = px.line(df_pred, x="날짜", y="예측_수요량(MJ)", color="지역",
                       title="지역별 예측 수요량",
                       category_orders={"지역": REGIONS})
        st.plotly_chart(fig3, use_container_width=True)

        # 날짜별 합산
        total_pred = df_pred.groupby("날짜")["예측_수요량(MJ)"].sum().reset_index()
        total_pred.columns = ["날짜", "전체합계_수요량(MJ)"]
        fig4 = px.bar(total_pred, x="날짜", y="전체합계_수요량(MJ)",
                      title="전체 지역 합산 예측 수요량",
                      color_discrete_sequence=["#FF6B35"])
        st.plotly_chart(fig4, use_container_width=True)

        # 합계 테이블
        st.subheader("📋 지역별 예측 결과 테이블")
        pivot = df_pred.pivot(index="날짜", columns="지역", values="예측_수요량(MJ)")
        pivot = pivot.reindex(columns=[r for r in REGIONS if r in pivot.columns])
        pivot["합계"] = pivot.sum(axis=1)
        pivot = pivot.reset_index()
        pivot["날짜"] = pivot["날짜"].dt.strftime("%Y-%m-%d")
        st.dataframe(pivot, use_container_width=True, height=350)

        # 전체 다운로드
        buf = io.BytesIO()
        pivot.to_csv(buf, index=False, encoding="utf-8-sig")
        st.download_button(
            label="⬇️ 전체 예측 결과 CSV 다운로드",
            data=buf.getvalue(),
            file_name="전체지역_수요량예측.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("각 지역 페이지의 '예측 모델' 탭을 먼저 실행하면 여기서 합산 결과를 볼 수 있습니다.")

# ── 라우팅 ────────────────────────────────────────────────────────
if selected == "📊 전체 합계":
    page_total()
else:
    page_region(selected)
