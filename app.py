
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# =============================================================
# 페이지 설정
# =============================================================
st.set_page_config(
    page_title="수능 한파는 정말 있을까?",
    page_icon="🥶",
    layout="wide",
)

# =============================================================
# 데이터 로딩 (캐싱)
# =============================================================
@st.cache_data
def load_data():
    """기상청 서울(108) 일별 기온 데이터를 불러옵니다.
    
    원본 CSV는 날짜 앞에 탭 문자가 있고, 컬럼명에 공백이 있어
    cleaning이 필요합니다.
    """
    df = pd.read_csv('ta_20260527071918.csv')
    df.columns = [c.strip() for c in df.columns]
    df['날짜'] = df['날짜'].str.strip()
    df['날짜'] = pd.to_datetime(df['날짜'])
    df['연도'] = df['날짜'].dt.year
    df['월'] = df['날짜'].dt.month
    df['일'] = df['날짜'].dt.day
    return df

# 실제 역대 수능일 (한국교육과정평가원 공식)
# 1994학년도 1차 수능부터 2026학년도(2025년 시행)까지
SUNEUNG_DATES = [
    '1993-08-20',  # 1994학년도 1차 (이례적, 분석에서 제외 옵션)
    '1993-11-16',  # 1994학년도 2차
    '1994-11-23', '1995-11-22', '1996-11-13', '1997-11-19', '1998-11-18',
    '1999-11-17', '2000-11-15', '2001-11-07', '2002-11-06', '2003-11-05',
    '2004-11-17', '2005-11-23', '2006-11-16', '2007-11-15', '2008-11-13',
    '2009-11-12', '2010-11-18', '2011-11-10', '2012-11-08', '2013-11-07',
    '2014-11-13', '2015-11-12', '2016-11-17', '2017-11-23', '2018-11-15',
    '2019-11-14', '2020-12-03', '2021-11-18', '2022-11-17', '2023-11-16',
    '2024-11-14', '2025-11-13',
]

df = load_data()

# =============================================================
# 헤더
# =============================================================
st.title("🥶 수능 한파는 정말 있을까?")
st.markdown(
    "**기상청 서울 관측소(108) 1907~2026 데이터로 검증하는 도시 전설**"
)

with st.expander("📋 이 앱은 무엇을 하나요? (먼저 읽어보기)"):
    st.markdown("""
    매년 11월이면 뉴스에서 **"수능 한파가 찾아왔습니다"**라는 표현을 자주 듣습니다.
    그런데 정말 수능날만 유독 추운 걸까요? 아니면 우리가 '한파였던 해'만 강하게
    기억하는 걸까요? (이걸 통계학에서는 **확증 편향**이라고 부릅니다.)
    
    이 앱에서는 다음 세 가지 방법으로 검증합니다.
    
    1. **수능 당일 기온** vs **수능 ±3일 평균 기온** 비교 (당일이 정말 추웠나?)
    2. **하위 20% 한파 기준**과 수능일 비교 (수능일이 한파에 해당한 비율은?)
    3. **통계 검정**으로 우연인지 진짜 패턴인지 확인 (paired t-test)
    
    > 💡 **수업 활용 팁**: 각 탭을 순서대로 보면서 학생들에게 "이 결과를
    > 어떻게 해석할 수 있을까?" 질문해 보세요. 데이터는 결론을 주지 않습니다.
    > 해석은 사람의 몫입니다.
    """)

# =============================================================
# 사이드바: 분석 옵션
# =============================================================
st.sidebar.header("⚙️ 분석 옵션")

window_days = st.sidebar.slider(
    "비교 기간 (수능 ±며칠)",
    min_value=1, max_value=14, value=3,
    help="수능 당일을 제외한 앞뒤 N일의 평균과 비교합니다."
)

cold_percentile = st.sidebar.slider(
    "한파 기준 (하위 N%)",
    min_value=5, max_value=50, value=20,
    help="11월 전체 최저기온 분포에서 하위 N%를 '한파'로 정의합니다."
)

include_1993_aug = st.sidebar.checkbox(
    "1993년 8월(1차 수능) 포함",
    value=False,
    help="1994학년도 첫 수능은 8월에 1차, 11월에 2차로 두 번 치러졌습니다. 11월만 분석하려면 체크 해제."
)

exclude_2020 = st.sidebar.checkbox(
    "2020년(코로나 12월) 제외",
    value=False,
    help="2021학년도 수능은 코로나로 12월에 치러졌습니다. 11월 한파 가설 검증에서 제외할 수 있습니다."
)

# 수능일 리스트 필터링
suneung_list = SUNEUNG_DATES.copy()
if not include_1993_aug:
    suneung_list = [d for d in suneung_list if d != '1993-08-20']
if exclude_2020:
    suneung_list = [d for d in suneung_list if d != '2020-12-03']

suneung_dt = pd.to_datetime(suneung_list)

# =============================================================
# 분석 함수
# =============================================================
def compute_comparison(df, suneung_dt, window_days):
    """수능 당일 vs ±window_days(당일 제외) 평균 비교 결과를 계산합니다."""
    rows = []
    for d in suneung_dt:
        day = df[df['날짜'] == d]
        if day.empty or day['최저기온(℃)'].isna().all():
            continue
        # ±window_days 범위에서 당일은 제외
        win = df[(df['날짜'] >= d - pd.Timedelta(days=window_days)) &
                 (df['날짜'] <= d + pd.Timedelta(days=window_days)) &
                 (df['날짜'] != d)]
        if win.empty:
            continue
        rows.append({
            '수능일': d,
            '연도': d.year,
            '당일_평균기온': day['평균기온(℃)'].iloc[0],
            '당일_최저기온': day['최저기온(℃)'].iloc[0],
            '당일_최고기온': day['최고기온(℃)'].iloc[0],
            '주변_평균기온': win['평균기온(℃)'].mean(),
            '주변_최저기온': win['최저기온(℃)'].mean(),
            '평균기온_차이': day['평균기온(℃)'].iloc[0] - win['평균기온(℃)'].mean(),
            '최저기온_차이': day['최저기온(℃)'].iloc[0] - win['최저기온(℃)'].mean(),
        })
    return pd.DataFrame(rows)

res = compute_comparison(df, suneung_dt, window_days)

# 한파 임계값: 1994년 이후 11월 최저기온 하위 N%
nov_data = df[(df['월'] == 11) & (df['연도'] >= 1994)].dropna(subset=['최저기온(℃)'])
cold_threshold = nov_data['최저기온(℃)'].quantile(cold_percentile / 100)

# =============================================================
# 핵심 지표 (한눈에 보는 결론)
# =============================================================
st.header("📊 한눈에 보는 결론")

avg_diff = res['평균기온_차이'].mean()
cold_count = (res['당일_최저기온'] <= cold_threshold).sum()
cold_rate = cold_count / len(res) * 100
t_stat, p_value = stats.ttest_rel(res['당일_최저기온'], res['주변_최저기온'])

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "분석한 수능 횟수",
        f"{len(res)}회",
        help="결측치를 제외한 분석 가능 횟수"
    )
with col2:
    delta_str = f"{avg_diff:+.2f}℃"
    st.metric(
        "수능일이 주변보다",
        delta_str,
        delta=delta_str + " (음수면 더 추웠음)",
        help=f"수능 당일 평균기온 - 주변 ±{window_days}일 평균"
    )
with col3:
    st.metric(
        f"한파(하위 {cold_percentile}%) 적중률",
        f"{cold_rate:.1f}%",
        delta=f"{cold_rate - cold_percentile:+.1f}%p (기대치 대비)",
        delta_color="inverse",
        help=f"순수 우연이면 {cold_percentile}%여야 합니다."
    )
with col4:
    significance = "✅ 유의미" if p_value < 0.05 else "❌ 유의미하지 않음"
    st.metric(
        "통계 검정 (p-value)",
        f"{p_value:.3f}",
        delta=significance,
        delta_color="off",
        help="p<0.05이면 우연이 아닌 진짜 차이일 가능성이 높습니다."
    )

# 결론 메시지
if p_value < 0.05 and avg_diff < 0:
    verdict = "🥶 **수능 한파는 통계적으로 실재합니다.**"
    color = "error"
elif p_value < 0.05 and avg_diff > 0:
    verdict = "☀️ **놀랍게도 수능일은 통계적으로 주변보다 따뜻합니다!**"
    color = "warning"
else:
    verdict = (
        "🤔 **'수능 한파'는 통계적으로 입증되지 않습니다.** "
        "수능일 기온은 주변 며칠과 의미 있는 차이가 없습니다. "
        "우리가 '추웠던 해'만 강하게 기억하는 확증 편향일 가능성이 큽니다."
    )
    color = "info"

if color == "info":
    st.info(verdict)
elif color == "warning":
    st.warning(verdict)
else:
    st.error(verdict)

# =============================================================
# 탭 구조
# =============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1️⃣ 수능일 vs 주변",
    "2️⃣ 한파 기준 비교",
    "3️⃣ 통계 검정",
    "4️⃣ 원본 데이터 탐색",
    "5️⃣ 함께 생각해보기",
])

# -------------------------------------------------------------
# 탭 1: 수능일 vs 주변
# -------------------------------------------------------------
with tab1:
    st.subheader(f"수능 당일 기온 vs 수능 ±{window_days}일(당일 제외) 평균")
    st.markdown(
        f"각 막대는 한 해의 수능입니다. "
        f"**파란색(음수)** = 주변보다 추웠음, **빨간색(양수)** = 주변보다 따뜻했음."
    )

    fig = go.Figure()
    colors = ['#3498db' if v < 0 else '#e74c3c' for v in res['최저기온_차이']]
    fig.add_trace(go.Bar(
        x=res['연도'],
        y=res['최저기온_차이'],
        marker_color=colors,
        text=[f"{v:+.1f}℃" for v in res['최저기온_차이']],
        textposition='outside',
        hovertemplate='<b>%{x}학년도 수능</b><br>최저기온 차이: %{y:+.2f}℃<extra></extra>',
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_hline(
        y=res['최저기온_차이'].mean(),
        line_dash="dot", line_color="purple",
        annotation_text=f"평균: {res['최저기온_차이'].mean():+.2f}℃",
        annotation_position="right",
    )
    fig.update_layout(
        xaxis_title="수능 시행 연도",
        yaxis_title=f"최저기온 차이 (당일 - 주변 ±{window_days}일)",
        height=500,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    colored_count = (res['최저기온_차이'] < 0).sum()
    st.markdown(
        f"📌 분석된 {len(res)}회 중 **{colored_count}회({colored_count/len(res)*100:.1f}%)**가 "
        f"주변보다 추웠습니다. (순수 우연이라면 50%여야 합니다.)"
    )

# -------------------------------------------------------------
# 탭 2: 한파 기준 비교
# -------------------------------------------------------------
with tab2:
    st.subheader(f"수능일은 정말 '한파'였는가?")
    st.markdown(
        f"1994년 이후 모든 11월의 최저기온을 모아 분포를 보고, "
        f"**하위 {cold_percentile}% 임계값({cold_threshold:.2f}℃)**을 '한파' 기준선으로 잡았습니다."
    )

    # 히스토그램 + 수능일 점
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=nov_data['최저기온(℃)'],
        nbinsx=40,
        name='11월 전체 일별 최저기온',
        marker_color='lightblue',
        opacity=0.7,
    ))
    # 한파 임계선
    fig.add_vline(
        x=cold_threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"한파 기준: {cold_threshold:.2f}℃",
        annotation_position="top right",
    )
    # 수능일 점 (히스토그램 위에 rug처럼)
    fig.add_trace(go.Scatter(
        x=res['당일_최저기온'],
        y=[5] * len(res),  # 모두 같은 높이에
        mode='markers',
        marker=dict(
            size=12,
            color=['red' if t <= cold_threshold else 'orange' 
                   for t in res['당일_최저기온']],
            line=dict(color='black', width=1),
        ),
        text=[f"{y}학년도 수능 ({t:.1f}℃)" 
              for y, t in zip(res['연도'], res['당일_최저기온'])],
        hovertemplate='%{text}<extra></extra>',
        name='수능일',
    ))
    fig.update_layout(
        xaxis_title="최저기온 (℃)",
        yaxis_title="11월 일수",
        height=450,
        barmode='overlay',
    )
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("순수 우연이면 기대되는 한파 비율", f"{cold_percentile}%")
    with col_b:
        st.metric(
            "실제 수능일 한파 비율",
            f"{cold_rate:.1f}%",
            delta=f"{cold_rate - cold_percentile:+.1f}%p"
        )

    if cold_rate <= cold_percentile + 5:
        st.success(
            f"💡 수능일 한파 비율({cold_rate:.1f}%)이 우연 수준({cold_percentile}%)과 "
            f"거의 같습니다. **'수능날 유독 한파'라는 패턴은 보이지 않습니다.**"
        )
    else:
        st.warning(
            f"⚠️ 수능일 한파 비율({cold_rate:.1f}%)이 우연 수준({cold_percentile}%)보다 "
            f"높습니다. 추가 분석이 필요합니다."
        )

# -------------------------------------------------------------
# 탭 3: 통계 검정
# -------------------------------------------------------------
with tab3:
    st.subheader("정말 우연일까? — Paired t-test")
    st.markdown("""
    **t-검정**은 두 집단의 평균 차이가 우연 때문인지, 진짜 패턴인지 판단하는 방법입니다.
    여기서는 같은 해의 '수능 당일'과 '주변 평균'을 짝지어 비교하는 **paired t-test**를 씁니다.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("t-statistic", f"{t_stat:.3f}")
    with col2:
        st.metric("p-value", f"{p_value:.4f}")
    with col3:
        st.metric("유의수준 0.05 기준",
                  "✅ 유의" if p_value < 0.05 else "❌ 비유의")

    st.markdown("---")
    st.markdown("#### 📖 결과 해석 가이드")

    if p_value < 0.05:
        st.markdown(f"""
        **p-value = {p_value:.4f} < 0.05**
        
        → "수능 당일과 주변 일자의 기온 차이가 우연일 확률은 약 {p_value*100:.2f}%"
        라는 뜻입니다. 매우 낮으므로, **이 차이는 우연이 아닌 진짜 패턴일 가능성이
        높다**고 봅니다.
        """)
    else:
        st.markdown(f"""
        **p-value = {p_value:.4f} ≥ 0.05**
        
        → "수능 당일과 주변 일자의 기온 차이가 우연일 확률은 약 {p_value*100:.1f}%"
        라는 뜻입니다. 충분히 높으므로, **관찰된 차이를 우연이 아니라고 주장할
        근거가 부족합니다**.
        
        쉽게 말해: 수능날 기온은 그 주의 다른 날들과 통계적으로 구분되지 않습니다.
        '수능 한파'는 데이터로 뒷받침되지 않습니다.
        """)

    # 차이값 분포
    st.markdown("#### 수능일 - 주변 평균 (최저기온 차이) 분포")
    fig = px.histogram(
        res, x='최저기온_차이', nbins=15,
        labels={'최저기온_차이': '최저기온 차이 (℃)'},
    )
    fig.add_vline(x=0, line_dash="dash", line_color="gray",
                  annotation_text="차이 없음")
    fig.add_vline(x=res['최저기온_차이'].mean(), line_dash="dot",
                  line_color="red",
                  annotation_text=f"평균 {res['최저기온_차이'].mean():+.2f}℃")
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "💭 분포가 0을 중심으로 좌우 대칭에 가까울수록, "
        "수능날과 주변 날의 기온 차이는 '랜덤'이라고 볼 수 있습니다."
    )

# -------------------------------------------------------------
# 탭 4: 원본 데이터 탐색
# -------------------------------------------------------------
with tab4:
    st.subheader("수능일별 상세 데이터")
    display_df = res.copy()
    display_df['수능일'] = display_df['수능일'].dt.strftime('%Y-%m-%d')
    for c in display_df.select_dtypes(include=[np.number]).columns:
        display_df[c] = display_df[c].round(2)
    st.dataframe(display_df, use_container_width=True, height=400)

    csv = display_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        "📥 분석 결과 CSV 다운로드",
        data=csv,
        file_name='suneung_temperature_analysis.csv',
        mime='text/csv',
    )

    st.markdown("---")
    st.subheader("특정 연도 11월 기온 추이 보기")
    selected_year = st.selectbox(
        "연도 선택", sorted(res['연도'].unique(), reverse=True)
    )
    year_data = df[(df['연도'] == selected_year) & (df['월'].isin([11, 12]))]
    suneung_of_year = res[res['연도'] == selected_year]['수능일'].iloc[0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=year_data['날짜'], y=year_data['최고기온(℃)'],
        name='최고기온', line=dict(color='#e74c3c')))
    fig.add_trace(go.Scatter(
        x=year_data['날짜'], y=year_data['평균기온(℃)'],
        name='평균기온', line=dict(color='#2ecc71')))
    fig.add_trace(go.Scatter(
        x=year_data['날짜'], y=year_data['최저기온(℃)'],
        name='최저기온', line=dict(color='#3498db')))
    # plotly 6.x에서 datetime 축의 add_vline + annotation 조합은
    # 내부 평균 계산(sum 초깃값 int)이 Timestamp/문자열과 충돌해 TypeError 발생.
    # add_shape + add_annotation으로 분리해서 안전하게 그립니다.
    suneung_str = suneung_of_year.strftime('%Y-%m-%d')
    fig.add_shape(
        type="line",
        x0=suneung_str, x1=suneung_str,
        yref="paper", y0=0, y1=1,
        line=dict(color="purple", width=2, dash="dash"),
    )
    fig.add_annotation(
        x=suneung_str, yref="paper", y=1.02,
        text="🎓 수능일", showarrow=False,
        font=dict(color="purple", size=13),
    )
    fig.update_layout(
        title=f"{selected_year}년 11~12월 기온 추이",
        xaxis_title="날짜", yaxis_title="기온 (℃)", height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------
# 탭 5: 함께 생각해보기 (교육적 마무리)
# -------------------------------------------------------------
with tab5:
    st.subheader("🤔 데이터를 봤다면, 이제 생각할 차례")
    st.markdown("""
    데이터는 답을 주지 않습니다. **해석은 사람의 몫**이에요.
    아래 질문에 자기 생각을 정리해 보세요.
    
    ---
    
    #### 📝 탐구 질문
    
    **Q1.** 통계적으로 '수능 한파'가 입증되지 않는다면, 왜 사람들은 매년
    "수능날이 춥다"고 느낄까요? (힌트: **확증 편향**, **기저율 무시**)
    
    **Q2.** 옆 사이드바에서 비교 기간(±N일)을 1일, 7일, 14일로 바꿔보세요.
    결과가 어떻게 달라지나요? **분석 설계의 자유도**가 결론에 미치는 영향은?
    
    **Q3.** 한파 기준을 5%, 20%, 50%로 바꿔보세요. 어디서부터 '한파'라고
    부를 수 있을까요? **정의(definition)는 객관적인가요, 주관적인가요?**
    
    **Q4.** 이 분석의 한계는 무엇일까요?
    - 서울 한 곳의 데이터만 사용 (지역 차이 무시)
    - '한파'를 최저기온으로만 정의 (체감온도, 바람 미반영)
    - 1994년 이전은 수능이 없음 (대조군 한정)
    
    **Q5.** 만약 "수능 한파는 미신이다"라는 결론을 SNS에 올린다면, 어떤
    근거를 제시하고 어떤 한계를 함께 적어야 정직한 글일까요?
    
    ---
    
    #### 💡 더 해보기 (도전 과제)
    
    1. **쉬움**: 수능날이 가장 추웠던 해 Top 3, 가장 따뜻했던 해 Top 3 찾기
    2. **보통**: 1994년 이전 11월 셋째 주 목요일도 같은 방식으로 분석해
       "수능이 없던 시절"의 같은 날짜도 추웠는지 비교 (가짜 대조군 만들기)
    3. **어려움 ⭐**: 1907년부터 11월 기온의 **장기 추세**를 그려보고,
       '한파' 임계값이 시대별로 어떻게 달라지는지 분석"
    """)
    


# =============================================================
# 푸터
# =============================================================
st.markdown("---")
st.caption(
    "📡 데이터: 기상청 기상자료개방포털 (서울 관측소 108, 1907.10~2026.05) | "
    "🎓 만든 이: 석리송 (당곡고 수리정보교육부) | "
    "🤝 함께 만든 도구: Claude"
)
