import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.graph_objects as go

st.set_page_config(page_title="영화 흥행 예측기", layout="wide")
st.title("🎬 영화 흥행 예측기")
st.caption("다중 회귀 모델로 영화의 총 관객 수를 예측해 봅니다.")

# ----------------------------------------------------
# 1. 데이터 불러오기
# ----------------------------------------------------
DAILY_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_daily.csv"
MOVIES_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_movies.csv"

@st.cache_data
def load_data():
    daily = pd.read_csv(DAILY_URL, encoding="utf-8")
    movies = pd.read_csv(MOVIES_URL, encoding="utf-8")
    return daily, movies

daily_df, movies_df = load_data()

# ----------------------------------------------------
# 2. 기준 기간 계산 (일별 표의 날짜 열에서 추출)
# ----------------------------------------------------
date_col = daily_df["날짜"].astype(str)
date_parsed = pd.to_datetime(date_col, format="%Y%m%d")
start_date = date_parsed.min().strftime("%Y-%m-%d")
end_date = date_parsed.max().strftime("%Y-%m-%d")

st.info(f"📅 기준 기간(박스오피스 일별 데이터): **{start_date} ~ {end_date}**")

# ----------------------------------------------------
# 3. 영화별 표 원본 보여주기 (맨 위 몇 줄)
# ----------------------------------------------------
st.subheader("📋 영화별 데이터 (원본 미리보기)")
st.dataframe(movies_df.head())

# ----------------------------------------------------
# ⚠️ 사후 집계 데이터임을 알리는 안내
# ----------------------------------------------------
st.warning(
    "⚠️ **주의**: 이 표에 있는 `first_scrn`, `first_show`, `first_week_audi` 등의 변수는 "
    "영화가 실제로 개봉되고 난 뒤에 집계된 값입니다. "
    "따라서 이 모델은 **개봉 전에 흥행을 미리 내다보는 예측이 아니라**, "
    "개봉 후 초반 성적을 바탕으로 최종 총 관객 수를 설명·추정하는 모델입니다. "
    "실제 '개봉 전 흥행 예측'에 활용하려면 개봉 전에 알 수 있는 정보(장르, 국가, 시즌 등)만 써야 합니다."
)

# ----------------------------------------------------
# 4. 영화코드 순 정렬 후 학습/테스트 분리
# ----------------------------------------------------
movies_sorted = movies_df.sort_values("movieCd").reset_index(drop=True)

def split_train_test(df, group_size=10, test_count=3):
    """group_size개 단위로 묶어서 앞의 test_count개는 테스트, 나머지는 학습으로 분리"""
    train_idx = []
    test_idx = []
    for start in range(0, len(df), group_size):
        group = list(range(start, min(start + group_size, len(df))))
        test_idx.extend(group[:test_count])
        train_idx.extend(group[test_count:])
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)

train_df, test_df = split_train_test(movies_sorted, group_size=10, test_count=3)

# ----------------------------------------------------
# 5. 사용할 변수(피처) 체크박스로 선택 (자유 실험용)
# ----------------------------------------------------
st.subheader("🔧 예측에 사용할 변수 선택 (직접 실험해 보기)")

candidate_features = {
    "first_scrn": "첫 관측일 스크린수",
    "first_show": "첫 관측일 상영횟수",
    "peak": "성수기 개봉 여부(1/0)",
    "first_week_audi": "첫 주 관객",
    "days_in_top10": "10위권 유지 일수",
}

cols = st.columns(len(candidate_features))
selected_features = []
for i, (col_name, label) in enumerate(candidate_features.items()):
    with cols[i]:
        checked = st.checkbox(label, value=True, key=col_name)
        if checked:
            selected_features.append(col_name)

if len(selected_features) == 0:
    st.warning("⚠️ 최소 하나 이상의 변수를 선택해야 합니다.")
    st.stop()

# ----------------------------------------------------
# 6. 모델 학습 및 평가를 위한 공통 함수
# ----------------------------------------------------
target = "total_audi"

def train_and_evaluate(features, train_df, test_df, target):
    """주어진 변수 목록으로 모델을 학습하고 평가 지표를 반환"""
    use_cols = features + [target]
    train_clean = train_df.dropna(subset=use_cols)
    test_clean = test_df.dropna(subset=use_cols)

    X_train = train_clean[features]
    y_train = train_clean[target]
    X_test = test_clean[features]
    y_test = test_clean[target]

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, a_min=0, a_max=None)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        "model": model,
        "train_clean": train_clean,
        "test_clean": test_clean,
        "y_test": y_test,
        "y_pred": y_pred,
        "mae": mae,
        "r2": r2,
    }

# ----------------------------------------------------
# 7. 두 가지 변수 조합 정의 및 학습
#    - 기본 3종: 첫 주 관객 수 제외
#    - 기본 3종 + 첫 주 관객 수
# ----------------------------------------------------
BASE_FEATURES = ["first_scrn", "first_show", "peak"]
EXTENDED_FEATURES = BASE_FEATURES + ["first_week_audi"]

result_base = train_and_evaluate(BASE_FEATURES, train_df, test_df, target)
result_extended = train_and_evaluate(EXTENDED_FEATURES, train_df, test_df, target)

# ----------------------------------------------------
# 8. 두 모델의 평가 결과를 나란히 비교
# ----------------------------------------------------
st.subheader("📊 변수 조합별 예측 성능 비교")

st.markdown(
    "아래는 **기본 변수 세 가지**(스크린수·상영횟수·성수기 여부)만 쓴 경우와, "
    "여기에 **첫 주 관객 수**까지 추가한 경우의 예측 성능을 비교한 것입니다."
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🅰️ 기본 변수 3종")
    st.caption(", ".join(BASE_FEATURES))
    st.metric("학습 영화 편수", f"{len(result_base['train_clean'])}편")
    st.metric("평가 영화 편수", f"{len(result_base['test_clean'])}편")
    st.metric("R² (결정계수)", f"{result_base['r2']:.3f}")
    st.metric("MAE (평균 절대 오차)", f"{result_base['mae']:,.0f}명")

with col2:
    st.markdown("### 🅱️ 기본 변수 3종 + 첫 주 관객 수")
    st.caption(", ".join(EXTENDED_FEATURES))
    st.metric("학습 영화 편수", f"{len(result_extended['train_clean'])}편")
    st.metric("평가 영화 편수", f"{len(result_extended['test_clean'])}편")
    st.metric("R² (결정계수)", f"{result_extended['r2']:.3f}")
    st.metric("MAE (평균 절대 오차)", f"{result_extended['mae']:,.0f}명")

r2_diff = result_extended["r2"] - result_base["r2"]
mae_diff = result_extended["mae"] - result_base["mae"]

st.info(
    f"첫 주 관객 수를 추가하면 R²는 **{r2_diff:+.3f}**, "
    f"MAE는 **{mae_diff:+,.0f}명** 만큼 변화했습니다. "
    f"(R²는 클수록, MAE는 작을수록 더 좋은 모델입니다.)"
)

st.caption(
    "기준 기간: " + f"{start_date} ~ {end_date}"
)

# ----------------------------------------------------
# 9. 내가 선택한 변수 조합으로 모델 학습 및 평가 (아래는 체크박스 선택 결과)
# ----------------------------------------------------
st.markdown("---")
st.subheader("🔍 내가 선택한 변수 조합으로 결과 보기")

result_selected = train_and_evaluate(selected_features, train_df, test_df, target)

col3, col4, col5 = st.columns(3)
col3.metric("학습에 사용한 영화 편수", f"{len(result_selected['train_clean'])}편")
col4.metric("평가에 사용한 영화 편수", f"{len(result_selected['test_clean'])}편")
col5.metric("기준 기간", f"{start_date} ~ {end_date}")

col6, col7 = st.columns(2)
col6.metric("R² (결정계수)", f"{result_selected['r2']:.3f}")
col7.metric("MAE (평균 절대 오차)", f"{result_selected['mae']:,.0f}명")

st.caption("MAE는 예측한 관객 수가 실제 관객 수와 평균적으로 얼마나 차이 나는지를 나타냅니다.")

# ----------------------------------------------------
# 10. 1,000명 미만 예측치 처리 (그래프 바닥에 붙이기 위한 준비)
# ----------------------------------------------------
FLOOR_VALUE = 1000

y_pred = result_selected["y_pred"]
y_test = result_selected["y_test"]
test_clean = result_selected["test_clean"]

low_pred_count = int(np.sum(y_pred < FLOOR_VALUE))
y_pred_display = np.where(y_pred < FLOOR_VALUE, FLOOR_VALUE, y_pred)

st.write(f"🔻 예측값이 {FLOOR_VALUE:,}명보다 작게 나온 영화: **{low_pred_count}편** (그래프 바닥에 표시됨)")

# ----------------------------------------------------
# 11. Plotly 산점도 (로그-로그 스케일, 대각선 포함)
# ----------------------------------------------------
st.subheader("📈 실제 vs 예측 총 관객 수 (테스트 영화, 선택한 변수 기준)")

x_vals = np.where(y_test <= 0, 1, y_test)
y_vals = np.where(y_pred_display <= 0, 1, y_pred_display)

movie_names = test_clean["movieNm"].values if "movieNm" in test_clean.columns else test_clean["movieCd"].values

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=x_vals,
    y=y_vals,
    mode="markers",
    name="테스트 영화",
    text=movie_names,
    hovertemplate="영화명: %{text}<br>실제 관객수: %{x:,.0f}<br>예측 관객수: %{y:,.0f}<extra></extra>",
    marker=dict(size=10, color="royalblue", opacity=0.7),
))

min_val = min(x_vals.min(), y_vals.min())
max_val = max(x_vals.max(), y_vals.max())
fig.add_trace(go.Scatter(
    x=[min_val, max_val],
    y=[min_val, max_val],
    mode="lines",
    name="실제=예측 (대각선)",
    line=dict(color="red", dash="dash"),
))

fig.update_layout(
    xaxis=dict(type="log", title="실제 총 관객 수 (로그 스케일)"),
    yaxis=dict(type="log", title="예측 총 관객 수 (로그 스케일)"),
    hovermode="closest",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=600,
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------
# 12. 테스트 결과 표로 보여주기
# ----------------------------------------------------
st.subheader("🎯 테스트 영화별 예측 결과 (선택한 변수 기준)")

result_df = test_clean[["movieCd", "movieNm"]].copy() if "movieNm" in test_clean.columns else test_clean[["movieCd"]].copy()
result_df["실제 총 관객 수"] = y_test.values
result_df["예측 총 관객 수"] = np.round(y_pred).astype(int)
result_df["오차 (예측-실제)"] = result_df["예측 총 관객 수"] - result_df["실제 총 관객 수"]

st.dataframe(result_df)
