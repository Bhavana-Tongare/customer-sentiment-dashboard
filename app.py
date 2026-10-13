"""
STEP 2: the one-page dashboard.   Run:  streamlit run app.py
Needs reviews_scored.csv created by analysis.py
"""
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer

st.set_page_config(page_title="Customer Sentiment Dashboard",
                   page_icon="🛒", layout="wide")

COLORS = {"positive": "#29BF12", "neutral": "#F3A712", "negative": "#E4572E"}
LABELS = ["negative", "neutral", "positive"]


@st.cache_data
def load_data():
    df = pd.read_csv("reviews_scored.csv")
    df["product_short"] = df["product_name"].str[:40]
    return df


def top_phrases(texts, n=15):
    if len(texts) < 5:
        return pd.DataFrame(columns=["phrase", "count"])
    try:
        vec = CountVectorizer(stop_words="english", ngram_range=(2, 2), min_df=2)
        X = vec.fit_transform(texts)
    except ValueError:
        return pd.DataFrame(columns=["phrase", "count"])
    return (pd.DataFrame({"phrase": vec.get_feature_names_out(), "count": X.sum(axis=0).A1})
            .sort_values("count", ascending=False).head(n))


df = load_data()

# ------------------------------------------------------------ SIDEBAR FILTERS
st.sidebar.header("Filters")
products = st.sidebar.multiselect("Product", sorted(df["product_short"].unique()))
rating_range = st.sidebar.slider("Star rating", 1, 5, (1, 5))
source = st.sidebar.radio("Sentiment based on",
                          ["model_sentiment", "rating_sentiment"],
                          format_func=lambda x: "AI model (DistilBERT)" if x == "model_sentiment"
                          else "Star rating")

f = df[df["rating"].between(*rating_range)]
if products:
    f = f[f["product_short"].isin(products)]
if f.empty:
    st.warning("No reviews match these filters.")
    st.stop()

# ------------------------------------------------------------ HEADER + KPIs
st.title("🛒 Customer Sentiment Dashboard")
st.caption("Flipkart product reviews analysed with a pretrained DistilBERT model")

share = f[source].value_counts(normalize=True) * 100
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total reviews", f"{len(f):,}")
k2.metric("Average rating", f"{f['rating'].mean():.2f} ⭐")
k3.metric("Positive", f"{share.get('positive', 0):.1f}%")
k4.metric("Neutral", f"{share.get('neutral', 0):.1f}%")
k5.metric("Negative", f"{share.get('negative', 0):.1f}%")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Overview", "📦 Products", "⚠️ Complaints", "🧪 Model quality", "💡 Insights"])

# ------------------------------------------------------------ TAB 1
with tab1:
    c1, c2 = st.columns(2)
    pie = px.pie(f, names=source, hole=0.5, color=source, color_discrete_map=COLORS,
                 title="Sentiment split")
    c1.plotly_chart(pie, width="stretch")

    rc = f["rating"].value_counts().sort_index().reset_index()
    rc.columns = ["rating", "reviews"]
    c2.plotly_chart(px.bar(rc, x="rating", y="reviews", text="reviews",
                           title="Rating distribution"), width="stretch")

# ------------------------------------------------------------ TAB 2
with tab2:
    min_rev = st.slider("Minimum reviews per product (avoids unfair small samples)", 5, 200, 30)
    g = (f.groupby("product_short")
           .agg(reviews=("rating", "size"), avg_rating=("rating", "mean"),
                negative_pct=(source, lambda s: (s == "negative").mean() * 100))
           .query("reviews >= @min_rev").sort_values("negative_pct", ascending=False).round(1))
    if g.empty:
        st.info("No product has that many reviews. Lower the slider.")
    else:
        top = g.head(10).iloc[::-1].reset_index()
        st.plotly_chart(px.bar(top, x="negative_pct", y="product_short", orientation="h",
                               color="negative_pct", color_continuous_scale="Reds",
                               title="Products with the highest % negative reviews"),
                        width="stretch")
        st.dataframe(g, width="stretch")

# ------------------------------------------------------------ TAB 3
with tab3:
    st.write("Most common 2-word phrases in **1–2 star** reviews:")
    tp = top_phrases(f.loc[f["rating"] <= 2, "summary"].fillna(""))
    if tp.empty:
        st.info("Not enough low-rated reviews with the current filters.")
    else:
        st.plotly_chart(px.bar(tp.iloc[::-1], x="count", y="phrase", orientation="h",
                               title="Top complaint phrases"), width="stretch")
    st.write("Sample of negative reviews")
    st.dataframe(f[f[source] == "negative"][["product_short", "rating", "review", "summary"]]
                 .head(50), width="stretch")

# ------------------------------------------------------------ TAB 4
with tab4:
    acc = (f["model_sentiment"] == f["rating_sentiment"]).mean() * 100
    st.metric("Model agrees with star rating", f"{acc:.1f}%")
    cm = (pd.crosstab(f["rating_sentiment"], f["model_sentiment"])
            .reindex(index=LABELS, columns=LABELS, fill_value=0))
    st.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                              labels=dict(x="Model predicted", y="Star rating says"),
                              title="Where the model and ratings agree/disagree"),
                    width="stretch")
    st.write("Reviews where the model **disagrees strongly** (5★ but model says negative, "
             "or 1★ but model says positive):")
    bad = f[((f["rating"] == 5) & (f["model_sentiment"] == "negative")) |
            ((f["rating"] == 1) & (f["model_sentiment"] == "positive"))]
    st.dataframe(bad[["rating", "review", "summary", "model_sentiment", "model_confidence"]]
                 .head(50), width="stretch")

# ------------------------------------------------------------ TAB 5
with tab5:
    st.subheader("Key findings")
    st.markdown("""
- **EDIT THIS** after you see your real numbers, e.g. "X% reviews are positive, Y% negative."
- Which 2–3 products have the highest negative %? Name them.
- Which complaint phrases appear most? (Check the Complaints tab.)
- **Recommendation:** fix the most frequent complaint first.
""")
    with st.expander("Limitations (always include these!)"):
        st.markdown("""
- Dataset has **no dates**, so trend over time could not be analysed.
- Star ratings are only a proxy for true sentiment (some 3★ reviews are really positive).
- Pretrained model was not tuned on Flipkart data; short reviews like "fair" can be misread.
- Complaints show correlation with low ratings, **not** that they cause customers to leave.
""")
