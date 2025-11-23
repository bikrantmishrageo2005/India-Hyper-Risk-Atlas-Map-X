import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="India HyperRisk Atlas 10X", layout="wide")


@st.cache_data
def load_hazard_data():
    df = pd.read_csv("india_state_multi_hazard.csv")
    hazard_cols = [
        "eq_risk",
        "flood_risk",
        "cyclone_risk",
        "tsunami_risk",
        "landslide_risk",
        "heatwave_risk",
        "drought_risk",
        "forestfire_risk",
        "airpollution_risk",
        "lightning_risk",
    ]
    df["multi_hazard_score"] = df[hazard_cols].mean(axis=1)
    return df, hazard_cols


@st.cache_data
def load_geojson():
    with open("india_states.geojson", "r", encoding="utf-8") as f:
        geo = json.load(f)
    return geo


@st.cache_resource
def train_vulnerability_model(df: pd.DataFrame, hazard_cols):
    overall = df[hazard_cols].mean(axis=1)
    labels = []
    for v in overall:
        if v < 35:
            labels.append("Low")
        elif v < 70:
            labels.append("Moderate")
        else:
            labels.append("High")
    df = df.copy()
    df["vulnerability_label"] = labels
    X = df[hazard_cols].values
    y = df["vulnerability_label"].values
    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X, y)
    return model


def plot_bar(labels, scores):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, scores)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Hazard index (0–100)")
    ax.set_xticklabels(labels, rotation=45, ha="right")
    fig.tight_layout()
    return fig


def plot_radar(labels, scores):
    import math

    n = len(labels)
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
    scores = np.array(scores)
    scores = np.concatenate([scores, [scores[0]]])
    angles = np.concatenate([angles, [angles[0]]])

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, scores, linewidth=2)
    ax.fill(angles, scores, alpha=0.25)
    ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    return fig


def main():
    df, hazard_cols = load_hazard_data()
    geo = load_geojson()

    st.title("India HyperRisk Atlas 10X")
    st.write(
        "State-level multi-hazard dashboard combining ten normalized hazard indices "
        "into a single multi-hazard profile for each Indian state."
    )

    df = df.rename(columns={"state": "STATE"})
    states = sorted(df["STATE"].unique())

    col_sidebar, col_main = st.columns([1, 4])

    with col_sidebar:
        selected_state = st.selectbox("Select state", states)
        show_map_metric = st.checkbox("Show multi-hazard map", value=True)

    with col_main:
        st.subheader("Multi-hazard choropleth map")
        if show_map_metric:
            fig_map = px.choropleth(
                df,
                geojson=geo,
                featureidkey="properties.ST_NM",
                locations="STATE",
                color="multi_hazard_score",
                color_continuous_scale="YlOrRd",
                range_color=(0, 100),
                labels={"multi_hazard_score": "Hazard index"},
            )
            fig_map.update_geos(fitbounds="locations", visible=False)
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("Enable the map from the left panel to view state-wise multi-hazard scores.")

    model = train_vulnerability_model(df, hazard_cols)

    hazard_labels = [
        "Earthquake",
        "Flood",
        "Cyclone",
        "Tsunami",
        "Landslide",
        "Heatwave",
        "Drought",
        "Forest Fire",
        "Air Pollution",
        "Lightning",
    ]

    row = df[df["STATE"] == selected_state].iloc[0]
    scores = [row[c] for c in hazard_cols]
    overall_index = float(np.mean(scores))
    input_array = np.array(scores).reshape(1, -1)
    predicted_label = model.predict(input_array)[0]

    st.subheader(f"Hazard profile: {selected_state}")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.write("Individual hazard indices")
        fig_bar = plot_bar(hazard_labels, scores)
        st.pyplot(fig_bar)

    with col2:
        st.metric("Multi-hazard index", f"{overall_index:.1f} / 100")
        st.write("Estimated vulnerability category:", predicted_label)
        sorted_pairs = sorted(
            zip(hazard_labels, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        st.write("Top contributing hazards:")
        for name, val in sorted_pairs[:3]:
            st.write(f"- {name}: {val:.1f}")

    st.subheader("Hazard radar chart")
    fig_radar = plot_radar(hazard_labels, scores)
    st.pyplot(fig_radar)

    st.subheader("State-wise comparison by multi-hazard index")
    df["overall_index"] = df[hazard_cols].mean(axis=1)
    df_sorted = df[["STATE", "overall_index"]].sort_values("overall_index", ascending=False)
    st.dataframe(df_sorted.rename(columns={"STATE": "State", "overall_index": "Multi-hazard index"}))


if __name__ == "__main__":
    main()
