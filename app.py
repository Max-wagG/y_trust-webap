import streamlit as st
import pandas as pd
import requests
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- CONFIGURATION ---
API_BASE = "https://y-trust-003-51424904642.europe-west1.run.app/docs"  # TODO: Replace with your actual deployed API URL

# --- PAGE SETTINGS ---
st.set_page_config(
    page_title="🍳 Recipe Ingredient Analyzer",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS STYLING ---
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #3a7d44;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #d0f0c0 0%, #9ccc65 100%);
        padding: 1rem;
        border-radius: 10px;
        color: #2e7d32;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown('<div class="main-header">🍳 Recipe Ingredient Analyzer</div>', unsafe_allow_html=True)

# --- RECIPE NAME INPUT ---
st.subheader("🔍 Select a Recipe")
recipe_name = st.text_input("Enter a recipe name:", placeholder="e.g. quick bolognese sauce")

if recipe_name:
    with st.spinner("Contacting API and analyzing ingredients..."):
        try:
            response = requests.post(f"{API_BASE}/ingredients/predict", json={"recipe_name": recipe_name.strip().lower()})
            if response.status_code == 200:
                data = response.json()
                ingredients = data["ingredients"]
                matches = pd.DataFrame(data["matches"])

                if matches.empty:
                    st.warning("No matches found.")
                else:
                    st.success(f"Found {len(matches)} matches for '{recipe_name}'")

                    # --- METRICS ---
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Ingredients", len(ingredients))
                    with col2:
                        st.metric("Avg Match Score", f"{matches['match_score'].mean():.1f}%")
                    with col3:
                        st.metric("Total Calories", f"{matches['energy-kcal_100g'].sum():.0f}")

                    # --- DETAILED TABLE ---
                    st.subheader("📋 Matched Ingredients")
                    st.dataframe(matches[[
                        'searched_ingredient', 'matched_product', 'match_score',
                        'energy-kcal_100g', 'carbohydrates_100g', 'proteins_100g', 'fat_100g'
                    ]])

                    # --- VISUALIZATIONS ---
                    st.subheader("📊 Nutritional Visuals")

                    # Radar Chart
                    avg_vals = {
                        'Calories': matches['energy-kcal_100g'].mean(),
                        'Carbs': matches['carbohydrates_100g'].mean(),
                        'Protein': matches['proteins_100g'].mean(),
                        'Fat': matches['fat_100g'].mean()
                    }

                    radar_fig = go.Figure()
                    radar_fig.add_trace(go.Scatterpolar(
                        r=list(avg_vals.values()),
                        theta=list(avg_vals.keys()),
                        fill='toself',
                        name='Average Nutrition',
                        line_color='#66bb6a'
                    ))
                    radar_fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=False)
                    st.plotly_chart(radar_fig, use_container_width=True)

                    # Histogram
                    hist_fig = px.histogram(matches, x="match_score", nbins=20, title="Match Score Distribution")
                    st.plotly_chart(hist_fig, use_container_width=True)

                    # --- EXPORT ---
                    st.subheader("💾 Export Results")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button("Download CSV", matches.to_csv(index=False), file_name=f"{recipe_name}_matches.csv", mime="text/csv")
                    with col2:
                        st.download_button("Download JSON", matches.to_json(orient="records", indent=2), file_name=f"{recipe_name}_matches.json", mime="application/json")

            else:
                st.error(f"API error {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"Something went wrong: {e}")
else:
    st.info("Enter a recipe name above to begin analysis.")
