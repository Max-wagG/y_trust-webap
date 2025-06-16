import streamlit as st
import pandas as pd
import requests
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- CONFIGURATION ---
API_BASE = "https://y-trust-003-51424904642.europe-west1.run.app"  # TODO: Replace with your actual deployed API URL

# --- PAGE SETTINGS ---
st.set_page_config(
    page_title="🌱 Y-TRUST Ingredient Analyzer",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS STYLING ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2e7d32;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #a8e6cf, #dcedc1);
        padding: 1rem;
        border-radius: 10px;
        color: #1b5e20;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 3px 6px rgba(0,0,0,0.1);
    }
    .info-box {
        background: #e0f2f1;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        border-radius: 5px;
        color: #004d40;
        margin-bottom: 1.5rem;
    }
    .header-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0 1rem;
        margin-bottom: 1rem;
    }
    .logo {
        height: 40px;
    }
    .top-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem;
        background-color: #f1f8e9;
        border-radius: 0 0 10px 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .menu-button {
        font-size: 1.5rem;
        font-weight: bold;
        cursor: pointer;
        color: #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER BAR ---
st.markdown("""
<div class="top-bar">
    <div class="menu-button">☰</div>
    <div><img src="logo.png" class="logo"></div>
</div>
""", unsafe_allow_html=True)

# --- MAIN HEADER ---
st.markdown('<div class="main-header">🌿 Y-TRUST Recipe Ingredient Analyzer</div>', unsafe_allow_html=True)

# --- FETCH RECIPE NAMES FOR DROPDOWN ---
try:
    recipe_names_response = requests.get(f"{API_BASE}/data/recipes")
    if recipe_names_response.status_code == 200:
        all_recipe_names = recipe_names_response.json().get("recipes", [])
    else:
        all_recipe_names = []
        st.error("Failed to fetch recipe names from API.")
except Exception as e:
    all_recipe_names = []
    st.error(f"Could not connect to API: {e}")

# --- TAB SELECTION ---
selected_tab = st.radio("Select a view:", ["🔎 Ingredient Matching", "📊 Nutrition Details"], horizontal=True, key="top_nav")

# --- LAYOUT SPLIT ---
col_input, col_output = st.columns([1, 2])

with col_input:
    st.subheader("📝 Recipe Selection")
    recipe_name = st.selectbox("Select a recipe:", options=all_recipe_names, placeholder="Type to search...")

with col_output:
    if recipe_name:
        with st.spinner("🔎 Contacting API and analyzing ingredients..."):
            try:
                response = requests.post(f"{API_BASE}/ingredients/predict", json={"recipe_name": recipe_name.strip().lower()})
                if response.status_code == 200:
                    data = response.json()
                    ingredients = data["ingredients"]
                    matches = pd.DataFrame(data["matches"])

                    if matches.empty:
                        st.warning("No matches found.")
                    else:
                        if selected_tab == "🔎 Ingredient Matching":
                            st.success(f"🌱 Found {len(matches)} matches for '{recipe_name}'")

                            # --- METRICS ---
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.markdown(f"""
                                    <div class="metric-card">
                                        <div style='font-size: 1.5rem;'>{len(ingredients)}</div>
                                        <div style='font-size: 0.9rem;'>Ingredients</div>
                                    </div>""", unsafe_allow_html=True)
                            with col2:
                                avg_score = matches['match_score'].mean()
                                st.markdown(f"""
                                    <div class="metric-card">
                                        <div style='font-size: 1.5rem;'>{avg_score:.1f}%</div>
                                        <div style='font-size: 0.9rem;'>Avg Match Score</div>
                                    </div>""", unsafe_allow_html=True)
                            with col3:
                                total_kcal = matches['energy-kcal_100g'].sum()
                                st.markdown(f"""
                                    <div class="metric-card">
                                        <div style='font-size: 1.5rem;'>{total_kcal:.0f}</div>
                                        <div style='font-size: 0.9rem;'>Total Calories</div>
                                    </div>""", unsafe_allow_html=True)

                            # --- DETAILED TABLE ---
                            st.subheader("📋 Matched Ingredients")
                            st.dataframe(matches[[
                                'searched_ingredient', 'matched_product', 'match_score',
                                'energy-kcal_100g', 'carbohydrates_100g', 'proteins_100g', 'fat_100g'
                            ]])

                        elif selected_tab == "📊 Nutrition Details":
                            st.subheader("📊 Nutritional Visuals")
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

                            hist_fig = px.histogram(matches, x="match_score", nbins=20, title="Match Score Distribution", color_discrete_sequence=['#81c784'])
                            st.plotly_chart(hist_fig, use_container_width=True)

                            # --- EXPORT ---
                            st.subheader("💾 Export Results")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.download_button("📄 Download CSV", matches.to_csv(index=False), file_name=f"{recipe_name}_matches.csv", mime="text/csv")
                            with col2:
                                st.download_button("📋 Download JSON", matches.to_json(orient="records", indent=2), file_name=f"{recipe_name}_matches.json", mime="application/json")

                else:
                    st.error(f"🚫 API error {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"🚨 Something went wrong: {e}")
    else:
        st.info("🌿 Enter a recipe name above to begin analysis.")
