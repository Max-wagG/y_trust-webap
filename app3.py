import streamlit as st
import pandas as pd
import requests
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sentence_transformers import SentenceTransformer, util
import torch

# --- CONFIGURATION ---
API_BASE = "https://y-trust-003-51424904642.europe-west1.run.app"

# --- PAGE SETTINGS ---
st.set_page_config(
    page_title="🌱 Y-TRUST Ingredient Analyzer",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SESSION STATE SETUP ---
if "show_menu" not in st.session_state:
    st.session_state.show_menu = False

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
    .top-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem;
        background-color: #f1f8e9;
        border-radius: 0 0 10px 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .logo {
        height: 40px;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER BAR ---
col_menu, col_logo = st.columns([1, 5])
with col_menu:
    if st.button("☰ Menu"):
        st.session_state.show_menu = not st.session_state.show_menu
with col_logo:
    st.markdown("""
    <div style='text-align: right;'>
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/React-icon.svg/1024px-React-icon.svg.png" class="logo">
    </div>
    """, unsafe_allow_html=True)

# --- COLLAPSIBLE MENU PANEL ---
if st.session_state.show_menu:
    with st.container():
        st.markdown("## 🧭 Navigation Menu")
        st.markdown("- Ingredient Matching")
        st.markdown("- Nutrition Details")
        st.markdown("- Export Options")

# --- MAIN HEADER ---
st.markdown('<div class="main-header">🌿 Y-TRUST Recipe Ingredient Analyzer</div>', unsafe_allow_html=True)

# --- LOAD RECIPES & EMBEDDINGS ---
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def load_recipes():
    df = pd.read_csv("data/recipes.csv")  # or hosted URL
    return df.dropna(subset=["name"])

df_recipes = load_recipes()
recipe_names = df_recipes["name"].tolist()
model = load_model()
recipe_embeddings = model.encode(recipe_names, convert_to_tensor=True)

# --- USER QUERY ---
st.subheader("🧠 Describe your craving")
user_query = st.text_input("Type what you're in the mood for:", placeholder="e.g. I want something sweet and easy to cook")

if user_query:
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, recipe_embeddings)[0]
    best_index = scores.argmax()
    matched_recipe = recipe_names[best_index]
    st.success(f"✅ Matched recipe: **{matched_recipe}**")

    # Call API with matched recipe
    with st.spinner("🔎 Contacting API and analyzing ingredients..."):
        try:
            response = requests.post(f"{API_BASE}/ingredients/predict", json={"recipe_name": matched_recipe.strip().lower()})
            if response.status_code == 200:
                data = response.json()
                ingredients = data["ingredients"]
                matches = pd.DataFrame(data["matches"])

                if matches.empty:
                    st.warning("No matches found.")
                else:
                    st.success(f"🌱 Found {len(matches)} matches for '{matched_recipe}'")

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

                    # --- TABS ---
                    selected_tab = st.radio("Select a view:", ["📋 Matched Ingredients", "📊 Nutrition Visuals"], horizontal=True)

                    if selected_tab == "📋 Matched Ingredients":
                        st.dataframe(matches[[
                            'searched_ingredient', 'matched_product', 'match_score',
                            'energy-kcal_100g', 'carbohydrates_100g', 'proteins_100g', 'fat_100g'
                        ]])

                    elif selected_tab == "📊 Nutrition Visuals":
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
                            st.download_button("📄 Download CSV", matches.to_csv(index=False), file_name=f"{matched_recipe}_matches.csv", mime="text/csv")
                        with col2:
                            st.download_button("📋 Download JSON", matches.to_json(orient="records", indent=2), file_name=f"{matched_recipe}_matches.json", mime="application/json")

            else:
                st.error(f"🚫 API error {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"🚨 Something went wrong: {e}")
else:
    st.info("🌿 Enter your craving to begin analysis.")
