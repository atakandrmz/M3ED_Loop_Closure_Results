import streamlit as st
import scipy.io
import plotly.graph_objects as pl
import numpy as np
from PIL import Image
import os
from streamlit_image_comparison import image_comparison

st.set_page_config(page_title="Loop Closure Analysis Tool", layout="wide")

# ========================================================================
# Dataset Configuration (Hugging Face Auto-Download)
# ========================================================================
BASE_DIR = os.path.dirname(__file__)
ACCELERATED_FEATURES_DIR = os.path.join(BASE_DIR, "accelerated_features")
NETVLAD_DIR = BASE_DIR

target_folder = os.path.join(NETVLAD_DIR, "spot_forest_hard_data_images_rgb")

def is_dataset_ready(folder):
    if not os.path.exists(folder):
        return False
    try:
        # Check if we have at least 1000 images to confirm it's not a broken/empty folder
        return len(os.listdir(folder)) > 1000
    except:
        return False

if not is_dataset_ready(target_folder):
    with st.spinner("📦 İlk açılış: Hugging Face Dataset'inden resimler indiriliyor... (Bu işlem yalnızca 1 kez yapılacak ve 1-2 dakika sürebilir)"):
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="Umutsoo/SimularityGui-Data",
                repo_type="dataset",
                local_dir=NETVLAD_DIR,
                resume_download=True
            )
        except Exception as e:
            st.error(f"Failed to download dataset: {e}")

sequences = [
    {
        "name": "spot_forest_hard_data_images_rgb",
        "resultDir": os.path.join(ACCELERATED_FEATURES_DIR, "realtime_results_v2_spot_forest_hard_data_images_rgb"),
        "imageDir": os.path.join(NETVLAD_DIR, "spot_forest_hard_data_images_rgb")
    },
    {
        "name": "spot_indoor_building_loop_data_images_rgb",
        "resultDir": os.path.join(ACCELERATED_FEATURES_DIR, "realtime_results_v2_spot_indoor_building_loop_data_images_rgb"),
        "imageDir": os.path.join(NETVLAD_DIR, "spot_indoor_building_loop_data_images_rgb")
    },
    {
        "name": "spot_indoor_obstacles_data_images_rgb",
        "resultDir": os.path.join(ACCELERATED_FEATURES_DIR, "realtime_results_v2_spot_indoor_obstacles_data_images_rgb"),
        "imageDir": os.path.join(NETVLAD_DIR, "spot_indoor_obstacles_data_images_rgb")
    },
    {
        "name": "spot_outdoor_day_skatepark_1_data_images_rgb",
        "resultDir": os.path.join(ACCELERATED_FEATURES_DIR, "realtime_results_v2_spot_outdoor_day_skatepark_1_data_images_rgb"),
        "imageDir": os.path.join(NETVLAD_DIR, "spot_outdoor_day_skatepark_1_data_images_rgb")
    },
    {
        "name": "spot_outdoor_day_skatepark_2_data_images_rgb",
        "resultDir": os.path.join(ACCELERATED_FEATURES_DIR, "realtime_results_v2_spot_outdoor_day_skatepark_2_data_images_rgb"),
        "imageDir": os.path.join(NETVLAD_DIR, "spot_outdoor_day_skatepark_2_data_images_rgb")
    }
]

# Create a mapping for easy lookup
seq_dict = {seq["name"]: seq for seq in sequences}

# ========================================================================
# Helper Functions
# ========================================================================

@st.cache_data(show_spinner=False)
def load_matrices(result_dir):
    sim_file = os.path.join(result_dir, "frame_similarity.mat")
    energy_file = os.path.join(result_dir, "energy.mat")
    
    S = None
    E = None
    
    if os.path.exists(sim_file):
        tmp_s = scipy.io.loadmat(sim_file)
        keys = [k for k in tmp_s.keys() if not k.startswith('__')]
        if keys:
            S = tmp_s[keys[0]]
            
    if os.path.exists(energy_file):
        tmp_e = scipy.io.loadmat(energy_file)
        keys = [k for k in tmp_e.keys() if not k.startswith('__')]
        if keys:
            E = tmp_e[keys[0]]
            
    return S, E

def get_image(image_dir, frame_id):
    img_path = os.path.join(image_dir, f"image_{frame_id:05d}.jpg")
    if os.path.exists(img_path):
        return Image.open(img_path)
    return None

# ========================================================================
# Main App
# ========================================================================

st.title("Loop Closure Analysis Tool")
st.markdown("Select a sequence below, configure frames via sliders, or click on the **Similarity Matrix** to explore loop closures.")

# Dropdown
selected_seq_name = st.selectbox("Select a sequence", list(seq_dict.keys()))
selected_seq = seq_dict[selected_seq_name]

# Load Data
with st.spinner("Loading matrices..."):
    S, E = load_matrices(selected_seq["resultDir"])

if S is None or E is None:
    st.error(f"Could not load matrices from {selected_seq['resultDir']}.")
    st.stop()

# ========================================================================
# State Management & Callbacks (Fix for StreamlitAPIException)
# ========================================================================
if "master_row" not in st.session_state:
    st.session_state.master_row = 0
if "master_col" not in st.session_state:
    st.session_state.master_col = 0
if "last_sim_sel" not in st.session_state:
    st.session_state.last_sim_sel = None
if "current_seq" not in st.session_state:
    st.session_state.current_seq = selected_seq_name

# Reset values if sequence changed
if st.session_state.current_seq != selected_seq_name:
    st.session_state.current_seq = selected_seq_name
    st.session_state.master_row = 0
    st.session_state.master_col = 0
    st.session_state.last_sim_sel = None

# Process Matrix Clicks from the PREVIOUS run BEFORE UI is rendered
if "sim_matrix" in st.session_state:
    current_sel = st.session_state.sim_matrix.get("selection", {})
    # Check if the click selection actually changed
    if current_sel != st.session_state.last_sim_sel:
        st.session_state.last_sim_sel = current_sel
        if current_sel and "points" in current_sel and len(current_sel["points"]) > 0:
            pt = current_sel["points"][0]
            st.session_state.master_col = int(pt["x"])
            st.session_state.master_row = int(pt["y"])

# Enforce bounds
max_row = S.shape[0] - 1
max_col = S.shape[1] - 1
st.session_state.master_row = max(0, min(st.session_state.master_row, max_row))
st.session_state.master_col = max(0, min(st.session_state.master_col, max_col))

# Callbacks to sync manual inputs and sliders with the master state
def sync_row_num(): st.session_state.master_row = st.session_state.row_num
def sync_row_sld(): st.session_state.master_row = st.session_state.row_sld
def sync_col_num(): st.session_state.master_col = st.session_state.col_num
def sync_col_sld(): st.session_state.master_col = st.session_state.col_sld

# ========================================================================
# 1. Sliders & Frame Selection
# ========================================================================
st.subheader("Frame Selection")
sel_col1, sel_col2 = st.columns(2)

with sel_col1:
    st.number_input("Row (Manuel Giriş)", min_value=0, max_value=max_row, value=st.session_state.master_row, key="row_num", on_change=sync_row_num)
    st.slider("Row (Kaydırıcı)", min_value=0, max_value=max_row, value=st.session_state.master_row, key="row_sld", on_change=sync_row_sld, label_visibility="collapsed")
with sel_col2:
    st.number_input("Column (Manuel Giriş)", min_value=0, max_value=max_col, value=st.session_state.master_col, key="col_num", on_change=sync_col_num)
    st.slider("Column (Kaydırıcı)", min_value=0, max_value=max_col, value=st.session_state.master_col, key="col_sld", on_change=sync_col_sld, label_visibility="collapsed")

# Pull values to use below
display_row = st.session_state.master_row
display_col = st.session_state.master_col

# Calculate associated values
sim_val = S[display_row, display_col]
energy_val = E[display_row, display_col]
frame_row = display_row * 12
frame_col = display_col * 12

img_row_path = os.path.join(selected_seq["imageDir"], f"image_{frame_row:05d}.jpg")
img_col_path = os.path.join(selected_seq["imageDir"], f"image_{frame_col:05d}.jpg")

st.info(f"**Row:** {display_row} | **Col:** {display_col} | **Similarity:** {sim_val:.1f} | **Energy:** {energy_val:.1f} | **FrameA:** {frame_row} | **FrameB:** {frame_col}")


# ========================================================================
# 2. Matrices & Graphs
# ========================================================================
st.divider()
col1, col2 = st.columns(2)

with col1:
    st.subheader("Frame Similarity Matrix")
    
    # Plotly Heatmap
    fig_sim = pl.Figure(data=pl.Heatmap(
        z=S,
        colorscale='Viridis',
        hoverongaps=False
    ))
    
    # Update layout to match MATLAB's `axis image`
    fig_sim.update_layout(
        xaxis=dict(scaleanchor="y", constrain="domain"),
        yaxis=dict(autorange="reversed"), # MATLAB imagesc reverses Y axis
        margin=dict(l=0, r=0, t=30, b=0),
        height=500
    )
    
    # Add an invisible scatter plot on top of the Heatmap to capture click events in Streamlit natively
    X, Y = np.meshgrid(np.arange(S.shape[1]), np.arange(S.shape[0]))
    fig_sim.add_trace(pl.Scattergl(
        x=X.flatten(),
        y=Y.flatten(),
        mode='markers',
        marker=dict(size=12, color='rgba(0,0,0,0)', symbol='square'),
        hoverinfo='none',
        showlegend=False
    ))
    
    # Render with Streamlit's native on_select and a unique key
    st.plotly_chart(fig_sim, width="stretch", on_select="rerun", selection_mode="points", key="sim_matrix")

with col2:
    st.subheader("Energy Matrix")
    
    # Downsample to avoid browser crash
    ds = max(1, E.shape[0] // 50)
    
    X_e, Y_e = np.meshgrid(np.arange(E.shape[1]), np.arange(E.shape[0]))
    X_e_ds = X_e[::ds, ::ds]
    Y_e_ds = Y_e[::ds, ::ds]
    E_ds = E[::ds, ::ds]
    
    fig_energy = pl.Figure(data=[pl.Surface(
        x=X_e_ds, 
        y=Y_e_ds, 
        z=E_ds, 
        colorscale='Viridis'
    )])
    
    if display_col is not None and display_row is not None:
        try:
            energy_val_plot = float(E[display_row, display_col])
            fig_energy.add_trace(pl.Scatter3d(
                x=[display_col],
                y=[display_row],
                z=[energy_val_plot],
                mode='markers',
                marker=dict(size=8, color='red', symbol='circle'),
                name='Selected'
            ))
        except IndexError:
            pass

    fig_energy.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=500,
        uirevision="constant_3d_view" # Prevent camera reset
    )
    
    st.plotly_chart(fig_energy, width="stretch")


# ========================================================================
# 3. Image Comparison Display
# ========================================================================
st.divider()

try:
    if os.path.exists(img_row_path) and os.path.exists(img_col_path):
        
        st.subheader("Frame Comparison")
        # Ensure image comparison fits layout
        image_comparison(
            img1=img_row_path,
            img2=img_col_path,
            label1=f"Row Frame {frame_row}",
            label2=f"Col Frame {frame_col}",
            width=800
        )
        
        st.subheader("Individual Frames")
        ind_col1, ind_col2 = st.columns(2)
        with ind_col1:
            st.image(img_row_path, caption=f"Row Frame {frame_row}")
        with ind_col2:
            st.image(img_col_path, caption=f"Col Frame {frame_col}")
    else:
        if not os.path.exists(img_row_path):
            st.warning(f"Row Frame {frame_row} not found. (Expected at: {img_row_path})")
        if not os.path.exists(img_col_path):
            st.warning(f"Column Frame {frame_col} not found. (Expected at: {img_col_path})")
            
except Exception as e:
    st.error(f"Error loading images: {e}")