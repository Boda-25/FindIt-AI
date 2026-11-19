
import os
import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import faiss

from PIL import Image
from transformers import CLIPModel, CLIPProcessor


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="FindIt AI | Visual Search",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.7rem;
        font-weight: 800;
        margin-bottom: 0;
    }
    .subtitle {
        color: #9CA3AF;
        font-size: 1.05rem;
        margin-top: 0.2rem;
        margin-bottom: 1.5rem;
    }
    .result-card {
        padding: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 12px;
        margin-bottom: 12px;
    }
    .score {
        font-weight: 700;
        color: #16A34A;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# PATHS
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "final_artifacts")

INDEX_PATH = os.path.join(
    ARTIFACTS_DIR, "findit_faiss.index"
)

METADATA_PATH = os.path.join(
    ARTIFACTS_DIR, "metadata.csv"
)

MODEL_PATH = os.path.join(
    ARTIFACTS_DIR, "clip_model"
)

PROCESSOR_PATH = os.path.join(
    ARTIFACTS_DIR, "clip_processor"
)


# ==========================================
# RESOURCE LOADING
# ==========================================

@st.cache_resource
def load_resources():

    required_paths = [
        INDEX_PATH,
        METADATA_PATH,
        MODEL_PATH,
        PROCESSOR_PATH
    ]

    missing = [
        path for path in required_paths
        if not os.path.exists(path)
    ]

    if missing:
        raise FileNotFoundError(
            "Missing project files:\n" + "\n".join(missing)
        )

    device = torch.device("cpu")

    model = CLIPModel.from_pretrained(
        MODEL_PATH,
        local_files_only=True
    )

    processor = CLIPProcessor.from_pretrained(
        PROCESSOR_PATH,
        local_files_only=True
    )

    model.to(device)
    model.eval()

    index = faiss.read_index(INDEX_PATH)

    metadata = pd.read_csv(
        METADATA_PATH,
        dtype={"id": str}
    )

    if len(metadata) != index.ntotal:
        raise ValueError(
            "Metadata rows do not match FAISS index vectors."
        )

    return model, processor, index, metadata, device


try:
    with st.spinner("Loading FindIt AI models and search index..."):
        model, processor, index, metadata, device = load_resources()

except Exception as error:
    st.error("Unable to load FindIt AI resources.")
    st.exception(error)
    st.stop()


# ==========================================
# IMAGE DIRECTORY
# ==========================================

default_image_dir = os.environ.get(
    "FINDIT_IMAGE_DIR",
    ""
)

st.sidebar.title("⚙️ Search Settings")

image_dir = st.sidebar.text_input(
    "Dataset Images Folder",
    value=default_image_dir,
    placeholder="Path to the folder containing product JPG images"
)

if not image_dir or not os.path.isdir(image_dir):
    st.sidebar.warning(
        "Set a valid dataset images folder to display product images."
    )


# ==========================================
# FILTER OPTIONS
# ==========================================

def get_filter_options(column):

    if column not in metadata.columns:
        return ["All"]

    values = (
        metadata[column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    return ["All"] + sorted(values)


st.sidebar.divider()
st.sidebar.subheader("Product Filters")

gender_filter = st.sidebar.selectbox(
    "Gender",
    get_filter_options("gender")
)

category_filter = st.sidebar.selectbox(
    "Category",
    get_filter_options("articleType")
)

color_filter = st.sidebar.selectbox(
    "Color",
    get_filter_options("baseColour")
)

top_k = st.sidebar.slider(
    "Number of Results",
    min_value=5,
    max_value=30,
    value=10
)


# ==========================================
# EMBEDDING FUNCTIONS
# ==========================================

def encode_text(text):

    inputs = processor(
        text=[text],
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            return_dict=True
        )

        features = model.text_projection(
            outputs.pooler_output
        )

        features = F.normalize(
            features,
            p=2,
            dim=-1
        )

    return features.cpu().numpy().astype(np.float32)


def encode_image(image):

    inputs = processor(
        images=image.convert("RGB"),
        return_tensors="pt"
    )

    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():

        outputs = model.vision_model(
            pixel_values=pixel_values,
            return_dict=True
        )

        features = model.visual_projection(
            outputs.pooler_output
        )

        features = F.normalize(
            features,
            p=2,
            dim=-1
        )

    return features.cpu().numpy().astype(np.float32)


# ==========================================
# SEARCH + FILTERING
# ==========================================

def search_products(query_embedding, requested_k):

    # Search the complete index so filters are not limited
    # to an arbitrary small candidate pool.
    scores, indices = index.search(
        query_embedding,
        index.ntotal
    )

    valid = indices[0] >= 0

    retrieved_ids = indices[0][valid].astype(int)
    retrieved_scores = scores[0][valid]

    results = metadata.iloc[retrieved_ids].copy()

    results["similarity_score"] = retrieved_scores
    results["faiss_id"] = retrieved_ids

    if gender_filter != "All" and "gender" in results.columns:
        results = results[
            results["gender"].astype(str) == gender_filter
        ]

    if category_filter != "All" and "articleType" in results.columns:
        results = results[
            results["articleType"].astype(str) == category_filter
        ]

    if color_filter != "All" and "baseColour" in results.columns:
        results = results[
            results["baseColour"].astype(str) == color_filter
        ]

    results = results.sort_values(
        "similarity_score",
        ascending=False
    )

    return results.head(requested_k).reset_index(drop=True)


# ==========================================
# PRODUCT IMAGE PATH
# ==========================================

def get_product_image_path(product_id):

    if not image_dir:
        return None

    product_id = str(product_id).strip()

    # Handle numeric IDs that may have a trailing .0
    if product_id.endswith(".0"):
        product_id = product_id[:-2]

    return os.path.join(
        image_dir,
        f"{product_id}.jpg"
    )


# ==========================================
# DISPLAY RESULTS
# ==========================================

def display_results(results):

    if results.empty:
        st.warning(
            "No products matched the current search and filters."
        )
        return

    st.success(f"{len(results)} products found")

    columns_per_row = 5

    for start in range(0, len(results), columns_per_row):

        row = results.iloc[
            start:start + columns_per_row
        ]

        columns = st.columns(columns_per_row)

        for column, (_, product) in zip(columns, row.iterrows()):

            with column:

                image_path = get_product_image_path(
                    product["id"]
                )

                if image_path and os.path.isfile(image_path):
                    st.image(
                        image_path,
                        use_container_width=True
                    )
                else:
                    st.info("Product image unavailable")

                product_name = product.get(
                    "productDisplayName",
                    "Fashion Product"
                )

                st.markdown(
                    f"**{product_name}**"
                )

                score = float(
                    product["similarity_score"]
                )

                st.markdown(
                    f'<p class="score">Similarity: {score:.3f}</p>',
                    unsafe_allow_html=True
                )

                article_type = product.get("articleType", "")
                color = product.get("baseColour", "")
                gender = product.get("gender", "")

                st.caption(
                    f"{article_type} · {color} · {gender}"
                )


# ==========================================
# HEADER
# ==========================================

st.markdown(
    '<p class="main-title">🔎 FindIt AI</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">'
    'AI-Powered Fashion Visual Search Engine · Powered by CLIP + FAISS'
    '</p>',
    unsafe_allow_html=True
)

metric1, metric2, metric3 = st.columns(3)

metric1.metric(
    "Fashion Products",
    f"{len(metadata):,}"
)

metric2.metric(
    "Indexed Vectors",
    f"{index.ntotal:,}"
)

metric3.metric(
    "Embedding Dimension",
    str(index.d)
)

st.divider()


# ==========================================
# SEARCH INTERFACE
# ==========================================

search_mode = st.radio(
    "Choose your search method",
    ["📝 Search by Text", "🖼️ Search by Image"],
    horizontal=True
)


if search_mode == "📝 Search by Text":

    st.subheader("Describe the product you're looking for")

    with st.form("text_search_form"):

        query = st.text_input(
            "Product description",
            placeholder="Example: black leather jacket"
        )

        submitted = st.form_submit_button(
            "🔍 Search Products",
            use_container_width=True
        )

    if submitted:

        if not query.strip():
            st.warning("Please enter a product description.")

        else:

            with st.spinner("Finding visually relevant products..."):

                query_embedding = encode_text(query)

                results = search_products(
                    query_embedding,
                    top_k
                )

            st.subheader("Search Results")
            display_results(results)


else:

    st.subheader("Upload a product image")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is not None:

        try:
            query_image = Image.open(
                uploaded_file
            ).convert("RGB")

            st.image(
                query_image,
                caption="Your query image",
                width=250
            )

            if st.button(
                "🔍 Find Similar Products",
                use_container_width=True
            ):

                with st.spinner("Searching for similar products..."):

                    query_embedding = encode_image(query_image)

                    results = search_products(
                        query_embedding,
                        top_k
                    )

                st.subheader("Similar Products")
                display_results(results)

        except Exception as error:
            st.error("Could not process the uploaded image.")
            st.exception(error)


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "FindIt AI | CLIP Image Embeddings | FAISS Similarity Search"
)
