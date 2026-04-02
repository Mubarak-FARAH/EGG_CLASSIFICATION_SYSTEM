from __future__ import annotations

import base64
import re
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import timm
from PIL import Image, ImageDraw
from torchvision import transforms

from rag_utils import answer_book_question, build_book_index

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
except Exception:
    GradCAM = None
    show_cam_on_image = None

import json

# Load egg attribute dataset once at startup
_EGG_DATASET_PATH = Path(__file__).resolve().parent / "egg_dataset.json"
try:
    with open(_EGG_DATASET_PATH, "r") as _f:
        EGG_DATA: List[Dict] = json.load(_f)
except Exception:
    EGG_DATA = []




# =============================================================================
# app config
# =============================================================================

st.set_page_config(
    page_title="Eggcellent ID",
    page_icon=None,
    layout="centered",
    initial_sidebar_state="collapsed",
)

RAM_LOGO_PATH = "Branding/Royal_Alberta_Museum_idvrGqHZJ__1.png"

try:
    st.logo(
        image=RAM_LOGO_PATH,
        icon_image=RAM_LOGO_PATH,
        size="small",
    )
except Exception:
    pass


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
HIGH_CONFIDENCE_THRESHOLD = 0.75
TOP_K = 3
APP_VERSION = "1.0.6"
MODEL_FILENAME = "best_model.pth"
MODEL_SCRIPT_FILENAME = "train_evaluate.py"
SPECIES_FILENAME = "species_info.xlsx"
APP_ARCH_DIRNAME = "APP_ARCHITECTURE"
BIRD_IMAGES_DIRNAME = "Birds_images"
MODEL_DIRNAME = "MODEL"

HOME_DISCLAIMER = (
    "Eggcellent ID provides AI-assisted species identification based on visual morphological "
    "characteristics. Results are probabilistic and expressed as confidence coefficients. They "
    "are intended to support, not replace, expert taxonomic judgment. The ultimate determination "
    "of species remains the exclusive responsibility of the qualified professional.\n\n"
    "The accuracy of the system may be influenced by factors such as image quality, lighting "
    "conditions, specimen deterioration, or morphological similarities between species.\n\n"
    "The developers assume no liability for misidentification or decisions made based on the app's "
    "output. Please use the application in accordance with your institution's scientific standards "
    "and protocols."
)

EXPLAINABILITY_TEXT_1 = "Highlighted areas show the parts of the egg image the model relied on most."
EXPLAINABILITY_TEXT_2 = "This is a visual explanation, not a guarantee of correctness."

ABOUT_CONTENT = {
    "project_description": (
        "Eggcellent ID is a mobile-first prototype that supports bird egg identification using "
        "an EfficientNet-B0 image classification model and Grad-CAM visual explanation."
    ),
    "app_version": APP_VERSION,
    "model_version": "EfficientNet-B0 classification model, 21 classes, input size 640x640.",
    "dataset_version": "Model-ready dataset used during training and validation.",
    "acknowledgements": (
        "Royal Alberta Museum, project collaborators, and supporting data and domain contributors."
    ),
}

ERROR_MESSAGES = {
    "camera": {
        "title": "Camera issue",
        "message": "The camera could not capture a usable image.",
        "actions": [
            "try again",
            "check camera permissions",
            "make sure one egg fills the frame",
        ],
    },
    "model": {
        "title": "Model issue",
        "message": "The prediction model could not be loaded or used.",
        "actions": [
            "confirm the model file exists",
            "confirm the file paths are correct",
            "restart the app after checking dependencies",
        ],
    },
    "results": {
        "title": "Results issue",
        "message": "A confident result could not be generated from this image.",
        "actions": [
            "retake the image with better lighting",
            "make sure only one egg is visible",
            "use manual input for additional support",
        ],
    },
    "device": {
        "title": "Device issue",
        "message": "This device may not support the current capture or display step well.",
        "actions": [
            "try another browser",
            "rotate the device and retry",
            "use another device if needed",
        ],
    },
}

MANUAL_INPUT_OPTIONS = {
    "color": [
        "White",
        "Cream",
        "Blue",
        "Green",
        "Gray",
    ],
    "pattern": [
        "None",
        "Spotted",
        "Speckled",
        "Blotched",
        "Streaked",
        "Lined",
        "Marbled",
    ],
    "marking_color": [
        "Brown",
        "Black",
        "Red",
        "Purple",
    ],
    "marking_intensity": [
        "None",
        "Light",
        "Moderate",
        "Heavy",
    ],
    "size": [
        "Very Small",
        "Small",
        "Medium",
        "Large",
    ],
}

BOOKS_DIRNAME = "books"
BOOK_FILENAME = "nests_and_eggs_of_north_american_birds.pdf"
BOOK_TOP_K = 5


# =============================================================================
# styles
# =============================================================================

def inject_css() -> None:
    st.markdown(
        """
        <style>

            /* global layout */
            .block-container {
                max-width: 720px;
                padding-top: 2.5rem !important;
                padding-bottom: 1.25rem !important;
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }

            div[data-testid="stVerticalBlock"] {
                gap: 0.8rem !important;
            }

            /* typography spacing */
            h1, h2, h3 {
                letter-spacing: -0.02em;
                margin-bottom: 0.5rem !important;
            }

            p {
                margin-bottom: 0.75rem !important;
                line-height: 1.5 !important;
            }

            /* cards */
            .content-card, .result-card, .error-card {
                border: 1px solid rgba(49, 51, 63, 0.10);
                border-radius: 18px;
                padding: 1rem 1rem 1.1rem 1rem;
                background: rgba(255,255,255,0.88);
                margin-bottom: 0.85rem;
            }

            /* secondary text */
            .small-muted {
                color: rgba(49, 51, 63, 0.72);
                font-size: 0.93rem;
                line-height: 1.45;
            }

            .frame-note {
                border-left: 4px solid #4c6fff;
                padding-left: 0.85rem;
                margin: 0.75rem 0;
                color: rgba(49, 51, 63, 0.84);
                font-size: 0.95rem;
            }

            /* result elements */
            .confidence-badge {
                display: inline-block;
                padding: 0.3rem 0.65rem;
                border-radius: 999px;
                border: 1px solid rgba(49, 51, 63, 0.14);
                font-size: 0.9rem;
                margin-bottom: 0.75rem;
            }

            .species-name {
                font-size: 1.55rem;
                font-weight: 700;
                margin-bottom: 0.25rem;
            }

            .section-label {
                font-size: 0.8rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                color: rgba(49, 51, 63, 0.62);
                margin-bottom: 0.35rem;
            }

            /* home screen */
            .home-subtitle {
                text-align: center;
                color: rgba(49, 51, 63, 0.68);
                font-size: 1rem;
                margin-top: 0.45rem;
                margin-bottom: 1rem;
            }

            .home-icon-wrap {
                width: 58px;
                height: 58px;
                border-radius: 999px;
                background: #f2f6f1;
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 0 auto 0.9rem auto;
                color: #94b39a;
                overflow: hidden;
            }

            .home-icon-wrap span {
                font-family: "Material Symbols Rounded";
                font-size: 30px;
                line-height: 1;
                font-weight: 400;
            }

            .home-card-title {
                text-align: center;
                font-size: 1.2rem;
                font-weight: 700;
                margin-bottom: 0.45rem;
            }

            .home-card-text {
                text-align: center;
                font-size: 0.95rem;
                color: rgba(49, 51, 63, 0.70);
                line-height: 1.45;
                margin-bottom: 1rem;
            }

            .home-card-wrap {
                padding-top: 0.4rem;
                padding-bottom: 0.25rem;
            }

            /* buttons */
            div[data-testid="stButton"] {
                display: flex !important;
                justify-content: flex-start !important;
                align-items: center !important;
            }

            div[data-testid="stButton"] > button {
                display: inline-flex !important;
                align-items: center !important;
                justify-content: center !important;
                text-align: center !important;
            }

            div[data-testid="stButton"] > button > div {
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                width: 100% !important;
            }

            div[data-testid="stButton"] > button p {
                margin: 0 !important;
                text-align: center !important;
                width: 100% !important;
                line-height: 1.2 !important;
            }

            button[kind="secondary"] {
                padding: 0.5rem 0.9rem !important;
                border-radius: 12px !important;
                font-size: 0.95rem !important;
            }

            /* back button */
            div[data-testid="stButton"] > button.back-arrow-btn {
                background: transparent !important;
                border: none !important;
                box-shadow: none !important;
                padding: 0 !important;
                margin: 0 !important;
                min-height: auto !important;
                height: auto !important;
                width: auto !important;
                color: rgb(49, 51, 63) !important;
                font-size: 1rem !important;
                font-weight: 500 !important;
                line-height: 1 !important;
                display: inline-flex !important;
                align-items: center !important;
                justify-content: flex-start !important;
            }

            div[data-testid="stButton"] > button.back-arrow-btn > div {
                justify-content: flex-start !important;
            }

            div[data-testid="stButton"] > button.back-arrow-btn p {
                text-align: left !important;
                width: auto !important;
                margin: 0 !important;
                line-height: 1 !important;
            }

            div[data-testid="stButton"] > button.back-arrow-btn:hover {
                background: transparent !important;
                border: none !important;
                box-shadow: none !important;
                color: rgb(49, 51, 63) !important;
            }

            div[data-testid="stButton"] > button.back-arrow-btn:focus {
                outline: none !important;
                box-shadow: none !important;
            }

            /* top nav */
            .top-nav-wrap {
                display: flex;
                align-items: center;
                min-height: 32px;
                margin-top: 0;
                margin-bottom: 1rem;
            }

            .top-nav-title {
                font-size: 1.85rem;
                font-weight: 700;
                line-height: 1.1;
                color: rgb(49, 51, 63);
                margin: 0;
                padding: 0;
            }

            /* camera component */
            iframe {
                border-radius: 18px !important;
                overflow: hidden !important;
            }
            /* version badge */
            .version-badge {
                position: fixed;
                bottom: 1rem;
                left: 1rem;
                font-size: 0.75rem;
                color: rgba(49, 51, 63, 0.45);
                z-index: 9999;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="version-badge">v{APP_VERSION}</div>',
        unsafe_allow_html=True,
    )

# =============================================================================
# helpers
# =============================================================================

@dataclass
class ProjectPaths:
    root: Path
    app_architecture: Path
    birds_images: Path
    species_info: Path
    model_dir: Path
    model_weights: Path
    model_script: Path
    books_dir: Path
    book_pdf: Path


@dataclass
class PredictionResult:
    predicted_index: int
    confidence: float
    top_predictions: List[Dict[str, object]]
    species_row: Dict[str, object]


class StreamlitFriendlyError(RuntimeError):
    def __init__(self, category: str, details: str = "") -> None:
        self.category = category
        self.details = details
        super().__init__(details or category)


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("_", " ")
    text = re.sub(r"[^a-z0-9 ]+", "", text)
    return text.strip()


def slugify(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = text.replace("&", "and")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def init_session_state() -> None:
    defaults = {
        "screen": "home",
        "history": [],
        "captured_image": None,
        "captured_image_name": None,
        "image_for_model": None,
        "prediction": None,
        "selected_species_key": None,
        "gradcam_image": None,
        "manual_input": {},
        "manual_matches": [],
        "last_error": None,
        "show_splash": False,
        "book_chat_history": [],
        "book_last_query": "",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def go_to(screen: str) -> None:
    current = st.session_state.get("screen")

    if current and current != screen:
        st.session_state.history.append(current)

    st.session_state.screen = screen

def go_back() -> None:
    if st.session_state.history:
        previous = st.session_state.history.pop()
        st.session_state.screen = previous
    else:
        st.session_state.screen = "home"

def reset_scan_state() -> None:
    st.session_state.captured_image = None
    st.session_state.captured_image_name = None
    st.session_state.image_for_model = None
    st.session_state.prediction = None
    st.session_state.selected_species_key = None
    st.session_state.gradcam_image = None
    st.session_state.last_error = None


def render_top_nav(title: str, show_back: bool = True) -> None:
    st.markdown(
        """
        <style>
        .top-nav-wrap {
            display: flex;
            align-items: center;
            min-height: 32px;
            margin-top: 0;
            margin-bottom: 1rem;
        }

        .top-nav-title {
            font-size: 1.85rem;
            font-weight: 700;
            line-height: 1.1;
            color: rgb(49, 51, 63);
            margin: 0;
            padding: 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.2, 8.8], vertical_alignment="center")

    with left:
        if show_back and st.session_state.screen != "home":
            if st.button("← Back", key=f"back_{st.session_state.screen}"):
                go_back()
                st.rerun()

    st.markdown(
        """
        <script>
        const buttons = window.parent.document.querySelectorAll('div[data-testid="stButton"] > button');
        buttons.forEach(btn => {
            const text = btn.innerText.trim();
            if (text === '← Back') {
                btn.classList.add('back-arrow-btn');
            }
        });
        </script>
        """,
        unsafe_allow_html=True,
    )

    with right:
        st.markdown(
            f"<div class='top-nav-wrap'><div class='top-nav-title'>{title}</div></div>",
            unsafe_allow_html=True,
        )


def find_project_root() -> ProjectPaths:
    override = st.session_state.get("project_root_override", "").strip()
    candidates: List[Path] = []

    if override:
        candidates.append(Path(override).expanduser())

    cwd = Path.cwd()
    candidates.extend(
        [
            cwd,
            cwd / "Streamlit app",
            Path(__file__).resolve().parent,
            Path(__file__).resolve().parent / "Streamlit app",
            Path("/mnt/data"),
            Path("/mnt/data") / "Streamlit app",
        ]
    )

    for base in candidates:
        app_arch = base / APP_ARCH_DIRNAME
        model_dir = base / MODEL_DIRNAME

        if app_arch.exists() and model_dir.exists():
            species_info = app_arch / SPECIES_FILENAME
            birds_images = app_arch / BIRD_IMAGES_DIRNAME
            model_weights = model_dir / MODEL_FILENAME
            model_script = model_dir / MODEL_SCRIPT_FILENAME
            books_dir = app_arch / BOOKS_DIRNAME
            book_pdf = books_dir / BOOK_FILENAME

            return ProjectPaths(
                root=base,
                app_architecture=app_arch,
                birds_images=birds_images,
                species_info=species_info,
                model_dir=model_dir,
                model_weights=model_weights,
                model_script=model_script,
                books_dir=books_dir,
                book_pdf=book_pdf,
            )

    raise StreamlitFriendlyError(
        "device",
        "Project folders were not found. Set the correct root path in the sidebar so the app can find APP_ARCHITECTURE and MODEL.",
    )

@st.cache_data(show_spinner=False)
def load_species_info(species_info_path: str) -> pd.DataFrame:
    df = pd.read_excel(species_info_path)
    columns = [str(col).strip().lower().replace(" ", "_") for col in df.columns]
    df.columns = columns
    df = df.rename(columns={df.columns[0]: "key"})
    df["key"] = pd.to_numeric(df["key"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["key"]).copy()
    df["key"] = df["key"].astype(int)

    for column in df.columns:
        if column != "key":
            df[column] = df[column].apply(lambda x: "" if pd.isna(x) else str(x).strip())

    return df.sort_values("key").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def build_bird_image_index(birds_images_dir: str) -> Dict[str, str]:
    directory = Path(birds_images_dir)
    image_paths = [p for p in directory.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]

    image_index: Dict[str, str] = {}
    for image_path in image_paths:
        stem = normalize_text(image_path.stem)
        image_index[stem] = str(image_path)
        image_index[slugify(image_path.stem)] = str(image_path)
    return image_index


def get_bird_image_path(species_row: Dict[str, object], image_index: Dict[str, str]) -> Optional[str]:
    candidates = [
        str(species_row.get("key", "")),
        normalize_text(species_row.get("scientific_name", "")),
        slugify(species_row.get("scientific_name", "")),
        normalize_text(species_row.get("common_name", "")),
        slugify(species_row.get("common_name", "")),
    ]

    for candidate in candidates:
        if candidate and candidate in image_index:
            return image_index[candidate]

    for key, value in image_index.items():
        sci = normalize_text(species_row.get("scientific_name", ""))
        com = normalize_text(species_row.get("common_name", ""))
        if sci and sci in key:
            return value
        if com and com in key:
            return value
    return None


@st.cache_resource(show_spinner=False)
def load_model(model_weights_path: str, num_classes: int) -> nn.Module:
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)
    state_dict = torch.load(model_weights_path, map_location=DEVICE)

    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        cleaned_key = key.replace("module.", "")
        cleaned_state_dict[cleaned_key] = value

    model.load_state_dict(cleaned_state_dict, strict=True)
    model.eval()
    model.to(DEVICE)
    return model


@st.cache_resource(show_spinner=False)
def load_book_index(book_pdf_path: str) -> Dict[str, object]:
    #building the book index once and reusing it while the app is running
    return build_book_index(book_pdf_path)


def preprocess_for_model(pil_image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return transform(pil_image.convert("RGB")).unsqueeze(0)


def draw_capture_guide(pil_image: Image.Image) -> Image.Image:
    image = pil_image.convert("RGB").copy()
    width, height = image.size
    left = int((1 - CENTER_CROP_WIDTH_RATIO) * width / 2) # type: ignore
    top = int((1 - CENTER_CROP_HEIGHT_RATIO) * height / 2) # type: ignore
    right = width - left
    bottom = height - top

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle([(0, 0), (width, height)], fill=(0, 0, 0, 90))
    draw.rectangle([(left, top), (right, bottom)], fill=(0, 0, 0, 0), outline=(255, 255, 255, 255), width=6)

    guide = Image.alpha_composite(image.convert("RGBA"), overlay)
    return guide.convert("RGB")


def validate_input_image(pil_image: Optional[Image.Image]) -> None:
    if pil_image is None:
        raise StreamlitFriendlyError("camera", "No image was captured.")
    width, height = pil_image.size
    if width < 150 or height < 150:
        raise StreamlitFriendlyError("camera", "The image is too small for reliable processing.")


def predict_with_model(model: nn.Module, image_for_model: Image.Image, species_df: pd.DataFrame) -> PredictionResult:
    input_tensor = preprocess_for_model(image_for_model).to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]

    top_indices = np.argsort(probabilities)[::-1][:TOP_K]
    predicted_index = int(top_indices[0])
    confidence = float(probabilities[predicted_index])

    top_predictions = []
    for index in top_indices:
        row = species_df.loc[species_df["key"] == int(index)]
        if row.empty:
            continue
        species_row = row.iloc[0].to_dict()
        top_predictions.append(
            {
                "index": int(index),
                "confidence": float(probabilities[index]),
                "common_name": species_row.get("common_name", "Unknown species"),
                "scientific_name": species_row.get("scientific_name", ""),
                "species_row": species_row,
            }
        )

    species_match = species_df.loc[species_df["key"] == predicted_index]
    if species_match.empty:
        raise StreamlitFriendlyError("results", f"No species row matched key {predicted_index}.")

    return PredictionResult(
        predicted_index=predicted_index,
        confidence=confidence,
        top_predictions=top_predictions,
        species_row=species_match.iloc[0].to_dict(),
    )


def generate_gradcam(model: nn.Module, image_for_model: Image.Image) -> Optional[Image.Image]:
    if GradCAM is None or show_cam_on_image is None:
        return None

    if hasattr(model, "conv_head"):
        target_layer = model.conv_head
    elif hasattr(model, "blocks"):
        target_layer = model.blocks[-1]
    else:
        return None

    input_tensor = preprocess_for_model(image_for_model).to(DEVICE)
    target_layers = [target_layer]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor)[0]

    base_image = image_for_model.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    base_array = np.asarray(base_image).astype(np.float32) / 255.0
    cam_image = show_cam_on_image(base_array, grayscale_cam, use_rgb=True)
    return Image.fromarray(cam_image)


def format_confidence(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_error_card(category: str, extra_details: str = "") -> None:
    payload = ERROR_MESSAGES.get(category, ERROR_MESSAGES["results"])
    st.markdown(
        f"<div class='error-card'><div class='section-label'>{payload['title']}</div><p>{payload['message']}</p></div>",
        unsafe_allow_html=True,
    )
    if extra_details:
        st.caption(extra_details)
    st.write("Try this:")
    for action in payload["actions"]:
        st.write(f"- {action}")



# =============================================================================
# JSON-based manual classification
# =============================================================================

# Maps the form's human-readable values to the lowercase tokens used in egg_dataset.json
def _normalize_manual_input(form: Dict[str, str]) -> Dict[str, object]:
    """Convert form values directly to the attribute schema used by egg_dataset.json.
    Form keys and values now match the JSON schema, so only lowercase conversion is needed.
    'very small' -> 'very_small' for the size field is the only special case.
    """
    raw_size = form.get("size", "small").lower()
    size = "very_small" if raw_size == "very small" else raw_size

    raw_intensity = form.get("marking_intensity", "none").lower()
    intensity = None if raw_intensity == "none" else raw_intensity

    raw_pattern = form.get("pattern", "none").lower()
    pattern = "" if raw_pattern == "none" else raw_pattern

    return {
        "color": form.get("color", "white").lower(),
        "pattern": pattern,
        "marking_color": form.get("marking_color", "brown").lower(),
        "marking_intensity": intensity,
        "size": size,
    }


def _score_species(user: Dict[str, object], species: Dict) -> int:
    """Return a match score for one species against user-supplied attributes."""
    score = 0
    attr = species["attributes"]

    if user["color"] in attr.get("color", []):
        score += 3

    if user["pattern"] and user["pattern"] in attr.get("pattern", []):
        score += 2

    if user["marking_color"] in attr.get("marking_color", []):
        score += 2

    if attr.get("marking_intensity") and user.get("marking_intensity") == attr.get("marking_intensity"):
        score += 1

    if user["size"] == attr.get("size"):
        score += 2

    return score


def get_top_matches(user_input: Dict[str, object], top_n: int = 3) -> List[Dict]:
    """Return the top-N best-matching species dicts from EGG_DATA."""
    results = [(species, _score_species(user_input, species)) for species in EGG_DATA]
    results.sort(key=lambda x: x[1], reverse=True)
    return [species for species, _score in results[:top_n]]


# =============================================================================
# screen renderers
# =============================================================================

def render_sidebar(paths: Optional[ProjectPaths]) -> None:
    with st.sidebar:
        st.markdown("## Menu")

        if st.button("Home", key="sidebar_home", use_container_width=True):
            st.session_state.history = []  #clearing navigation history when going home from menu
            go_to("home")
            st.rerun()

        if st.button("Ask the book", key="sidebar_book_chat", use_container_width=True):
            go_to("book_chat")
            st.rerun()

        if st.button("About", key="sidebar_about", use_container_width=True):
            go_to("about")
            st.rerun()

        if st.button("Scientific disclaimer", key="sidebar_disclaimer", use_container_width=True):
            go_to("disclaimer")
            st.rerun()


def render_home_screen() -> None:

    st.markdown(
        "<div class='home-subtitle'>How would you like to identify an egg?</div>",
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown("<div class='home-card-wrap'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='home-icon-wrap'><span>photo_camera</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div class='home-card-title'>Scan egg</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='home-card-text'>Capture a photo of an egg for identification.</div>",
            unsafe_allow_html=True,
        )
        if st.button("Take picture", key="home_take_picture", width="stretch"):
            reset_scan_state()
            st.session_state.history = []  #clearing previous navigation
            go_to("instructions")
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    with st.container(border=True):
        st.markdown("<div class='home-card-wrap'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='home-icon-wrap'><span>description</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div class='home-card-title'>Describe egg</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='home-card-text'>Describe the egg's visual characteristics to get a match prediction.</div>",
            unsafe_allow_html=True,
        )
        if st.button("Describe egg", key="home_describe_egg", width="stretch"):
            st.session_state.history = []  #clearing previous navigation
            go_to("manual_input")
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    with st.container(border=True):
        st.markdown("<div class='home-card-wrap'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='home-icon-wrap'><span>menu_book</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div class='home-card-title'>Ask the book</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='home-card-text'>Search the historical reference book and retrieve relevant passages about eggs, nests, and bird species.</div>",
            unsafe_allow_html=True,
        )
        if st.button("Ask the book", key="home_ask_the_book", width="stretch"):
            st.session_state.history = []  #clearing previous navigation
            go_to("book_chat")
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


def render_disclaimer_screen() -> None:
    render_top_nav("Scientific disclaimer and limitations")

    st.write(HOME_DISCLAIMER)

    if st.button("Back to home", key="back_from_disclaimer", width="stretch"):
        go_to("home")
        st.rerun()


def render_instructions_screen() -> None:
    render_top_nav("How to capture the egg")

    st.write("Use one egg only.")
    st.write("Center the egg inside the guide frame and let it fill the frame as much as possible.")
    st.write("Use even lighting and avoid shadows, glare, fingers, labels, rulers, or extra objects.")
    st.write("At this stage, the app supports direct camera capture only.")

    if st.button("Open camera", key="open_camera_button", width="stretch"):
        go_to("camera")
        st.rerun()


def render_camera_screen() -> None:
    render_top_nav("Capture the egg", show_back=True)

    st.markdown(
        """
        <style>
            .camera-screen-note {
                font-size: 1rem;
                color: #4b5563;
                line-height: 1.5;
                margin-top: 0.35rem;
                margin-bottom: 0.75rem;
            }
            .camera-help-text {
                text-align: center;
                font-size: 0.78rem;
                color: rgba(49, 51, 63, 0.72);
                line-height: 1.5;
                margin-top: 0.6rem;
            }

            div[data-testid="stCameraInput"] > div {
                border-radius: 22px !important;
                overflow: hidden !important;
                background: #000 !important;
            }

            div[data-testid="stCameraInput"] video {
                display: block !important;
                width: 100% !important;
                object-fit: cover !important;
            }

            div[data-testid="stCameraInput"] button {
                width: 100% !important;
                min-height: 64px !important;
                font-size: 1rem !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="camera-screen-note">Use one egg only. Center it inside the square.</div>',
        unsafe_allow_html=True,
    )

    # MutationObserver that re-applies height whenever Streamlit resets the video
    st.components.v1.html(
        """
        <script>
            function enforceVideoHeight() {
                const videos = window.parent.document.querySelectorAll(
                    'div[data-testid="stCameraInput"] video'
                );
                videos.forEach(video => {
                    video.style.setProperty('height', '420px', 'important');
                    video.style.setProperty('min-height', '420px', 'important');
                });
            }

            // Run immediately
            enforceVideoHeight();

            // Re-run whenever the DOM changes
            const observer = new MutationObserver(enforceVideoHeight);
            observer.observe(window.parent.document.body, {
                childList: true,
                subtree: true,
                attributes: true,
                attributeFilter: ['style', 'class']
            });

            // Also run on an interval as a fallback
            setInterval(enforceVideoHeight, 300);
        </script>
        """,
        height=0,
    )

    camera_file = st.camera_input("", key="egg_camera_native", label_visibility="collapsed")

    if camera_file is not None:
        try:
            captured_image = Image.open(camera_file).convert("RGB")
            square_image = crop_to_center_square(captured_image)
            validate_input_image(square_image)
            st.session_state["captured_image"] = square_image
            st.session_state["image_for_model"] = square_image
            go_to("review")
            st.rerun()
        except Exception as exc:
            st.session_state.last_error = ("camera", str(exc))
            render_error_card("camera", str(exc))
            return

    st.markdown(
        '<div class="camera-help-text">If your front camera opens first, switch to the back camera using your device controls.</div>',
        unsafe_allow_html=True,
    )

def render_review_screen() -> None:
    render_top_nav("Review the captured image")

    image = st.session_state.get("captured_image")

    if image is None:
        render_error_card("camera", "No captured image found in session.")
        if st.button("Back to camera", key="review_back_to_camera", width="stretch"):
            go_to("camera")
            st.rerun()
        return

    #showing the already-square captured image
    st.image(
        image,
        caption="Captured image",
        use_container_width=True
    )

    c1, c2 = st.columns(2)

    with c1:
        if st.button("Retake photo", key="retake_photo_button", width="stretch"):
            st.session_state.captured_image = None
            go_to("camera")
            st.rerun()

    with c2:
        if st.button("Process image", key="process_image_button", width="stretch"):
            go_to("processing")
            st.rerun()

def render_processing_screen(paths: ProjectPaths, species_df: pd.DataFrame) -> None:
    render_top_nav("Analyzing the image")

    image_for_model = st.session_state.get("image_for_model")
    if image_for_model is None:
        render_error_card("camera", "The captured image for prediction is missing.")
        return

    progress = st.progress(10)
    status = st.empty()

    try:
        status.write("Loading model")
        progress.progress(35)
        model = load_model(str(paths.model_weights),num_classes=21)

        status.write("Running prediction")
        progress.progress(65)
        prediction = predict_with_model(model, image_for_model, species_df)
        gradcam_image = generate_gradcam(model, image_for_model)

        status.write("Preparing results")
        progress.progress(100)

        st.session_state.prediction = prediction
        st.session_state.selected_species_key = prediction.predicted_index
        st.session_state.gradcam_image = gradcam_image
        go_to("results")
        st.rerun()

    except Exception as exc:
        st.session_state.last_error = ("model", str(exc))
        render_error_card("model", str(exc))
        if st.button("Try again", key="processing_try_again", width="stretch"):
            go_to("camera")
            st.rerun()

def render_results_screen(species_df: pd.DataFrame) -> None:
    render_top_nav("Prediction result")

    prediction: PredictionResult = st.session_state.get("prediction")
    if prediction is None:
        render_error_card("results", "No prediction result found.")
        return

    high_confidence = prediction.confidence >= HIGH_CONFIDENCE_THRESHOLD
    species_row = prediction.species_row

    if high_confidence:
        st.markdown(
            f"<div class='confidence-badge'>Confidence: {format_confidence(prediction.confidence)}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='species-name'>{species_row.get('common_name', 'Unknown species')}</div>",
            unsafe_allow_html=True,
        )
        scientific_name = species_row.get("scientific_name", "")
        if scientific_name:
            st.markdown(f"*{scientific_name}*")

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Why this result", key="results_why", width="stretch"):
                go_to("explainability")
                st.rerun()
        with c2:
            if st.button("View details", key="results_details", width="stretch"):
                st.session_state.selected_species_key = prediction.predicted_index
                go_to("species_details")
                st.rerun()
        with c3:
            if st.button("Scan again", key="results_scan_again", width="stretch"):
                reset_scan_state()
                go_to("camera")
                st.rerun()
    else:
        st.warning("The result has lower confidence. Review the top matches below and consider manual input.")
        for item in prediction.top_predictions:
            label = f"{item['common_name']} ({format_confidence(float(item['confidence']))})"
            if st.button(label, key=f"top_pred_{item['index']}", width="stretch"):
                st.session_state.selected_species_key = int(item["index"])
                go_to("species_details")
                st.rerun()

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Why this result", key="results_low_why", width="stretch"):
                go_to("explainability")
                st.rerun()
        with c2:
            if st.button("Manual input", key="results_low_manual", width="stretch"):
                go_to("manual_input")
                st.rerun()

    if st.session_state.get("image_for_model") is not None:
        st.image(st.session_state.image_for_model, caption="Image used for prediction", width="stretch")


def render_explainability_screen() -> None:
    render_top_nav("Why the model leaned toward this result")

    cropped = st.session_state.get("image_for_model")
    gradcam = st.session_state.get("gradcam_image")

    col1, col2 = st.columns(2)
    with col1:
        if cropped is not None:
            st.image(cropped.resize((IMG_SIZE, IMG_SIZE)), caption="Cropped egg image", width="stretch")
    with col2:
        if gradcam is not None:
            st.image(gradcam, caption="Grad-CAM heatmap", width="stretch")
        else:
            render_error_card(
                "model",
                "Grad-CAM is not available. Make sure grad-cam is installed and model.conv_head exists.",
            )

    st.write(EXPLAINABILITY_TEXT_1)
    st.write(EXPLAINABILITY_TEXT_2)

    if st.button("Back to results", key="explainability_back", width="stretch"):
        go_to("results")
        st.rerun()
    else:
            grad_status = "GradCAM imported OK" if GradCAM is not None else "GradCAM import FAILED"
            render_error_card(
                "model",
                f"Grad-CAM is not available. Status: {grad_status}. gradcam_image={st.session_state.get('gradcam_image')}",
            )

def render_species_details_screen(species_df: pd.DataFrame, image_index: Dict[str, str]) -> None:
    render_top_nav("Species profile")

    key = st.session_state.get("selected_species_key")
    if key is None:
        prediction: PredictionResult = st.session_state.get("prediction")
        key = prediction.predicted_index if prediction else None

    row_match = species_df.loc[species_df["key"] == int(key)] if key is not None else pd.DataFrame()
    if row_match.empty:
        render_error_card("results", "No species details were found for the selected key.")
        return

    row = row_match.iloc[0].to_dict()
    bird_image_path = get_bird_image_path(row, image_index)

    st.markdown(f"### {row.get('common_name', 'Unknown species')}")
    if row.get("scientific_name"):
        st.markdown(f"*{row['scientific_name']}*")

    if bird_image_path:
        st.image(bird_image_path, caption="Bird image", width="stretch")
    else:
        st.caption("Bird image not found for this species.")

    details_order = [
        ("Identity", row.get("identity_sentence", "")),
        ("Nesting habits", row.get("nesting_habits", "")),
        ("Location", row.get("location", "")),
        ("Habitat", row.get("habitat", "")),
        ("Egg confirmation", row.get("egg_confirmation", row.get("Egg confirmation", ""))),
    ]

    for label, value in details_order:
        st.markdown(f"**{label}**")
        st.write(value if value else "Not available.")

    if st.button("Back to results", key="species_back_to_results", width="stretch"):
        go_to("results")
        st.rerun()


def render_manual_input_screen() -> None:
    render_top_nav("Describe the egg manually")

    st.write("Use this when the image result is uncertain or when you want an additional structured record.")

    with st.form("manual_input_form"):
        manual_values = {}
        for field_name, options in MANUAL_INPUT_OPTIONS.items():
            label = field_name.replace("_", " ").title()
            manual_values[field_name] = st.selectbox(label, options, key=f"manual_{field_name}")

        submitted = st.form_submit_button("Find matches", width="stretch")

    if submitted:
        st.session_state.manual_input = manual_values

        if not EGG_DATA:
            st.error("Egg dataset not found. Make sure egg_dataset.json is in the same folder as app.py.")
        else:
            user_attrs = _normalize_manual_input(manual_values)
            top_matches = get_top_matches(user_attrs, top_n=3)

            # Store results so render_manual_results_screen can display them
            st.session_state.manual_matches = top_matches
            go_to("manual_results")
            st.rerun()


def render_manual_results_screen(species_df: pd.DataFrame, image_index: Dict[str, str]) -> None:
    render_top_nav("Manual match results")

    matches: List[Dict] = st.session_state.get("manual_matches", [])

    if not matches:
        st.warning("No matches found. Try adjusting the egg description.")
        if st.button("Back to description", key="manual_results_back", width="stretch"):
            go_to("manual_input")
            st.rerun()
        return

    st.write("Top matches based on your description. The first result is the closest match.")

    for rank, species in enumerate(matches):
        sci_name = species.get("scientific_name", "")
        common_name = species.get("common_name", sci_name)

        # Look up the full species row in the Excel data so we can reuse the
        # existing display pipeline (bird image + detail fields).
        row_match = species_df.loc[
            species_df["scientific_name"].str.strip().str.lower() == sci_name.strip().lower()
        ] if "scientific_name" in species_df.columns else pd.DataFrame()

        with st.container(border=True):
            label = "Best match" if rank == 0 else f"Alternative {rank}"
            st.markdown(f"<div class='section-label'>{label}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='species-name'>{common_name}</div>", unsafe_allow_html=True)
            if sci_name:
                st.markdown(f"*{sci_name}*")

            # Show bird image if available
            if not row_match.empty:
                row_dict = row_match.iloc[0].to_dict()
                bird_img = get_bird_image_path(row_dict, image_index)
                if bird_img:
                    st.image(bird_img, caption="Bird image", use_container_width=True)

            # Button to view full species details
            if not row_match.empty:
                key_val = int(row_match.iloc[0]["key"])
                if st.button(f"View details — {common_name}", key=f"manual_details_{rank}", width="stretch"):
                    st.session_state.selected_species_key = key_val
                    go_to("species_details")
                    st.rerun()

    st.write("")
    if st.button("Describe again", key="manual_results_redo", width="stretch"):
        go_to("manual_input")
        st.rerun()


def render_error_states_screen() -> None:
    render_top_nav("Recovery guidance")

    selected = st.selectbox(
        "Select an error category",
        list(ERROR_MESSAGES.keys()),
        format_func=lambda x: ERROR_MESSAGES[x]["title"],
    )
    render_error_card(selected)

    if st.session_state.get("last_error"):
        category, details = st.session_state["last_error"]
        st.markdown("**Latest error in this session**")
        render_error_card(category, details)


def render_about_screen() -> None:
    render_top_nav("About Eggcellent ID")

    st.markdown("**Project description**")
    st.write(ABOUT_CONTENT["project_description"])

    st.markdown("**Intended use**")
    st.write(
        "This application is designed to support museum and scientific workflows by helping trained "
        "users review likely species matches based on egg appearance."
    )

    st.markdown("**Model version**")
    st.write(ABOUT_CONTENT["model_version"])

    st.markdown("**Dataset version**")
    st.write(ABOUT_CONTENT["dataset_version"])

    st.markdown("**Acknowledgements**")
    st.write(ABOUT_CONTENT["acknowledgements"])

    if st.button("View scientific disclaimer", key="about_disclaimer_link", width="stretch"):
        go_to("disclaimer")
        st.rerun()


def render_book_chat_screen(paths: ProjectPaths) -> None:
    render_top_nav("Ask the book")

    st.write(
        "Ask a question about nests, eggs, bird species, colors, markings, or nesting habits based on the reference book."
    )

    if not paths.book_pdf.exists():
        render_error_card(
            "device",
            f"Book pdf not found at: {paths.book_pdf}"
        )
        return

    try:
        with st.spinner("Loading the book index"):
            book_index = load_book_index(str(paths.book_pdf))
    except Exception as exc:
        render_error_card("device", f"Could not load the book pdf: {exc}")
        return

    #showing previous chat messages
    chat_history = st.session_state.get("book_chat_history", [])
    for message in chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and message.get("sources"):
                st.markdown("**Sources**")
                for source in message["sources"]:
                    st.markdown(
                        f"- page {source['page_number']} | similarity: {source['score']:.3f}"
                    )

    #capturing new question
    user_query = st.chat_input("Ask something like: Which eggs are blue-green with spots?")

    if user_query:
        st.session_state.book_chat_history.append(
            {
                "role": "user",
                "content": user_query,
            }
        )

        with st.chat_message("user"):
            st.markdown(user_query)

        try:
            response = answer_book_question(
                query=user_query,
                book_index=book_index,
                top_k=BOOK_TOP_K,
            )

            answer_text = response["answer"]
            results = response["results"]

            source_payload = [
                {
                    "page_number": item.page_number,
                    "score": item.score,
                    "text": item.text,
                }
                for item in results
            ]

            with st.chat_message("assistant"):
                st.markdown(answer_text)

                if source_payload:
                    st.markdown("**Sources**")
                    for source in source_payload:
                        st.markdown(
                            f"- page {source['page_number']} | similarity: {source['score']:.3f}"
                        )

                    with st.expander("See retrieved passages"):
                        for index, source in enumerate(source_payload, start=1):
                            st.markdown(f"**Passage {index} — page {source['page_number']}**")
                            st.write(source["text"])

            st.session_state.book_chat_history.append(
                {
                    "role": "assistant",
                    "content": answer_text,
                    "sources": source_payload,
                }
            )

        except Exception as exc:
            with st.chat_message("assistant"):
                st.error(f"I could not answer from the book right now: {exc}")

    c1, c2 = st.columns(2)

    with c1:
        if st.button("Clear chat", key="clear_book_chat", width="stretch"):
            st.session_state.book_chat_history = []
            st.rerun()

    with c2:
        if st.button("Back to home", key="book_chat_back_home", width="stretch"):
            go_to("home")
            st.rerun()



def crop_to_center_square(pil_image: Image.Image) -> Image.Image:
    #cropping to a centered square to keep the model input safe from distortion
    image = pil_image.convert("RGB")
    width, height = image.size

    square_size = min(width, height)

    left = (width - square_size) // 2
    top = (height - square_size) // 2
    right = left + square_size
    bottom = top + square_size

    return image.crop((left, top, right, bottom))

# =============================================================================
# main
# =============================================================================

def main() -> None:
    inject_css()
    init_session_state()

    paths = None
    species_df = None
    image_index: Dict[str, str] = {}

    try:
        paths = find_project_root()

        if not paths.species_info.exists():
            raise StreamlitFriendlyError("device", f"Missing file: {paths.species_info}")

        if not paths.model_weights.exists():
            raise StreamlitFriendlyError("model", f"Missing file: {paths.model_weights}")

        species_df = load_species_info(str(paths.species_info))

        if paths.birds_images.exists():
            image_index = build_bird_image_index(str(paths.birds_images))

    except StreamlitFriendlyError as exc:
        st.session_state.last_error = (exc.category, exc.details)

    except Exception as exc:
        st.session_state.last_error = ("device", str(exc))

    render_sidebar(paths)

    screen = st.session_state.screen

    if paths is None or species_df is None:
        render_top_nav("Setup issue", show_back=False)
        render_error_card(
            *st.session_state.last_error
            if st.session_state.last_error
            else ("device", "Unknown setup error.")
        )
        st.stop()

    if screen == "home":
        render_home_screen()
    elif screen == "instructions":
        render_instructions_screen()
    elif screen == "camera":
        render_camera_screen()
    elif screen == "review":
        render_review_screen()
    elif screen == "processing":
        render_processing_screen(paths, species_df)
    elif screen == "results":
        render_results_screen(species_df)
    elif screen == "explainability":
        render_explainability_screen()
    elif screen == "species_details":
        render_species_details_screen(species_df, image_index)
    elif screen == "manual_input":
        render_manual_input_screen()
    elif screen == "manual_results":
        render_manual_results_screen(species_df, image_index)
    elif screen == "book_chat":
        render_book_chat_screen(paths)
    elif screen == "error_states":
        render_error_states_screen()
    elif screen == "about":
        render_about_screen()
    elif screen == "disclaimer":
        render_disclaimer_screen()
    else:
        go_to("home")
        st.rerun()


if __name__ == "__main__":
    main()