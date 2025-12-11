import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import io
from collections import Counter
import os

# -------------------------------
# Optional: OpenAI (for AI copy)
# -------------------------------
try:
    from openai import OpenAI

    client = OpenAI()  # uses OPENAI_API_KEY from environment
    HAS_OPENAI = bool(os.getenv("OPENAI_API_KEY"))
except Exception:
    client = None
    HAS_OPENAI = False

# -------------------------------
# Optional: background removal
# -------------------------------
try:
    from rembg import remove

    REMBG_AVAILABLE = True
except Exception:
    REMBG_AVAILABLE = False

# -------------------------------
# Brand profiles (presets)
# -------------------------------
BRAND_PROFILES = {
    "Default Biscuit Launch": {
        "headline": "Crunchy Cream Biscuits",
        "offer": "Intro Offer – 20% OFF",
        "cta": "Shop Now",
        "bg": "#F6E3F9",
        "band": "#F2CFF5",
        "category": "Biscuits / Snacks",
        "tone": "Discount-focused",
        "persona": "Family Friendly",
    },
    "Festive Treats": {
        "headline": "Festive Treat, Extra Crunch",
        "offer": "Diwali Special – Buy 1 Get 1",
        "cta": "Grab Now",
        "bg": "#FFF4D6",
        "band": "#F9C95E",
        "category": "Biscuits / Snacks",
        "tone": "Festive",
        "persona": "Family Friendly",
    },
    "Premium Indulgence": {
        "headline": "Crafted For True Indulgence",
        "offer": "New Launch – Limited Edition",
        "cta": "Discover More",
        "bg": "#141625",
        "band": "#222842",
        "category": "Biscuits / Snacks",
        "tone": "Premium",
        "persona": "Premium / Sophisticated",
    },
}

OBJECTIVE_OPTIONS = ["Awareness", "Consideration", "Conversion", "Loyalty"]

# -------------------------------
# Session-state init
# -------------------------------
def init_state():
    if "headline" not in st.session_state:
        st.session_state["headline"] = "Crunchy Cream Biscuits"
    if "offer" not in st.session_state:
        st.session_state["offer"] = "Intro Offer – 20% OFF"
    if "cta" not in st.session_state:
        st.session_state["cta"] = "Shop Now"
    if "bg" not in st.session_state:
        st.session_state["bg"] = "#F6E3F9"
    if "band" not in st.session_state:
        st.session_state["band"] = "#F2CFF5"
    if "category" not in st.session_state:
        st.session_state["category"] = "Biscuits / Snacks"
    if "tone" not in st.session_state:
        st.session_state["tone"] = "Discount-focused"
    if "persona" not in st.session_state:
        st.session_state["persona"] = "Family Friendly"
    if "layout_style" not in st.session_state:
        st.session_state["layout_style"] = "Minimal Light"
    if "objective" not in st.session_state:
        st.session_state["objective"] = "Awareness"


init_state()

# -------------------------------
# Helper functions (fonts & text)
# -------------------------------
def load_font(size: int) -> ImageFont.FreeTypeFont:
    """Load a TTF font if available, otherwise fall back to default."""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    canvas_width: int,
    y: int,
    text: str,
    font: ImageFont.FreeTypeFont,
    fill=(0, 0, 0),
):
    """Draw horizontally centered text (no heavy glow, for clarity)."""
    if not text:
        return

    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    x = (canvas_width - w) // 2
    draw.text((x, y), text, font=font, fill=fill)


def get_fitting_font(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_width: int,
    base_size: int,
) -> ImageFont.FreeTypeFont:
    """Reduce font size until text fits within max_width."""
    size = base_size
    while size > 10:
        font = load_font(size)
        bbox = draw.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        if w <= max_width or not text:
            return font
        size -= 2
    return load_font(10)


# -------------------------------
# Simple color & CV utilities
# -------------------------------
def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def get_luminance(rgb):
    r, g, b = [x / 255.0 for x in rgb]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def get_text_contrast(bg_rgb):
    """Return black or white depending on background luminance."""
    return (0, 0, 0) if get_luminance(bg_rgb) > 0.55 else (255, 255, 255)


def get_logo_dominant_color(logo_img: Image.Image):
    """
    Very simple dominant color detection from logo.
    Always returns an RGB tuple (3 channels), even if image has alpha.
    """
    small = logo_img.convert("RGBA").resize((50, 50))
    pixels = list(small.getdata())  # (R, G, B, A)

    # Remove fully transparent & near-white pixels
    filtered = [
        (r, g, b)
        for (r, g, b, a) in pixels
        if a > 10 and not (r > 240 and g > 240 and b > 240)
    ]
    if not filtered:
        filtered = [(r, g, b) for (r, g, b, a) in pixels]

    most_common = Counter(filtered).most_common(1)[0][0]
    return most_common  # (R, G, B)


def auto_crop_product(product_img: Image.Image):
    """
    Auto object cropping:
    Find bounding box of non-transparent pixels and crop to that region.
    """
    img = product_img.convert("RGBA")
    alpha = img.split()[3]
    bbox = alpha.getbbox()
    if bbox:
        img = img.crop(bbox)
    return img


# -------------------------------
# Copy intelligence helpers
# -------------------------------
def parse_brief_to_category_tone(brief: str):
    brief_low = (brief or "").lower()
    category = "Other"
    tone = "Casual / Everyday"

    if any(w in brief_low for w in ["biscuit", "snack", "cookie"]):
        category = "Biscuits / Snacks"
    elif any(w in brief_low for w in ["drink", "beverage", "juice", "soda"]):
        category = "Beverages"

    if any(w in brief_low for w in ["diwali", "festive", "festival", "celebration"]):
        tone = "Festive"
    elif any(w in brief_low for w in ["premium", "luxury", "indulgent", "rich"]):
        tone = "Premium"
    elif any(w in brief_low for w in ["discount", "offer", "% off", "sale", "deal"]):
        tone = "Discount-focused"

    return category, tone


def suggest_copy(category: str, tone: str):
    """Return (headline, offer, cta) from category + tone (rule-based)."""
    if category == "Biscuits / Snacks":
        if tone == "Festive":
            return (
                "Celebrate With Crunchy Cream Bites",
                "Festive Special – Share The Joy",
                "Celebrate Now",
            )
        if tone == "Premium":
            return (
                "Slow-Baked For True Indulgence",
                "Signature Cream-Filled Layers",
                "Discover The Taste",
            )
        if tone == "Discount-focused":
            return (
                "Crunchy Cream Biscuits",
                "Intro Offer – 20% OFF This Week",
                "Shop Now",
            )
        # Casual
        return (
            "Crunchy Cream For Every Break",
            "Perfect Snack For Any Time",
            "Add To Cart",
        )

    if category == "Beverages":
        if tone == "Festive":
            return (
                "Refresh Every Celebration",
                "Limited Festive Mix Pack",
                "Refresh Now",
            )
        if tone == "Premium":
            return (
                "Brewed For Pure Indulgence",
                "Hand-Crafted, Slow Brewed",
                "Taste The Difference",
            )
        if tone == "Discount-focused":
            return (
                "Grab Your Favourite Drinks",
                "Combo Offers Up To 25% OFF",
                "Stock Up",
            )
        return (
            "Stay Refreshed, All Day",
            "Flavourful Drinks For Every Mood",
            "Try It Today",
        )

    # Other
    return (
        "Bring Your Brand To Life",
        "Smart Launch Offer For Early Shoppers",
        "Learn More",
    )


# -------------------------------
# Aesthetic & performance scores
# -------------------------------
def compute_aesthetic_score(bg_rgb, band_rgb, text_rgb):
    """Return a simple 0-100 aesthetic score for UI demo."""
    bg_l = get_luminance(bg_rgb)
    band_l = get_luminance(band_rgb)
    text_l = get_luminance(text_rgb)

    contrast_band = abs(bg_l - band_l)
    contrast_text = abs(band_l - text_l)

    score = 0
    score += min(contrast_band * 100, 30)  # up to 30 points
    score += min(contrast_text * 150, 50)  # up to 50 points

    if 0.25 < bg_l < 0.8:
        score += 20
    else:
        score += 10

    return int(max(0, min(score, 100)))


def objective_placement_boost(objective: str, placement: str) -> int:
    """
    Extra score boost depending on objective + placement type.
    placement ∈ {"feed","story","facebook","tesco","gdn"}
    """
    mapping = {
        "Awareness": {"story": 7, "feed": 5, "facebook": 5, "tesco": 3, "gdn": 4},
        "Consideration": {"feed": 7, "facebook": 6, "tesco": 5, "gdn": 4, "story": 3},
        "Conversion": {"gdn": 7, "tesco": 6, "feed": 5, "facebook": 4, "story": 3},
        "Loyalty": {"facebook": 6, "feed": 5, "tesco": 4, "gdn": 3, "story": 3},
    }
    return mapping.get(objective, {}).get(placement, 0)


def estimate_performance_score(
    aesthetic_score: int,
    under_500kb: bool,
    logo_safe: bool,
    category: str,
    tone: str,
    objective: str,
    placement: str,
):
    """
    Prototype "social media performance" score 0–100.
    Combines design + constraints + basic marketing + objective fit.
    """
    score = aesthetic_score * 0.6
    score += 15 if under_500kb else 5
    score += 10 if logo_safe else 0

    # Tone boost
    if tone == "Festive":
        score += 7
    elif tone == "Discount-focused":
        score += 5
    elif tone == "Premium":
        score += 3

    # Category small tweak
    if category == "Biscuits / Snacks":
        score += 3
    elif category == "Beverages":
        score += 2

    # Objective × placement boost
    score += objective_placement_boost(objective, placement)

    return int(max(0, min(score, 100)))


def objective_comment(objective: str, placement: str):
    """Return (rating, message) for how well this placement supports the objective."""
    rating_map = {
        "Awareness": {
            "story": "High",
            "feed": "High",
            "facebook": "High",
            "tesco": "Medium",
            "gdn": "Medium",
        },
        "Consideration": {
            "feed": "High",
            "facebook": "High",
            "tesco": "Medium",
            "gdn": "Medium",
            "story": "Medium",
        },
        "Conversion": {
            "gdn": "High",
            "tesco": "High",
            "feed": "Medium",
            "facebook": "Medium",
            "story": "Low",
        },
        "Loyalty": {
            "facebook": "High",
            "feed": "Medium",
            "tesco": "Medium",
            "gdn": "Medium",
            "story": "Low",
        },
    }

    rating = rating_map.get(objective, {}).get(placement, "Medium")

    if rating == "High":
        msg = "Strong match for this objective."
    elif rating == "Low":
        msg = "Can be used, but not the strongest match for this objective."
    else:
        msg = "Reasonable fit for this objective."

    return rating, msg


def generate_insights(
    headline: str,
    offer_text: str,
    cta_text: str,
    file_kb: float,
    contrast_ok: bool,
    logo_safe: bool,
    aesthetic_score: int,
    objective: str,
    placement: str,
):
    """Return a list of bullet-point insight strings for judges."""
    insights = []

    # Copy structure
    if len(headline) <= 32:
        insights.append("Headline length is good for quick scanning.")
    else:
        insights.append(
            "Headline is a bit long; consider a tighter version for small screens."
        )

    if offer_text:
        insights.append("Offer line present – good for driving interest.")
    if cta_text:
        if len(cta_text) <= 18:
            insights.append("CTA is short and action-oriented.")
        else:
            insights.append(
                "CTA is present but a bit long – short verbs usually perform better."
            )
    else:
        insights.append("No CTA detected – add a clear action (e.g., “Shop Now”).")

    # Design quality
    if contrast_ok:
        insights.append(
            "Text contrast vs band is strong – should remain readable on most screens."
        )
    else:
        insights.append(
            "Text contrast is low – consider darker text or a deeper band colour."
        )

    if logo_safe:
        insights.append(
            "Logo is inside a safe margin – unlikely to be cropped by placements."
        )
    else:
        insights.append(
            "Logo sits close to the edge – might be cropped on some platforms."
        )

    if file_kb <= 500:
        insights.append(
            f"File size ~{file_kb:.1f} KB – friendly for ad platforms and fast loading."
        )
    else:
        insights.append(
            f"File size ~{file_kb:.1f} KB – consider compressing below 500 KB."
        )

    # Objective × placement
    rating, msg = objective_comment(objective, placement)
    insights.append(
        f"For **{objective}** objective, this placement is a **{rating} fit**. {msg}"
    )

    # Overall aesthetic comment
    if aesthetic_score >= 75:
        insights.append(
            "Overall layout looks strong; focus A/B tests on copy and colour tweaks."
        )
    elif aesthetic_score >= 55:
        insights.append(
            "Design is decent; small tweaks to spacing and hierarchy can further improve performance."
        )
    else:
        insights.append(
            "Visual score is on the lower side – consider simplifying layout and increasing contrast."
        )

    return insights


# -------------------------------
# Creative builder (with auto-crop)
# -------------------------------
def build_creative(
    product_img: Image.Image,
    logo_img: Image.Image,
    canvas_size=(1080, 1080),
    background_color="#F6E3F9",
    band_color="#F2CFF5",
    layout_style="Minimal Light",
    headline="Crunchy Cream Biscuits",
    offer_text="Intro Offer – 20% OFF",
    cta_text="Shop Now",
    text_color=(0, 0, 0),
    auto_crop=True,
    band_ratio=0.23,
    product_scale_override=None,
):
    """
    Compose product + logo + text into a single creative.
    Returns (final_image, metadata_dict).
    """

    # Canvas
    canvas = Image.new("RGB", canvas_size, background_color)
    cw, ch = canvas.size

    # --- Auto object cropping for product ---
    if auto_crop:
        product = auto_crop_product(product_img)
    else:
        product = product_img.copy()

    product = product.convert("RGBA")
    max_pw = int(cw * 0.65)
    max_ph = int(ch * 0.5)
    base_scale = min(max_pw / product.width, max_ph / product.height)

    if product_scale_override is not None:
        scale_p = base_scale * product_scale_override
    else:
        scale_p = base_scale

    new_p_size = (int(product.width * scale_p), int(product.height * scale_p))
    product = product.resize(new_p_size, Image.LANCZOS)

    # Placement – for short banners keep the pack higher
    if ch <= 140:
        product_y = int(ch * 0.05)
    elif layout_style == "Hero Zoom":
        product_y = int(ch * 0.2)
    else:
        product_y = int(ch * 0.18)

    product_x = (cw - new_p_size[0]) // 2
    canvas.paste(product, (product_x, product_y), product)

    # Logo (top-right, safe zone)
    logo = logo_img.convert("RGBA")
    max_lw = int(cw * 0.22)
    scale_l = min(max_lw / logo.width, 1.0)
    new_l_size = (int(logo.width * scale_l), int(logo.height * scale_l))
    logo = logo.resize(new_l_size, Image.LANCZOS)
    margin = int(cw * 0.06)
    logo_x = cw - new_l_size[0] - margin
    logo_y = margin
    canvas.paste(logo, (logo_x, logo_y), logo)

    # Bottom band
    band_height = int(ch * band_ratio)
    band_top = ch - band_height
    band = Image.new("RGB", (cw, band_height), band_color)
    canvas.paste(band, (0, band_top))

    draw = ImageDraw.Draw(canvas)

    # Dynamic font sizes (no cropping)
    max_text_width = int(cw * 0.9)
    base_head_size = int(ch * 0.06)
    base_sub_size = int(ch * 0.035)
    base_cta_size = int(ch * 0.035)

    font_head = get_fitting_font(draw, headline, max_text_width, base_head_size)
    font_sub = get_fitting_font(draw, offer_text, max_text_width, base_sub_size)
    font_cta = get_fitting_font(draw, cta_text, max_text_width, base_cta_size)

    head_y = band_top + int(band_height * 0.18)
    sub_y = band_top + int(band_height * 0.48)
    cta_y = band_top + int(band_height * 0.72)

    draw_centered_text(draw, cw, head_y, headline, font_head, text_color)
    draw_centered_text(draw, cw, sub_y, offer_text, font_sub, text_color)
    if cta_text:
        draw_centered_text(draw, cw, cta_y, cta_text, font_cta, text_color)

    metadata = {
        "canvas_size": canvas_size,
        "logo_pos": (logo_x, logo_y),
        "logo_size": new_l_size,
        "band_top": band_top,
        "band_height": band_height,
    }

    return canvas, metadata


def image_to_bytes(img: Image.Image, format: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()


# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(page_title="ARC-VISION Prototype", layout="centered")

# -------------------------------
# UI: Header & description
# -------------------------------
st.title("ARC-VISION – Self-Driving Retail Media Creative Engine")
st.subheader("Retail Media Creative Prototype")

st.markdown(
    """
Upload a product **packshot** and **brand logo**, describe your campaign,
and ARC-VISION auto-generates creatives for multiple placements:

- Instagram Feed, Story and Facebook
- Tesco retail banner and Google Display banner
- Brand & copy intelligence (profiles, brief understanding, tone/persona)
- Auto object cropping and smart theme from logo colours
- Compliance checks, **aesthetic score** and prototype **social performance score**
- Optional AI copy + A/B creative variations
"""
)

# -------------------------------
# STEP 1 – Uploads
# -------------------------------
st.markdown("### Step 1 – Upload Product & Logo")

col_p, col_l = st.columns(2)

with col_p:
    product_file = st.file_uploader(
        "Upload Product Image (packshot)",
        type=["jpg", "jpeg", "png"],
        key="product_uploader",
    )

with col_l:
    logo_file = st.file_uploader(
        "Upload Brand Logo",
        type=["jpg", "jpeg", "png"],
        key="logo_uploader",
    )

original_product = None
product_no_bg = None
logo_img = None

# Product handling
if product_file is not None:
    product_bytes = product_file.getvalue()
    original_product = Image.open(io.BytesIO(product_bytes)).convert("RGBA")

    st.markdown("**Product Preview**")
    st.image(original_product, caption="Original Product")

    if REMBG_AVAILABLE:
        with st.expander("Background Removal (optional)"):
            with st.spinner("Removing background using AI model..."):
                try:
                    removed_bytes = remove(product_bytes)
                    product_no_bg = Image.open(io.BytesIO(removed_bytes)).convert(
                        "RGBA"
                    )
                    st.image(
                        product_no_bg,
                        caption="Product with Background Removed",
                    )
                except Exception as e:
                    st.error(
                        f"Background removal failed – using original image. ({e})"
                    )
                    product_no_bg = original_product
    else:
        product_no_bg = original_product

# Logo handling
# Logo handling
if logo_file is not None:
    logo_img = Image.open(logo_file).convert("RGBA")
    st.markdown("**Brand Logo Preview**")
    st.image(logo_img, caption="Brand Logo", width=180)

    # OPTIONAL: light auto-theme from logo (only if user hasn't changed colors yet)
    try:
        # If bg/band are still the defaults from init_state, we treat this as first setup
        if st.session_state["bg"] == "#F6E3F9" and st.session_state["band"] == "#F2CFF5":
            dom_rgb = get_logo_dominant_color(logo_img)
            st.session_state["bg"] = rgb_to_hex(dom_rgb)
            darker = (
                max(dom_rgb[0] - 20, 0),
                max(dom_rgb[1] - 20, 0),
                max(dom_rgb[2] - 20, 0),
            )
            st.session_state["band"] = rgb_to_hex(darker)
    except Exception:
        # If anything goes wrong, we just keep the existing colors silently
        pass


# -------------------------------
# STEP 2 – Brand & Copy Intelligence
# -------------------------------
st.markdown("### Step 2 – Brand & Copy Intelligence")

col_brand, col_brief = st.columns(2)

# --- Brand profile side ---
with col_brand:
    profile_name = st.selectbox("Brand Profile", list(BRAND_PROFILES.keys()))
    if st.button("Apply Brand Profile"):
        prof = BRAND_PROFILES[profile_name]
        st.session_state["headline"] = prof["headline"]
        st.session_state["offer"] = prof["offer"]
        st.session_state["cta"] = prof["cta"]
        st.session_state["bg"] = prof["bg"]
        st.session_state["band"] = prof["band"]
        st.session_state["category"] = prof["category"]
        st.session_state["tone"] = prof["tone"]
        st.session_state["persona"] = prof["persona"]

# --- Brief side ---
with col_brief:
    brief = st.text_area(
        "Campaign Brief (optional)",
        placeholder="e.g. Diwali offer on crunchy cream biscuits with 20% off for families.",
        key="brief",
    )

    if st.button("Auto-fill from Brief"):
        cat, tone = parse_brief_to_category_tone(brief)
        st.session_state["category"] = cat
        st.session_state["tone"] = tone
        h, o, c = suggest_copy(cat, tone)
        st.session_state["headline"] = h
        st.session_state["offer"] = o
        st.session_state["cta"] = c

# Row of extra intelligence buttons
col_auto_cat, col_suggest, col_ai = st.columns(3)

with col_ai:
    if not HAS_OPENAI:
        # Button shown but disabled if no API key at all
        st.button("Generate Copy with AI (if available)", disabled=True)
        st.info(
            "AI copy is disabled on this machine (no valid OpenAI API key). "
            "You can still use rule-based copy suggestion above."
        )
    else:
        if st.button("Generate Copy with AI (if available)"):
            try:
                prompt = (
                    f"Write a short 1-line headline, 1-line offer and CTA for a "
                    f"{st.session_state['category']} brand. Tone: "
                    f"{st.session_state['tone']}. Persona: "
                    f"{st.session_state['persona']}."
                )
                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=80,
                )
                text = completion.choices[0].message.content.strip()
                parts = [p.strip() for p in text.split("\n") if p.strip()]

                if len(parts) >= 1:
                    st.session_state["headline"] = parts[0]
                if len(parts) >= 2:
                    st.session_state["offer"] = parts[1]
                if len(parts) >= 3:
                    st.session_state["cta"] = parts[2]

                st.success("AI copy generated.")
            except Exception as e:
                msg = str(e)

                # Clean handling for quota errors
                if "insufficient_quota" in msg or "You exceeded your current quota" in msg:
                    st.warning(
                        "AI copy is temporarily unavailable because the OpenAI API "
                        "key has no remaining quota.\n\n"
                        "You can continue using ‘Generate Suggested Copy from "
                        "Category & Tone’ without AI."
                    )
                else:
                    st.error(
                        "AI copy generation failed due to an API error. "
                        "Please check your API key or try again later."
                    )


# --- Category, Tone, Persona, Objective ---
col_cat, col_tone, col_persona, col_obj = st.columns(4)

with col_cat:
    category_options = ["Biscuits / Snacks", "Beverages", "Other"]
    st.selectbox(
        "Product Category",
        category_options,
        key="category",
    )

with col_tone:
    tone_options = ["Festive", "Discount-focused", "Premium", "Casual / Everyday"]
    st.selectbox(
        "Creative Tone",
        tone_options,
        key="tone",
    )

with col_persona:
    persona_options = ["Family Friendly", "Youthful", "Premium", "Fun & Quirky"]
    st.selectbox(
        "Brand Persona",
        persona_options,
        key="persona",
    )

with col_obj:
    st.selectbox(
        "Campaign Objective",
        OBJECTIVE_OPTIONS,
        key="objective",
        help="Used to adapt the performance score per placement.",
    )

# Text fields
st.text_input("Headline", key="headline")
st.text_input("Offer / Sub-line", key="offer")
st.text_input("CTA (Call To Action)", key="cta")

# Layout & colors
st.session_state["layout_style"] = st.selectbox(
    "Layout preset",
    ["Minimal Light", "Dark Focus", "Hero Zoom", "Soft Pastel"],
    index=["Minimal Light", "Dark Focus", "Hero Zoom", "Soft Pastel"].index(
        st.session_state["layout_style"]
    ),
)

col_bg, col_band = st.columns(2)

with col_bg:
    if logo_img is not None and st.button("Smart Theme from Product + Logo"):
        dom_rgb = get_logo_dominant_color(logo_img)
        st.session_state["bg"] = rgb_to_hex(dom_rgb)
        darker = (
            max(dom_rgb[0] - 20, 0),
            max(dom_rgb[1] - 20, 0),
            max(dom_rgb[2] - 20, 0),
        )
        st.session_state["band"] = rgb_to_hex(darker)

    st.color_picker("Background Color", st.session_state["bg"], key="bg")

with col_band:
    st.color_picker("Band Color", st.session_state["band"], key="band")

bg_hex = st.session_state["bg"]
band_hex = st.session_state["band"]
bg_rgb = hex_to_rgb(bg_hex)
band_rgb = hex_to_rgb(band_hex)
text_color = get_text_contrast(band_rgb)

# -------------------------------
# STEP 3 – Auto Creative Generation
# -------------------------------
st.markdown("### Step 3 – Auto Creative Generation")

if (original_product is None) or (logo_img is None):
    st.info("Please upload **both** product image and brand logo to generate creatives.")
    st.stop()

product_for_layout = product_no_bg if product_no_bg is not None else original_product
layout_style = st.session_state["layout_style"]

(
    tab_feed,
    tab_story,
    tab_fb,
    tab_banner,
    tab_gdn,
    tab_variants,
) = st.tabs(
    [
        "Instagram Feed 1080×1080",
        "Instagram Story 1080×1920",
        "Facebook 1200×628",
        "Tesco Retail Banner 1200×400",
        "Google Display 728×90",
        "Creative Variations (A/B)",
    ]
)

# --------- FEED ----------
with tab_feed:
    st.subheader("Instagram Feed Creative (1080×1080)")

    feed_img, feed_meta = build_creative(
        product_for_layout,
        logo_img,
        canvas_size=(1080, 1080),
        background_color=bg_hex,
        band_color=band_hex,
        layout_style=layout_style,
        headline=st.session_state["headline"],
        offer_text=st.session_state["offer"],
        cta_text=st.session_state["cta"],
        text_color=text_color,
        auto_crop=True,
        band_ratio=0.23,
    )

    st.image(feed_img, caption="Instagram Feed Creative")

    feed_bytes = image_to_bytes(feed_img, "PNG")
    feed_size_kb = len(feed_bytes) / 1024

    st.download_button(
        label="Download Instagram Feed",
        data=feed_bytes,
        file_name="arcvision_instagram_feed.png",
        mime="image/png",
    )

    # Compliance
    st.markdown("#### Compliance Check – Instagram Feed")
    w, h = feed_img.size
    logo_x, logo_y = feed_meta["logo_pos"]
    logo_w, logo_h = feed_meta["logo_size"]

    safe_margin = int(w * 0.05)
    logo_safe = (
        logo_x >= safe_margin
        and logo_y >= safe_margin
        and logo_x + logo_w <= w - safe_margin
    )
    in_band = feed_meta["band_top"] >= int(h * 0.7)
    under_500kb = feed_size_kb <= 500

    st.markdown(
        f"""
- Resolution: **{w}×{h}** — ✅  
- File size: **{feed_size_kb:.1f} KB** — {'✅' if under_500kb else '⚠️ > 500 KB'}  
- Logo in safe zone (top-right, margin applied) — {'✅' if logo_safe else '⚠️'}  
- Headline & offer inside bottom text band — {'✅' if in_band else '⚠️'}  
"""
    )

    # Aesthetic + performance scores
    aesthetic_score = compute_aesthetic_score(bg_rgb, band_rgb, text_color)
    st.markdown(f"**Aesthetic score (prototype):** {aesthetic_score}/100")

    perf_score = estimate_performance_score(
        aesthetic_score,
        under_500kb,
        logo_safe,
        st.session_state["category"],
        st.session_state["tone"],
        st.session_state["objective"],
        placement="feed",
    )
    st.markdown(
        f"**Estimated social performance score (prototype):** {perf_score}/100"
    )

    # Objective-fit & insights
    band_l = get_luminance(band_rgb)
    text_l = get_luminance(text_color)
    contrast_ok = abs(band_l - text_l) > 0.4

    insights = generate_insights(
        st.session_state["headline"],
        st.session_state["offer"],
        st.session_state["cta"],
        feed_size_kb,
        contrast_ok,
        logo_safe,
        aesthetic_score,
        st.session_state["objective"],
        placement="feed",
    )

    st.markdown("#### Design & Performance Insights")
    for i in insights:
        st.markdown(f"- {i}")

# --------- STORY ----------
with tab_story:
    st.subheader("Instagram Story Creative (1080×1920)")

    story_img, story_meta = build_creative(
        product_for_layout,
        logo_img,
        canvas_size=(1080, 1920),
        background_color=bg_hex,
        band_color=band_hex,
        layout_style=layout_style,
        headline=st.session_state["headline"],
        offer_text=st.session_state["offer"],
        cta_text=st.session_state["cta"],
        text_color=text_color,
        auto_crop=True,
        band_ratio=0.23,
    )

    st.image(story_img, caption="Instagram Story Creative")

    story_bytes = image_to_bytes(story_img, "PNG")

    st.download_button(
        label="Download Instagram Story",
        data=story_bytes,
        file_name="arcvision_instagram_story.png",
        mime="image/png",
    )

    rating, msg = objective_comment(st.session_state["objective"], "story")
    st.markdown(
        f"**Objective fit:** For **{st.session_state['objective']}**, Story is a **{rating} fit**. {msg}"
    )

# --------- FACEBOOK ----------
with tab_fb:
    st.subheader("Facebook Creative (1200×628)")

    fb_img, fb_meta = build_creative(
        product_for_layout,
        logo_img,
        canvas_size=(1200, 628),
        background_color=bg_hex,
        band_color=band_hex,
        layout_style=layout_style,
        headline=st.session_state["headline"],
        offer_text=st.session_state["offer"],
        cta_text=st.session_state["cta"],
        text_color=text_color,
        auto_crop=True,
        band_ratio=0.25,
    )

    st.image(fb_img, caption="Facebook Creative")

    fb_bytes = image_to_bytes(fb_img, "PNG")

    st.download_button(
        label="Download Facebook Creative",
        data=fb_bytes,
        file_name="arcvision_facebook.png",
        mime="image/png",
    )

    rating, msg = objective_comment(st.session_state["objective"], "facebook")
    st.markdown(
        f"**Objective fit:** For **{st.session_state['objective']}**, Facebook is a **{rating} fit**. {msg}"
    )

# --------- TESCO RETAIL BANNER ----------
with tab_banner:
    st.subheader("Tesco Retail Banner (1200×400)")

    banner_img, banner_meta = build_creative(
        product_for_layout,
        logo_img,
        canvas_size=(1200, 400),
        background_color=bg_hex,
        band_color=band_hex,
        layout_style=layout_style,
        headline=st.session_state["headline"],
        offer_text=st.session_state["offer"],
        cta_text=st.session_state["cta"],
        text_color=text_color,
        auto_crop=True,
        band_ratio=0.30,
    )

    st.image(banner_img, caption="Tesco Retail Banner Creative")

    banner_bytes = image_to_bytes(banner_img, "PNG")

    st.download_button(
        label="Download Tesco Retail Banner",
        data=banner_bytes,
        file_name="arcvision_tesco_banner.png",
        mime="image/png",
    )

    # Tesco-specific checklist (simple rules)
    bw, bh = banner_img.size
    is_horizontal = bw > bh
    cta_short = bool(st.session_state["cta"]) and len(st.session_state["cta"]) <= 20
    text_band_low_enough = banner_meta["band_top"] > bh * 0.55

    st.markdown("#### Tesco Retail Readiness Checklist")
    st.markdown(
        f"""
- Banner is horizontal format — {'✅' if is_horizontal else '⚠️ Prefer horizontal layout for Tesco slots'}  
- Single, short CTA (≤ 20 chars) — {'✅' if cta_short else '⚠️ Consider a shorter CTA for clarity'}  
- Text band anchored near bottom — {'✅' if text_band_low_enough else '⚠️ Keep offer and CTA closer to lower third'}  
"""
    )

    rating, msg = objective_comment(st.session_state["objective"], "tesco")
    st.markdown(
        f"**Objective fit:** For **{st.session_state['objective']}**, Tesco banner is a **{rating} fit**. {msg}"
    )

# --------- GOOGLE DISPLAY BANNER ----------
with tab_gdn:
    st.subheader("Google Display Banner (728×90)")

    # Smaller product_scale_override so biscuit is fully visible and not cropped
    gdn_img, gdn_meta = build_creative(
        product_for_layout,
        logo_img,
        canvas_size=(728, 90),
        background_color=bg_hex,
        band_color=band_hex,
        layout_style=layout_style,
        headline=st.session_state["headline"],
        offer_text=st.session_state["offer"],
        cta_text=st.session_state["cta"],
        text_color=text_color,
        auto_crop=True,
        band_ratio=0.45,
        product_scale_override=0.85,
    )

    st.image(gdn_img, caption="Google Display Creative")

    gdn_bytes = image_to_bytes(gdn_img, "PNG")
    gdn_size_kb = len(gdn_bytes) / 1024

    st.download_button(
        label="Download Google Display Banner",
        data=gdn_bytes,
        file_name="arcvision_google_display.png",
        mime="image/png",
    )

    rating, msg = objective_comment(st.session_state["objective"], "gdn")
    st.markdown(
        f"**Objective fit:** For **{st.session_state['objective']}**, Google Display is a **{rating} fit**. {msg}"
    )
    st.markdown(f"_File size: ~{gdn_size_kb:.1f} KB_")

# --------- CREATIVE VARIATIONS (A/B) ----------
with tab_variants:
    st.subheader("Creative Variations – A/B Test Set")

    if st.button("Generate 3 Variations"):
        var_configs = [
            ("Minimal Light", bg_hex, band_hex, "Variant A – Current Theme"),
            ("Hero Zoom", bg_hex, band_hex, "Variant B – Hero Zoom Layout"),
            ("Dark Focus", "#111522", "#181f35", "Variant C – Dark Focus Theme"),
        ]

        cols = st.columns(3)
        for col, (lay, vbg, vband, label) in zip(cols, var_configs):
            vbg_rgb = hex_to_rgb(vbg)
            vband_rgb = hex_to_rgb(vband)
            vtext_color = get_text_contrast(vband_rgb)

            v_img, _ = build_creative(
                product_for_layout,
                logo_img,
                canvas_size=(1080, 1080),
                background_color=vbg,
                band_color=vband,
                layout_style=lay,
                headline=st.session_state["headline"],
                offer_text=st.session_state["offer"],
                cta_text=st.session_state["cta"],
                text_color=vtext_color,
                auto_crop=True,
                band_ratio=0.23,
            )

            v_bytes = image_to_bytes(v_img, "PNG")
            v_size_kb = len(v_bytes) / 1024
            v_aesthetic = compute_aesthetic_score(vbg_rgb, vband_rgb, vtext_color)
            v_perf = estimate_performance_score(
                v_aesthetic,
                v_size_kb <= 500,
                True,  # assume logo still safe in variants
                st.session_state["category"],
                st.session_state["tone"],
                st.session_state["objective"],
                placement="feed",
            )

            with col:
                st.image(v_img, caption=label)
                st.caption(
                    f"Aesthetic: {v_aesthetic}/100 · "
                    f"Estimated performance: {v_perf}/100"
                )
