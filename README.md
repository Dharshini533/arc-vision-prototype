📌 **Demo Video:** See the full prototype walkthrough here → [DEMO.md](DEMO.md)
**ARC-VISION: AI-Powered Retail Creative Generator
Retail Media Innovation Jam — Hackathon Submission
Built by Dharshini Ramesh**

**Overview:**
ARC-VISION is an intelligent retail media creative generation tool that converts a simple product image + brief into multi-platform ready ad creatives within seconds.
It supports automated generation for:
1.Instagram Feed (1080×1080)
2.Instagram Story (1080×1920)
3.Facebook Ad (1200×628)
4.Tesco Retail Banner (1200×400)
5.Google Display Banner (728×90)
The system ensures each creative is correctly sized, branded, readable, and compliant with placement rules.

**Key Features:**
1. Apply Brand Profile
Automatically loads saved brand settings such as tone, persona, and objective.

2. Auto-Fill From Brief
Extracts headline, offer, CTA from user text input.

3. Auto-Detect Category From Product Image
Uses AI to detect the product type and auto-select category.

4. AI-Generated Copy (Optional)
Generates improved copy based on category + tone.
(If AI quota is exceeded, a safe fallback message is shown.)

5. Smart Theme Colors
The system intelligently selects:
Background color
Text band color
based on product and brand consistency.

6. High-Quality Image Scaling
Automatically resizes and positions the product image for each banner size.

7. Objective Fit Rating
Shows creative fit score for Awareness / Performance goals.

8. Compliance & Readiness Checks
Tesco banners get automated readiness validation:
Format check
CTA length
Safe logo placement
Readability checks

9. One-Click Downloads
User can download PNG creatives for every placement.

**Tech Stack:**

Python
Streamlit (UI + interactive workflow)
Pillow (PIL) – image processing
OpenAI API – optional copy generation
NumPy – color extraction
GitHub – version control

**Folder Structure:**

arc-vision-prototype/
│── app.py        
│── requirements.txt (optional)
│── assets/ (optional future folder for icons, sample images)
│── README.md

**Setup Instructions:**

1. **Clone the Repository**
git clone https://github.com/Dharshini533/arc-vision-prototype.git
cd arc-vision-prototype

2. **Install Dependencies**
pip install streamlit pillow numpy openai

3. **Run the App**
streamlit run app.py

The application will open in your browser:
👉 http://localhost:8501/

**How to Use the Application:**

Upload a product image

Click Apply Brand Profile

Click Auto-Fill From Brief (optional)

Click Auto-Detect Category From Image

Modify headline, offer, CTA if needed

See the auto-generated creatives

Click Download for each platform

Review Objective Fit, Readiness Check, and Insights

**Future Extensions:**

Full brand color extraction from product pack

Real-time leaderboard-style creative scoring

Packshot background removal

Multi-brand memory profile

Export-to-PDF creative report

**Author**:
Dharshini Ramesh
AI & Creative Automation Enthusiast.
