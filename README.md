# 🩻 RadVision AI 

**RadVision AI** is an advanced full-stack medical application designed to automate and enhance clinical MRI knee analysis. Using a state-of-the-art ensemble of Vision Transformers (ViT), Topological Data Analysis (TDA), and Natural Language Generation via local LLMs, RadVision generates professional-grade, privacy-first radiology reports.

---

## ✨ Key Features

- **AI-Powered Pathology Detection:** Predicts overall knee abnormality, ACL tears, and Meniscal tears using a Random Forest classifier.
- **Deep Visual Architecture:** Utilizes *Vision Transformers (ViT)* to extract high-level semantic embeddings from multiparametric MRI slices.
- **Geometric Complexity Analysis:** Applies *Topological Data Analysis (TDA)* to capture spatial anomalies, structural loops ($H_1$ persistence), and persistence entropy inside the knee joint.
- **Clinical Explainability:** Generates *Grad-CAM* attention heatmaps overlaid on the original MRI scans, empowering clinicians to visually verify the model's focus.
- **Generative AI Reporting:** Translates ML predictions and TDA matrices into fluid, expert-level clinical findings utilizing local LLMs (via Ollama).
- **100% Data Privacy:** Generates all clinical texts and PDF reports entirely on offline, local hardware to strictly maintain patient data privacy constraints.

## 🏗️ Tech Stack

- **Frontend:** React + Vite
- **Backend Core:** Python, Flask, Werkzeug
- **AI / ML Pipeline:** PyTorch, Scikit-Learn, Transformers (Hugging Face)
- **Topological Computing:** Ripser, Persim, Scikit-Image
- **Local Generative AI:** Ollama (LLaMA 3)
- **Database:** MongoDB 

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have the following installed to run the application locally:
- [Node.js](https://nodejs.org/en/)
- [Python 3.9+](https://www.python.org/)
- [Ollama](https://ollama.com/)
- [MongoDB](https://www.mongodb.com/try/download/community) (Running locally or via Atlas)

### 2. Running Local AI (Ollama)
RadVision AI uses a local LLM to preserve privacy while generating clinical reports. 
```bash
# Start the local Ollama service with LLaMA 3
ollama run llama3
```

### 3. Backend Setup
```bash
# Navigate to project root
cd Radvision

# Install all required Python dependencies
pip install -r requirements.txt

# Start the Flask Backend Server (Inside BACKEND folder)
cd BACKEND
python3 app.py
```

### 4. Frontend Setup
```bash
# Open a new terminal and navigate to the frontend directory
cd Radvision/Frontend

# Install node modules
npm install

# Start the Vite development server
npm run dev
```

---

## 🩺 System Workflow

1. **Upload:** A clinician uploads knee MRI `.npy` volume files (Axial, Coronal, Sagittal) to the interface.
2. **Preprocessing:** The backend normalizes the volumes and aligns geometric slices.
3. **Feature Extraction:** Both ViT and TDA pipelines extract dense embeddings representing visual textures and geometric structures respectively.
4. **Classification:** A Random Forest model classifies the arrays to determine positive/negative probabilities for Abnormality, ACL Tear, and Meniscal Tear.
5. **Report Generation:** Values are fed into a smart prompt locally inside `Llama 3`, which subsequently dictates a highly accurate clinical report mimicking a radiologist's cadence.
6. **PDF Finalization:** A 2-page appendix containing the diagnostic text, Grad-CAM heatmaps, and TDA persistence diagrams is instantly compiled and displayed to the user.

---

## 🔒 Disclaimer
*This application is an AI-assisted tool intended for educational or supplementary analytical purposes. The generated clinical reports and heatmaps must always be reviewed and confirmed by a certified radiologist prior to clinical integration.*
