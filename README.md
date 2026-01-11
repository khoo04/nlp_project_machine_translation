# M2M100 Machine Translation Portal

## Project Overview
This project is a web-based multilingual translation portal developed as part of the
**BAXI 3413 – Natural Language Processing** course. The system utilizes the **M2M100**
(Many-to-Many 100) Transformer-based neural machine translation model to perform direct
language-to-language translation without using English as an intermediary.

The portal also includes a chatbot to assist users in navigating the system and understanding the project.

---

## System Features
- Direct multilingual translation using M2M100
- Support for multiple languages (English, Malay, Chinese, Japanese, French, Spanish)
- Informative architecture and project pages
- Chatbot for project-related queries and navigation

---



## Installation Guide

### Software Requirements
- Python 3.13
-


### Step 1: Clone the Project Repository
```bash
git clone <repository-url>
cd m2m100-translator-portal
```

### Step 2: Create a Virtual Environment
```bash
python -m venv .venv
```

### Step 3: Activate the Virtual Environment
#### Windows
```bash
.venv\Scripts\activate
```

#### macOS / Linux
```bash
source .venv/bin/activate
```

### Step 4: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 5: Run the Application
```bash
streamlit run app.py
```
The application will be available at:
```bash
http://localhost:8501
```

---
## Project Structure
```bash
.
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── screenshots/            # Related images
├── references/             # Reference/Article used in the project
├── README.md               # Documentation
└── .venv/                  # Python virtual environment
```