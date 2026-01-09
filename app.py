import streamlit as st
import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import time

# --- 1. APP CONFIGURATION ---
st.set_page_config(page_title="LinguaBridge Portal", page_icon="🌐", layout="wide")

# --- 2. CUSTOM CSS FOR FLOATING ACTION BUTTON (FAB) ---
st.markdown(
    """
<style>
    /* This CSS targets a specific button type. 
       We will set the FAB to type="primary" to apply these styles.
    */
    div.st-key-fab_main button[kind="secondary"] {
        position: fixed;
        bottom: 30px;
        right: 30px;
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background-color: #FF4B4B;
        color: white; 
        font-size: 30px;
        border: none;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.3);
        z-index: 9999;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: transform 0.2s;
    }
    
    /* Hover effect */
     div.st-key-fab_main button[kind="secondary"]:hover {
        transform: scale(1.1);
        background-color: #FF2B2B;
    }
    
    /* Hide the text inside the button if needed, or style it */
     div.st-key-fab_main button[kind="secondary"] p {
        font-size: 24px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# --- 3. LOAD AI MODEL (Cached) ---
@st.cache_resource
def load_model():
    model_name = "facebook/m2m100_418M"
    # Using spinner to show progress on first load
    with st.spinner(f"Loading Neural Machine Translation Model ({model_name})..."):
        tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model


try:
    tokenizer, model = load_model()
except Exception as e:
    st.error(
        "⚠️ Error loading model. Please check your internet connection to download M2M100."
    )
    st.stop()

# --- 4. SESSION STATE MANAGEMENT ---
if "page" not in st.session_state:
    st.session_state.page = "Translator"  # Default Home Page


# --- 5. CHATBOT DIALOG LOGIC ---
@st.dialog("🤖 Project Assistant")
def show_chatbot():
    st.caption("I can guide you through this project portal.")

    # Initialize chat history inside the dialog
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": "Hello! Ask me about the project, the accuracy, or the team.",
            }
        ]

    # Display Chat History
    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).write(msg["content"])

    # Chat Input Handler
    if prompt := st.chat_input("Type your question..."):
        # 1. Show User Message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # 2. Determine Response (Rule-Based Logic)
        response_text = ""
        redirect_target = None
        q = prompt.lower()

        if "project" in q or "about" in q:
            response_text = "This is a Group Project for NLP (BAXI 3413). We built a 'Direct-to-Direct' translation system using M2M100."
            redirect_target = "Project Info"
        elif "technique" in q or "architecture" in q or "model" in q:
            response_text = "We use the Transformer Seq2Seq architecture with SentencePiece tokenization. I'll take you to the technical details."
            redirect_target = "Architecture"
        elif "accurate" in q or "accuracy" in q or "bleu" in q:
            response_text = "We evaluate our model using BLEU scores. We use Beam Search (k=5) to improve translation quality."
            redirect_target = "Architecture"
        elif "team" in q or "member" in q:
            response_text = "We are a team of 4 students from UTeM. You can see our names in the Project Info section."
            redirect_target = "Project Info"
        elif "translate" in q or "try" in q:
            response_text = "Sure! I'll take you to the main translation tool."
            redirect_target = "Translator"
        else:
            response_text = "I can help navigate the portal. Try asking: 'How does it work?' or 'Who made this?'"

        # 3. Show Bot Response
        st.session_state.chat_history.append(
            {"role": "assistant", "content": response_text}
        )
        st.chat_message("assistant").write(response_text)

        # 4. Handle Page Redirection
        if redirect_target and redirect_target != st.session_state.page:
            time.sleep(1.0)  # Pause so user can read
            st.session_state.page = redirect_target
            st.rerun()  # Refresh app to switch page


# --- 6. RENDER FLOATING BUTTON ---
# This button triggers the dialog defined above
if st.button("💬", key="fab_main"):
    show_chatbot()

# --- 7. SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
# Sync sidebar selection with session_state.page
selection = st.sidebar.radio(
    "Go to:",
    ["Translator", "Architecture", "Project Info"],
    index=["Translator", "Architecture", "Project Info"].index(st.session_state.page),
)

if selection != st.session_state.page:
    st.session_state.page = selection
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.info("Tip: Click the 💬 button in the bottom-right for help!")

# --- 8. PAGE CONTENT ---

# === PAGE 1: TRANSLATOR (Main Tool) ===
if st.session_state.page == "Translator":
    st.title("🌐 LinguaBridge: Universal Translator")
    st.markdown("### Neural Machine Translation Demo")
    st.markdown(
        "This tool uses **M2M100** to translate directly between languages without using English as a pivot."
    )

    col1, col2 = st.columns(2)

    # Language Code Mapping for M2M100
    lang_map = {
        "English": "en",
        "Malay": "ms",
        "Chinese": "zh",
        "Japanese": "ja",
        "French": "fr",
        "Spanish": "es",
    }

    with col1:
        src_name = st.selectbox("From:", list(lang_map.keys()), index=0)
    with col2:
        tgt_name = st.selectbox("To:", list(lang_map.keys()), index=1)

    src_lang = lang_map[src_name]
    tgt_lang = lang_map[tgt_name]

    source_text = st.text_area(
        "Enter Text:", height=150, placeholder="Type your sentence here..."
    )

    if st.button("Translate", type="primary"):
        if source_text:
            start_time = time.time()
            with st.spinner("Processing..."):
                # 1. Tokenize (Technique: SentencePiece)
                tokenizer.src_lang = src_lang
                encoded_input = tokenizer(source_text, return_tensors="pt")

                # 2. Generate (Technique: Beam Search + Forced BOS Token)
                generated_tokens = model.generate(
                    **encoded_input,
                    forced_bos_token_id=tokenizer.get_lang_id(tgt_lang),
                    num_beams=5,  # Higher beams = better quality, slower speed
                    max_length=200,
                )

                # 3. Decode
                result = tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )[0]
                end_time = time.time()

            st.success("Translation Complete")
            st.markdown(f"#### {result}")
            st.caption(f"Inference time: {round(end_time - start_time, 2)} seconds")
        else:
            st.warning("Please enter text to translate.")

# === PAGE 2: ARCHITECTURE (Documentation) ===
elif st.session_state.page == "Architecture":
    st.title("🛠️ System Architecture")

    st.markdown("### 1. The Model: M2M100")
    st.info(
        "We utilize the **Facebook M2M100 (418M)** model, a massive Many-to-Many neural machine translation model trained on 2,200 language directions."
    )

    st.markdown("### 2. Technical Implementation")

    st.markdown("""
    #### A. Transformer Seq2Seq
    The core architecture is an **Encoder-Decoder Transformer**.
    * **Encoder:** Processes the input source text into context vectors.
    * **Decoder:** Generates the target text token-by-token.
    """)

    # Placeholder for Architecture Diagram
    st.image(
        "https://jalammar.github.io/images/t5/t5-enc-dec-attention.png",
        caption="Figure 1: Standard Encoder-Decoder Attention Mechanism",
    )

    st.markdown("""
    #### B. Forced Decoding Strategy
    Unlike standard models, we must explicitly tell the decoder which language to speak.
    * We inject a **Language Token** (e.g., `__ms__`) at the start of the generation process.
    
    #### C. Optimization
    * **Tokenization:** SentencePiece (Unigram) to handle 100+ languages with a shared vocabulary.
    * **Decoding:** Beam Search (k=5) to find the most probable sentence sequence.
    """)

# === PAGE 3: PROJECT INFO ===
elif st.session_state.page == "Project Info":
    st.title("ℹ️ Project Information")

    st.subheader("BAXI 3413 - Natural Language Processing")
    st.write("Group Project: Intelligent Website Portal")

    st.markdown("---")
    st.subheader("👥 Group Members")

    cols = st.columns(4)
    members = [
        {"name": "Student A", "role": "Model Engineer"},
        {"name": "Student B", "role": "Frontend Dev"},
        {"name": "Student C", "role": "Data Analyst"},
        {"name": "Student D", "role": "Documentation"},
    ]

    for i, col in enumerate(cols):
        with col:
            st.info(f"**{members[i]['name']}**\n\n{members[i]['role']}")

    st.markdown("---")
    st.subheader("Project Objectives")
    st.markdown("""
    1. Develop a web portal for NLP applications.
    2. Implement a working Chatbot for user guidance.
    3. Demonstrate commercial potential through a polished UI.
    """)
