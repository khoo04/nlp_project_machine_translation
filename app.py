import streamlit as st
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import time

# --- 1. APP CONFIGURATION ---
st.set_page_config(
    page_title="M2M100 Translator Tool",
    page_icon=":material/translate:",
    layout="wide",
)


# --- 2. LOAD AI MODEL (Cached) ---
@st.cache_resource
def load_model():
    model_name = "facebook/m2m100_418M"
    with st.spinner(f"Loading Neural Machine Translation Model ({model_name})..."):
        tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model


try:
    tokenizer, model = load_model()
except Exception as e:
    st.error("⚠️ Error loading model. Please check your internet connection.")
    st.stop()


# --- 3. DEFINE PAGE FUNCTIONS ---
# --- Translator Page ---
def translator():
    st.title(":material/translate: Translator Tool")
    st.info("Direct-to-Direct translation using M2M100")

    c1, c2 = st.columns(2)
    lang_map = {
        "English": "en",
        "Malay": "ms",
        "Chinese": "zh",
        "Japanese": "ja",
        "French": "fr",
        "Spanish": "es",
    }

    with c1:
        src_name = st.selectbox("From:", list(lang_map.keys()), index=0)
    with c2:
        tgt_name = st.selectbox("To:", list(lang_map.keys()), index=1)

    src_lang = lang_map[src_name]
    tgt_lang = lang_map[tgt_name]

    source_text = st.text_area("Enter Text To Translate:", height=150)

    if st.button("Translate", type="primary"):
        if source_text:
            start_time = time.time()
            with st.spinner("Processing..."):
                tokenizer.src_lang = src_lang
                encoded_input = tokenizer(source_text, return_tensors="pt")
                generated_tokens = model.generate(
                    **encoded_input,
                    forced_bos_token_id=tokenizer.get_lang_id(tgt_lang),
                    num_beams=5,
                    max_length=200,
                )
                result = tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )[0]
                end_time = time.time()

            st.success("Translation Result")
            st.markdown(f"#### {result}")
            st.caption(f"Time taken : {round(end_time - start_time, 2)}s")
        else:
            st.warning("Please enter text.")


# --- Architecture Page ---
def architecture():
    st.title("🛠️ System Architecture")
    st.markdown("### 1. Model Overview")
    st.markdown(
        'The M2M100 (Many-to-Many 100) model, developed by Facebook AI (now Meta AI), is the first multilingual machine translation model that can translate directly between any pair of 100 languages without relying on English as a "pivot" or intermediary language.'
    )
    st.markdown("""
    - Pre-trained on 7.5 billion tokens across 100 languages.
    - Direct translation between any pair of languages without relying on English as an intermediary.
    - Based on the Transformer architecture with 12 encoder and 12 decoder layers.
    """)

    st.markdown("### 2. Model Architecture - Transformer Seq2Seq")
    st.markdown("""
M2M100 is built on the standard Transformer Sequence-to-Sequence (Seq2Seq) architecture, which consists of two main stacks:
- The Encoder:
    - Takes the source text (tokenized) and processes it into a context vector.
    - It uses Self-Attention mechanisms to understand the relationship between words in the source sentence, regardless of their distance from each other.

- The Decoder:
    - Takes the encoder's context vector and generates the translated text one token at a time.
    - It uses Cross-Attention to focus on relevant parts of the source sentence while generating each word of the target sentence.
    """)
    st.markdown("### 3. Mechanisms")
    st.markdown("""
A. Direct "Many-to-Many" Translation:  
   - Traditional Models (English-Centric): To translate Chinese to French, older systems would translate Chinese :material/arrow_right_alt: English :material/arrow_right_alt: French. This accumulation of errors is called the "pivot effect".
   - "M2M100 Strategy: It translates Chinese $\rightarrow$ French directly. It was trained on a massive dataset of 7.5 billion sentence pairs covering thousands of non-English directions. 

B. Language-Specific Tokens (Forced Decoding)  
Since the model is shared across 100 languages, it needs to know what language to speak.
- Encoder Side: The source text is prefixed with a special language token, for example, __zh__ for Chinese, so the encoder knows the input grammar/vocabulary.
- Decoder Side: The first token generated is forced to be the target language ID (e.g., __fr__). This signals the decoder to output text in French.  

C. Universal Tokenization
- It uses **SentencePiece** to break text into subwords.
- It employs a shared vocabulary of 128,000 tokens that covers all 100 supported languages, ensuring script diversity (e.g., Latin, Cyrillic, Chinese characters) is handled in one embedding space.                """)

    col1, col2 = st.columns(2, vertical_alignment="bottom")

    with col1:
        st.image(
            "screenshots/image_english_centric.png",
            caption="English-Centric Multilingual",
            use_container_width=True,
        )

    with col2:
        st.image(
            "screenshots/image_m2m100.png",
            caption="M2M-100: Many-to-Many Multilingual Mode",
            use_container_width=True,
        )

    st.markdown("### 4. Reason to choose M2M100")
    st.markdown("""
- **No Bias**: It preserves meaning better for non-English languages (e.g., translating Malay $\rightarrow$ Chinese directly is more accurate than going through English).
- **Simplicity**: You only need to load one model to handle 9,900 different translation directions ($100 \times 99$), rather than loading separate models for each language pair.
                """)


# --- Project Info Page ---
def info():
    st.title("ℹ️ Project Information")
    st.subheader("BAXI 3413 - NLP Group Project - Machine Translation")
    st.markdown("---")
    st.subheader("👥 Team Members")

    members = [
        "KHOO ZHEN XIAN",
        "AQEM ZAKWAN BIN AHMAD",
        "FARIZ DANISH BIN FADLI",
        "AHMAD MIRZA SHAHMI BIN ABDUL HANIF",
    ]
    cols = st.columns(2)
    for i, member in enumerate(members):
        with cols[i % 2]:
            st.success(member)

    st.markdown("---")

    # Commercial Value
    st.subheader("💼 Commercial Value & Potential")
    st.markdown("""
    This multilingual translation system has strong potential for commercialization due to its
    ability to perform direct many-to-many translations without relying on English as an intermediary.

    **Potential Commercial Applications:**
    - 🌍 Multilingual customer support portals for global businesses
    - 🎓 Educational platforms supporting multilingual learning content
    - 🏛️ Government and public service translation tools
    - 📱 Integration into chatbots and virtual assistants for international users

    **Business Value:**
    - Reduces dependency on multiple translation models
    - Improves translation accuracy for non-English languages
    - Scalable architecture that supports future language expansion
    - Can be deployed as a Software-as-a-Service (SaaS) solution or API-based service
    """)


# --- 4. SETUP NAVIGATION ---
pages = [
    st.Page(translator, title="Translator", icon=":material/translate:", default=True),
    st.Page(architecture, title="Architecture", icon="🛠️"),
    st.Page(info, title="Project Info", icon="ℹ️"),
]

page_dict = {page.title: page for page in pages}

pg = st.navigation(pages)


# --- 5. DEFINE LAYOUT (Split Screen) ---
main_col, chat_col = st.columns([0.75, 0.25], gap="medium")


# --- 6. RUN NAVIGATION IN LEFT COLUMN ---
with main_col:
    pg.run()


# --- 7. CHATBOT LOGIC (RIGHT PANEL) ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": "Hello! I can guide you through this project portal. Ask me about the project, architecture, or translation tool.",
        }
    ]

with chat_col:
    st.subheader(":material/smart_toy: Chatbot Assistant")
    chat_container = st.container(height=500, border=True)

    with chat_container:
        for msg in st.session_state.chat_history:
            st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask about the project..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with chat_container:
            st.chat_message("user").write(prompt)

        # Response Logic
        response_text = ""
        target_page_title = None  # Store the Title (string) here
        q = prompt.lower()

        # 1. Project Overview
        if any(k in q for k in ["project", "about", "website", "this system"]):
            response_text = (
                "This is an NLP group project focused on multilingual machine translation "
                "using the M2M100 model."
            )
            target_page_title = "Project Info"

        # 2. Team Members
        elif any(k in q for k in ["team", "members", "group", "developer"]):
            response_text = "This project was developed by four students as part of the BAXI 3413 course."
            target_page_title = "Project Info"

        # 3. Translation Usage
        elif any(k in q for k in ["translate", "translation", "translator", "convert"]):
            response_text = "You can translate text between multiple languages using our translator tool."
            target_page_title = "Translator"

        # 4. Supported Languages
        elif any(k in q for k in ["language", "languages", "supported"]):
            response_text = "The system supports multiple languages such as English, Malay, Chinese, Japanese, French, and Spanish."
            target_page_title = "Translator"

        # 5. Model Used
        elif any(k in q for k in ["model", "ai model", "m2m100"]):
            response_text = "This project uses the M2M100 multilingual neural machine translation model."
            target_page_title = "Architecture"

        # 6. System Architecture
        elif any(k in q for k in ["architecture", "system design", "how it works"]):
            response_text = "The system is built using a Transformer-based Seq2Seq architecture with encoder and decoder components."
            target_page_title = "Architecture"

        # 7. Tokenization Method
        elif any(k in q for k in ["token", "tokenization", "sentencepiece"]):
            response_text = "SentencePiece tokenization is used to efficiently handle more than 100 languages using a shared vocabulary."
            target_page_title = "Architecture"

        # 8. Advantages of M2M100
        elif any(k in q for k in ["advantage", "benefit", "why m2m100", "why choose"]):
            response_text = "M2M100 enables direct many-to-many translation without relying on English, which improves translation accuracy for non-English languages."
            target_page_title = "Architecture"

        # 9. Commercial Potential
        elif any(k in q for k in ["commercial", "business", "real world", "industry"]):
            response_text = "This system has potential for commercialization in multilingual customer support, education platforms, and global communication services."
            target_page_title = "Project Info"

        # 10. Help / Navigation
        elif any(k in q for k in ["help", "guide", "what can you do"]):
            response_text = "You can ask me about the project, team members, translation tool, supported languages, system architecture, or AI model."

        # Fallback
        else:
            response_text = "I'm not sure about that yet. Try asking about the project, translation, architecture, or team members."

        st.session_state.chat_history.append(
            {"role": "assistant", "content": response_text}
        )
        with chat_container:
            st.chat_message("assistant").write(response_text)

        if target_page_title:
            # Retrieve the actual st.Page object using the title string
            st.switch_page(page_dict[target_page_title])
