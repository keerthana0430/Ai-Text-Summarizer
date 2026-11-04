import streamlit as st
from transformers import pipeline
st.set_page_config(page_title="AI Text Summarizer", layout="centered")
st.markdown("""
<style>
.stApp {
    background-color: #E6F4FF;
}

h1 {
    color: #174A70;
}

p, label {
    color: #263746;
}

.stTextArea textarea {
    background-color: #FFFFFF;
    color: #222222;
    border-radius: 10px;
}

.stButton button {
    background-color: #3498DB;
    color: white;
    border-radius: 8px;
    border: none;
}

.stButton button:hover {
    background-color: #2176AE;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.title("📝 AI Text Summarizer")
st.write("Summarize long paragraphs using Hugging Face AI.")

@st.cache_resource
def load_model():
    return pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")

text = st.text_area("Enter your text:", height=200)

if st.button("Generate Summary"):
    if text.strip():
        with st.spinner("Generating summary..."):
            model = load_model()
            messages = [
                {"role": "user", "content": "Summarize this text in simple English:\n" + text}
            ]
            result = model(messages, max_new_tokens=100, do_sample=False)
            summary = result[0]["generated_text"][-1]["content"]

        st.subheader("Summary")
        st.success(summary)
        st.caption(f"Original words: {len(text.split())} | Summary words: {len(summary.split())}")
    else:
        st.warning("Please enter some text.")
