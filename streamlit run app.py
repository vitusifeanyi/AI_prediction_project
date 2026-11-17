import streamlit as st
from transformers import pipeline


st.set_page_config(page_title="My AI App")
st.title("SENTIMENT AND WORD GUESSING APP")


# Load each model only once
@st.cache_resource
def load_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )


@st.cache_resource
def load_fill_mask_model():
    return pipeline(
        "fill-mask",
        model="distilbert-base-uncased"
    )


# Create the two tabs
tab1, tab2 = st.tabs(["Sentiment", "Missing word"])


# Tab 1: Sentiment analysis
with tab1:
    st.write("Type a sentence. The model will say if it is POSITIVE or NEGATIVE.")

    text = st.text_area(
        "Your sentence",
        "I love learning about artificial intelligence"
    )

    if st.button("Check sentiment"):
        if not text.strip():
            st.warning("Please type a sentence first.")
        else:
            with st.spinner("Thinking..."):
                result = load_sentiment_model()(text, truncation=True)[0]

            message = f"{result['label']} (score: {round(result['score'], 3)})"

            if result["label"] == "POSITIVE":
                st.success(message)
            else:
                st.error(message)


# Tab 2: Guess the hidden word
with tab2:
    st.write("Enter a sentence with a missing word.")

    text = st.text_input(
        "Your sentence",
        "I love learning about [MASK]"
    )

    if st.button("Guess missing word"):
        if "[MASK]" not in text:
            st.warning("Please include [MASK] in your sentence.")
        else:
            with st.spinner("Guessing..."):
                results = load_fill_mask_model()(text)

            for result in results:
                st.write(
                    result["sequence"],
                    round(result["score"], 3)
                )

