from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

def summarize_text(text, max_length=150):
    input_text = "summarize: " + text
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=40, length_penalty=2.0, num_beams=4)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# save this as app.py

import streamlit as st

st.title("📝 Simple Text Summarizer")

text_input = st.text_area("Enter Your Text: ")

if st.button("Summarize"):
    if text_input:
        summary = summarize_text(text_input)
        st.subheader("Summarized")
        st.write(summary)
    else:
        st.warning("Please Enter Your Text.")
