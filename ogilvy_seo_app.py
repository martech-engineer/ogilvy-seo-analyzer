import streamlit as st

# Set this FIRST — before anything else
st.set_page_config(page_title="Ogilvy Copy Evaluator", layout="centered")

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Model configuration
MODEL_ID = "Qwen/Qwen1.5-1.8B-Chat"

@st.cache_resource
def load_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto"
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

pipe = load_pipeline()

def build_prompt(content):
    return f"""<|im_start|>system
You are an advertising strategist trained in David Ogilvy’s 15 principles.

You will be given a piece of marketing copy. Do not provide general commentary. Follow the instructions below exactly and return your response in the specified format.

## Instructions

1. Evaluate the copy using Ogilvy principles.
   - Assign a **total score from 0–100**.
   - **Do not include a breakdown of all 15 principles.**
   - Identify and display only the **3 lowest-scoring principles**. For each, provide:
     - The principle name
     - A score (0–6.7 scale)
     - A short explanation of why it scored low

2. Then suggest **specific edits or improvements** to address the 3 weak principles.

3. Finally, rewrite the full marketing copy to achieve a **perfect score of 100/100**, ensuring all 15 principles are addressed effectively.

## Output Format

---
<|im_end|>
<|im_start|>user
{content}
<|im_end|>
<|im_start|>assistant
"""

st.title("🧠 Ogilvy Copy Evaluator")
st.caption("Evaluate marketing copy with 15 principles from David Ogilvy — get a score, feedback, and rewrite.")

user_input = st.text_area("Paste your marketing copy below:", height=200)

if st.button("Evaluate My Copy"):
    if not user_input.strip():
        st.warning("Please enter some marketing copy.")
    else:
        with st.spinner("Analyzing your copy..."):
            prompt = build_prompt(user_input)
            output = pipe(prompt, max_new_tokens=1000, temperature=0.7)[0]["generated_text"]
            result = output.split("<|im_start|>assistant")[-1].strip()

        st.markdown("### ✍️ Evaluation Result")
        st.markdown(result)
