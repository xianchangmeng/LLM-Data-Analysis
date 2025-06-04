import streamlit as st
import pandas as pd
from llama_index.experimental.query_engine.pandas import PandasQueryEngine
from llama_index.llms.huggingface import HuggingFaceLLM



@st.cache_resource
def load_llm():
    # Use our own local model, downloaded from Huggingface
    model_name = "./Mistral-7B-Instruct"

    return HuggingFaceLLM(
        model_name=model_name,
        tokenizer_name=model_name,
        context_window=4096,
        max_new_tokens=512,
        device_map='mps',
        generate_kwargs={"temperature": 0.1},
        model_kwargs={
            "torch_dtype": "auto"        # or use torch.float16 if supported
        }
    )


llm = load_llm()


# Streamlit UI
st.title("📊 LLM Data Analysis")

uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format.")
            st.stop()

        st.success("File uploaded successfully!")
        
        # Data preview
        st.subheader("👀 Data Preview")
        st.dataframe(df.head())

        # Initialize query engine
        query_engine = PandasQueryEngine(df=df, llm=llm, verbose=True)

        # Maintain question history in session state
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        question = st.text_input("Ask a question about your data:")

        if st.button("Ask"):
            if question:
                with st.spinner("Thinking..."):
                    response = query_engine.query(question)
                    st.session_state.chat_history.append((question, response.response))
            else:
                st.warning("Please enter a question.")

        # Show previous Q&A
        if st.session_state.chat_history:
            st.markdown("---")
            st.subheader("💬 Conversation History")
            for q, a in reversed(st.session_state.chat_history):
                st.markdown(f"**Q:** {q}")
                st.markdown(f"**A:** {a}")

            # Optional: try to plot the latest result
            raw_output = response.metadata.get("raw_pandas_output", "") if "response" in locals() else ""
            if raw_output:
                try:
                    if "Name:" in raw_output:
                        result_series = pd.read_json(raw_output, typ="series")
                        st.line_chart(result_series)
                    else:
                        result_df = pd.read_json(raw_output, typ="frame")
                        st.line_chart(result_df)
                except Exception:
                    st.info("No plottable data returned or could not parse output.")

    except Exception as e:
        st.error(f"Failed to read file: {e}")
else:
    st.info("Upload a CSV or Excel file to begin.")
