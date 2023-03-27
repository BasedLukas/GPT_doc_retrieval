import streamlit as st
import openai
import tiktoken
import json
from query_data import query, get_gpt_response

# Load config file
with open("config.json") as json_file:
    config = json.load(json_file)

# Set up OpenAI API key
openai.api_key = config["openai_api_key"]


def main():
    st.title("GPT Response Generator")
    st.markdown("Enter your query and context data in the text boxes below:")

    # Input boxes for user query and data
    user_query = st.text_area("User Query:", value="", height=100)
    data_input = st.text_area("Data Input:", value="", height=200)

    # Button to trigger the function to get the response
    if st.button("Get GPT Response"):
        if user_query and data_input:
            userq, system_prompt = query(user_query, data_input)
            gpt_response = get_gpt_response(userq, system_prompt)
            st.markdown("### GPT Response:")
            st.write(gpt_response)
        else:
            st.warning("Please fill in both input boxes before submitting.")

    # Button to clear input boxes
    if st.button("Clear"):
        user_query = ""
        data_input = ""
        st.experimental_rerun()

if __name__ == "__main__":
    main()
