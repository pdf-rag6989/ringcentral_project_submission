from openai import OpenAI
import streamlit as st
from retriever import DataRetriver
from generation import Results_Generation

generation = Results_Generation()
searcher = DataRetriver()

st.title("Car Manual Chat Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

st.sidebar.title("Car Manual Chat Assistant")
st.sidebar.markdown(
    """
    A **conversational assistant** designed to answer your car manual-related queries. 
    Simply type your query, and the assistant will fetch the relevant details from the manuals.

    **Supported Manuals**:
    - MG Astor
    - Tata Tiago

    **Sample Queries**:
    - "How to turn on the indicator in MG Astor?"
    - "What is the recommended tire pressure for Tata Tiago?"
    """
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about your car manual:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        response_placeholder.markdown("Processing...")

        user_query = prompt
        entity = generation.extract_entities(user_query)
        if entity['car_model_name'] == 'Not_in_scope':
            final_result = "The requested car manual is not currently in scope."
        elif entity['car_model_name'] is None:
            if entity['image_url'] is None:
                information_extracted = searcher.retrieve_similar_texts(user_query)
            else:
                image_path = entity['image_url']
                information_extracted = searcher.retrieve_similar_images(image_path)
            updated_prompt = generation.final_prompt.format(
                user_query=user_query, general_info_requested='yes', context=information_extracted
            )
            message_text = [{"role": "system", "content": updated_prompt}]
            final_result = generation.base_gpt4_model(message_text)
        else:
            if entity['image_url'] is None:
                information_extracted = searcher.retrieve_similar_texts(
                    user_query, pdf_name_filter=entity['car_model_name']
                )
            else:
                image_path = entity['image_url']
                information_extracted = searcher.retrieve_similar_images(
                    image_path, pdf_name_filter=entity['car_model_name']
                )
            updated_prompt = generation.final_prompt.format(
                user_query=user_query, general_info_requested='no', context=information_extracted
            )
            message_text = [{"role": "system", "content": updated_prompt}]
            final_result = generation.base_gpt4_model(message_text)

        response_placeholder.markdown(final_result)
        st.session_state.messages.append({"role": "assistant", "content": final_result})
