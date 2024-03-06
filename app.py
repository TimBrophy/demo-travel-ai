import streamlit as st
from elasticsearch import Elasticsearch
import os
import json
from langchain.chat_models import AzureChatOpenAI, BedrockChat
import boto3
import nltk
from nltk.tokenize import word_tokenize
from langchain.schema import (
    SystemMessage,
    HumanMessage
)
from PIL import Image
import pandas as pd
import time
import uuid
from datetime import timezone, datetime

st.session_state.llm = "azure"
transformer_model = '.elser_model_2_linux-x86_64'
search_index = 'search-travel-info'
logging_index = 'llm_interactions'
logging_pipeline = 'ml-inference-llm_interactions'

if st.session_state.llm == 'aws':
    st.session_state.llm_model = os.environ['aws_model_id']
elif st.session_state.llm == 'azure':
    st.session_state.llm_model = os.environ['openai_api_model']
    BASE_URL = os.environ['openai_api_base']
    API_KEY = os.environ['openai_api_key']
    DEPLOYMENT_NAME = os.environ['DEPLOYMENT_NAME']

es = Elasticsearch(os.environ['elastic_url'], api_key=os.environ['elastic_api_key'])

def log_llm_interaction(question, prompt, response, sent_time, received_time, answer_type):
    log_id = uuid.uuid4()
    dt_latency = received_time - sent_time
    actual_latency = dt_latency.total_seconds()
    body = {
        "@timestamp": datetime.now(tz=timezone.utc),
        "question": question,
        "answer": response,
        "provider": st.session_state.llm,
        "model": st.session_state.llm_model,
        "timestamp_sent": sent_time,
        "timestamp_received": received_time,
        "answer_type": answer_type,
        "llm_latency": actual_latency

    }
    response = es.index(index=logging_index, id=log_id, document=body)
    return

def check_qa_log(question):
    model_id = transformer_model
    query = {
                "match": {
                    "question": question
                }
            }

    cache_results = es.search(index=logging_index, query=query, size=1)
    if cache_results['hits']['total']['value'] > 0:
        answer_value = cache_results['hits']['hits'][0]['_source']['answer']
    else:
        answer_value = 0
    return answer_value

def init_chat_model(llm_type):
    if llm_type == 'azure':
        llm = AzureChatOpenAI(
            openai_api_base=BASE_URL,
            openai_api_version=os.environ['openai_api_version'],
            deployment_name=DEPLOYMENT_NAME,
            openai_api_key=API_KEY,
            openai_api_type="azure",
            temperature=2
        )
    elif llm_type == 'aws':
        bedrock_client = boto3.client(service_name="bedrock-runtime", region_name=os.environ['aws_region'],
                                      aws_access_key_id=os.environ['aws_access_key'],
                                      aws_secret_access_key=os.environ['aws_secret_key'])
        llm = BedrockChat(
            client=bedrock_client,
            model_id=os.environ['aws_model_id'],
            streaming=True,
            model_kwargs={"temperature": 1})
    return llm


def truncate_text(text, max_tokens):
    nltk.download('punkt')
    tokens = word_tokenize(text)
    trimmed_text = ' '.join(tokens[:max_tokens])
    return trimmed_text


def content_search(index, question):
    model_id = transformer_model
    query = {
        "bool": {
            "should": [
                {
                    "text_expansion": {
                        "ml.inference.title_expanded.predicted_value": {
                            "model_id": model_id,
                            "model_text": question,
                            "boost": 5
                        }
                    }
                },
                {
                    "text_expansion": {
                        "ml.inference.body_content_expanded.predicted_value": {
                            "model_id": model_id,
                            "model_text": question
                        }
                    }
                },
                {
                    "match": {
                        "body_content": question
                    }
                },
                {
                    "match": {
                        "title": question
                    }
                }
            ]
        }
    }

    field_list = ['title', 'body_content', '_score']
    results = es.search(index=index, query=query, size=100, fields=field_list, min_score=10)
    response_data = [{"_score": hit["_score"], **hit["_source"]} for hit in results["hits"]["hits"]]
    documents = []
    # Check if there are hits
    if "hits" in results and "total" in results["hits"]:
        total_hits = results["hits"]["total"]
        # Check if there are any hits with a value greater than 0
        if isinstance(total_hits, dict) and "value" in total_hits and total_hits["value"] > 0:
            for hit in response_data:
                doc_data = {field: hit[field] for field in field_list if field in hit}
                documents.append(doc_data)
    return documents


def construct_prompt(question, results):
    for record in results:
        if "_score" in record:
            del record["_score"]
    result = ""
    for item in results:
        result += f"Title: {item['title']} , Body: {item['body_content']}\n"
    reduced_string_results = truncate_text(result, 10000)
    # interact with the LLM
    augmented_prompt = f"""Using only the context below, answer the query.
    Context: {reduced_string_results}
    Query: {question}"""
    messages = [
        SystemMessage(
            content="You are a helpful analyst that answers questions based only on the context provided. "
                    "When you respond, please cite your source and where possible, always summarise your answers."),
        HumanMessage(content=augmented_prompt)
    ]
    return messages

# search form
image = Image.open('images/logo_1.png')
st.image(image, width=120)
st.title("Travel assistant")
st.session_state.llm = st.selectbox("Choose your LLM", ["azure", "aws"])
question = st.text_input("Question", placeholder="What would you like to know?")
submitted = st.button("search")

if submitted:
    existing_answer = check_qa_log(question)
    # existing_answer = 0
    results = content_search(search_index, question)
    df_results = pd.DataFrame(results)
    with st.status("Searching the data...") as status:
        status.update(label=f'Retrieved {len(results)} results from Elasticsearch', state="running")
    with st.chat_message("ai travel assistant", avatar="🤖"):
        full_response = ""
        message_placeholder = st.empty()
        sent_time = datetime.now(tz=timezone.utc)
        prompt = construct_prompt(question, results)
        if existing_answer == 0:
            chat_model = init_chat_model(st.session_state.llm)
            current_chat_message = chat_model(prompt).content
            answer_type = 'original'
        else:
            current_chat_message = existing_answer
            answer_type = 'existing'
        # current_chat_message = chat_model(prompt).content
        for chunk in current_chat_message.split():
            full_response += chunk + " "
            time.sleep(0.02)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
        received_time = datetime.now(tz=timezone.utc)
        string_prompt = str(prompt)
        log_llm_interaction(question, string_prompt, current_chat_message, sent_time, received_time,
                            answer_type)
        status.update(label="AI response complete!", state="complete")
    st.dataframe(df_results)
