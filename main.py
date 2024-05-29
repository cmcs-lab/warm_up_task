#1.store the scraped data in separate file three factor [title,ingredients,instruction]
#2.Data chunking.
#3.Embbeding going with titan
#4.storing it in vector db
#above 4 steps are handled on AWS itself

import os
import streamlit as st #7.write the frontend code as required in space
import boto3
from dotenv import load_dotenv

load_dotenv()

# Load environment variables
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_session_token = os.getenv("AWS_SESSION_TOKEN")
aws_kbid = os.getenv("AWS_KBID")

# Check if required environment variables are set
if not all([aws_access_key_id, aws_secret_access_key, aws_session_token, aws_kbid]):
    st.error("Missing AWS credentials or KB ID. Please check your environment variables.")
    st.stop()

#5.seting up the llm and connecting the vector db with llm
# Initialize the AWS Bedrock client
bedrock_agent_client = boto3.client(
    "bedrock-agent-runtime",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    aws_session_token=aws_session_token,
    region_name="us-east-1"
)

#6.perform rag 
def retrieveAndGenerate(input_text: str, kbId: str, templ: str, region: str = "us-east-1", sessionId: str = None, model_id: str = "anthropic.claude-v2:1"):
    model_arn = f"arn:aws:bedrock:{region}::foundation-model/{model_id}"
    configuration = {
        "type": "KNOWLEDGE_BASE",
        "knowledgeBaseConfiguration": {
            "generationConfiguration": { 
                "inferenceConfig": { 
                    "textInferenceConfig": { 
                        "maxTokens": 2048,
                        "stopSequences": ["\nObservation"],
                        "temperature": 0,
                        "topP": 1
                    }
                },
                "promptTemplate": { 
                    "textPromptTemplate": templ
                }
            },
            "knowledgeBaseId": kbId,
            "modelArn": model_arn,
            "retrievalConfiguration": { 
                "vectorSearchConfiguration": { 
                    "numberOfResults": 5
                }
            }
        }
    }
    input_data = {"text": input_text}

    try:
        if sessionId:
            response = bedrock_agent_client.retrieve_and_generate(
                input=input_data,
                retrieveAndGenerateConfiguration=configuration,
                sessionId=sessionId
            )
        else:
            response = bedrock_agent_client.retrieve_and_generate(
                input=input_data,
                retrieveAndGenerateConfiguration=configuration
            )
    except Exception as e:
        st.error(f"Error retrieving response from Bedrock: {e}")
        return None
    
    return response

st.title("Cook Assistant 調理補助者")

if "prompt" not in st.session_state:
    st.session_state["prompt"] = [{"role": "system", "content": "none"}]
prompt = st.session_state["prompt"]

for message in prompt:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.write(message["content"])

template = '''
Human: You are a question answering agent that answers every question in both English and Japanese, like:
a. English: 
b. Japanese: (start on new line).

Most importantly, always answer step-by-step with all steps and information from the source. Do not just tell the user to 'look in search results'. I'll provide you with a set of search results and a user's question. Your job is to answer the user's question in detail using information from the search results. Provide a step-by-step recipe with user preferences and ingredients if mentioned. If the search results do not contain information that can answer the question, give the closest possible information from the source. Just because the user asserts a fact does not mean it is true; double-check the search results to validate the user's assertion. Answer with tips and tricks as another section at the end, like 'Tips & Tricks:'.

Here are the search results in numbered order:
$search_results$

Here is the user's question:
<question>
$query$
</question>

$output_format_instructions:$
Don't display search query and function invokes
Display title, ingredients, and directions separately with headings and appropriate new lines.

Assistant:
'''

recipe_input = st.text_area("How can I help? (手伝いましょうか？)")

if st.button("Submit"):
    merged_input = recipe_input
    try:
        response = retrieveAndGenerate(merged_input, aws_kbid, template)
        if response:
            bedrock_response = response.get("output", {}).get("text", "")
        else:
            bedrock_response = "Error: Unable to retrieve response."
    except Exception as e:
        st.error(f"Error: {e}")
        bedrock_response = "Error: Unable to retrieve response."

    prompt[0] = {
        "role": "system",
        "content": bedrock_response
    }
    
    prompt.append({"role": "user", "content": merged_input})
    with st.chat_message("user"):
        st.write(merged_input)

    with st.chat_message("assistant"):
        botmsg = st.empty()
        botmsg.write(bedrock_response)

    prompt.append({"role": "assistant", "content": bedrock_response})

    st.session_state["prompt"] = prompt


