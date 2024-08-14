import streamlit as st
import requests
from inference_quantized import get_reply_finetuned
import json, time

st.title("AI Pharmaceutical")

# Initialize session state to store chat history and input
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# List of options for the dropdown
options = ["Llama2 Base Model", "Llama2 finetuned Model"]

# Create a single-select dropdown
selected_model = st.selectbox("Select a Model:", options)



# Function to display available commands
def show_commands():
    with st.expander("Available Commands"):
        st.write(
            """
        - `/summarize`: Summarize the chat history
        - `/info`: Display information about the assistant
        """
        )


model_name = "CaptainAI"
def clean_resp(resp):
    try:
        resp = resp.split("Assistant: ")[1]
        return resp
    except:
        return resp
    


# chat_history = "hi"
def get_base_response(chat_history):
    import requests, json
    data = {
        "model": model_name,
        "prompt": chat_history,
        "stream": False
    }
    headers = {
        "Content-Type": "application/json"
        
    }
    try :
        url = "http://localhost:11434/api/generate"
        start_time = time.time()
        response = requests.post(url, headers=headers, data = json.dumps(data))
        end_time = time.time()
        execution_time = end_time - start_time

        if response.status_code == 200:
            response = response.text
            data = json.loads(response)
            filtered = clean_resp(data['response'])
            return filtered, execution_time
        else :
            return "Issue with the model. Please fix me :()", 0
    except:
        return "Somethings went wrong. Could not connect. Please fix me :(", 0


def clean_summary(resp):
    try:
        resp = resp.split(":")[1]
        rm_char = ['{','}',']','[']
        for i in range(len()):
            resp = resp.replace(rm_char[i],'')
        return resp
    except:
        return resp
    
def get_summary(chat_history):
    import requests, json
    system_prompt = 'Summarize the follwing conversation with no heading and made it from doctor point of view'
    conversation_string = chat_history
    prompt = f"{system_prompt}: {conversation_string}"
    data = {
        "model": "llama3",
        "prompt": prompt,#"Summarize in a medical note format" + chat_history,
        # "prompt": "[Doctor: Hi, how are you feeling today? Patient: I'm okay, thank you. Doctor: Can you tell me what brings you here today? Patient: I have a problem with my breasts. They have been growing and getting bigger for a long time. Doctor: I see. How long has this been going on? Patient: Since I was 9 months old. Doctor: And have you sought medical advice before? Patient: Yes, I have been to a few primary health care facilities, but they always told me not to worry. Doctor: I understand. Is there anything else you would like to add about your symptoms? Patient: No, that's all. Doctor: Okay. Let me examine you. Can you tell me your weight and height? Patient: My weight is 17 kg and my height is 108 cm. Doctor: That's good. Your blood pressure is normal. Let me check your growth chart. You are at the 50th centile for weight and 90th centile for height. Patient: Hmm, okay. Doctor: And have you had any vaginal bleeding or pubic or axillary hair? Patient: No, I haven't. Doctor: I see. And how is your vagina? Patient: It's reddish and mucoid. Doctor: Okay. Can you tell me about your clitoromegaly, acne, hirsutism, or any abdominal mass? Patient: No, I don't have any of those. Doctor: That's good. We did a wrist X-ray and found that your bone age is 8 years. Patient: Hmm, okay. Doctor: We also did a hormonal evaluation using fluorometric enzyme immunoassay. Your luteinizing hormone was 3.1 mIU/L, but it increased to 8.8 mIU/L 45 minutes after gonadotrophin-releasing hormone stimulation.]",#prompt,

        "stream": False
    }
    headers = {
        "Content-Type": "application/json"
        
    }
    try :
        url = "http://localhost:11434/api/generate"
        start_time = time.time()
        response = requests.post(url, headers=headers, data = json.dumps(data))
        end_time = time.time()
        execution_time = end_time - start_time

        if response.status_code == 200:
            response = response.text
            data = json.loads(response)
            filtered = clean_summary(data['response'])
            return filtered, execution_time
        else :
            return "Issue with the model. Please fix me :()", 0
    except:
        return "Somethings went wrong. Could not connect. Please fix me :(", 0
    
def get_finetuned_reply(json_data):
    try:
        headers = {
        "Content-Type": "application/json"
        
        }   
        url = "http://127.0.0.1:5000/predict"
        response = requests.post(
                    url, headers=headers, data=json.dumps(json_data),
                )
        if response.status_code == 200:
            return response.text
        else:
            st.error(f"Error from flask api response. Please fix me :( \n Response: {response.text}")
            return response.text
    except:
        return "Somethings went wrong. Could not connect. Please fix me :("
        
def check_duplicate_reply(chat_hist, response):
    import ast
    try:
        conversation_list = ast.literal_eval(chat_hist.strip())

        reply_exists = any(reply['content'] == response for reply in conversation_list if reply['role'] == 'assistant')

        if reply_exists:
            return True
        else:
            return False
    except:
        return False

# Display available commands from the start
show_commands()

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
prompt = st.chat_input("Say something")
chat_hist = ''
if prompt:
    if prompt.lower().startswith("/summarize"):
        # Placeholder summary
        summary,ets = get_summary('Chat History: \n' + str(st.session_state.chat_history))
        # Display the summary

        st.markdown(
            f"""
        <div style="font-size: small;">
        <hr />
        <b>Summary of the conversation</b> </br> {summary}</br></br>
        <b> Response Time: {ets}</b>
        </div>
            """,
            unsafe_allow_html=True,
        )
    elif prompt.lower().startswith("/info"):
        # Display information about the assistant
        # info = "This AI assistant helps with pharmaceutical inquiries and general health information."
        # st.write(info)

        st.markdown(
            """
        <div style="font-size: small;">
        <hr />
        <b>Information</b> </br> This AI chatbot is trained on conversation between doctor and patient.
        </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        chat_hist = 'Chat History: \n' + str(st.session_state.chat_history) + '\n' + "User: " + prompt
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        # Display the new messages
        with st.chat_message("user"):
            st.write(prompt)
        if selected_model == "Llama2 Base Model":
            response,et = get_base_response(chat_hist) # Replace with your AI model's response
            
        else:
            import json
            response, et = get_reply_finetuned(chat_hist)#get_finetuned_reply({'text':chat_hist})#get_reply_finetuned(chat_hist)#'hey'#get_reply_finetuned(chat_hist)
            if check_duplicate_reply(response, str(st.session_state.chat_history)):
                response = get_reply_finetuned(chat_hist)#get_finetuned_reply({'text':chat_hist})#
        st.session_state.chat_history.append({"role": "assistant", "content": response})

        
        with st.chat_message("assistant"):
            output = response + '\n\n' + f"ResponseTime: {et}ms"
            st.write(output)
            


else:
    # Disclaimer
    st.markdown(
        """
        <div style="font-size: small;">
            <hr />
            <b>Disclaimer:</b> This AI health assistant is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding medical condition.
        </div>
        """,
        unsafe_allow_html=True,
    )

# Hide the Streamlit menu and footer
st.markdown(
    """
    <style>
    .css-1aumxhk {
        display: none;
    }
    .css-15zrgzn {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
