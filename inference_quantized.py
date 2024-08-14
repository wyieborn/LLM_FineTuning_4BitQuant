from llama_cpp import Llama
import torch


def process_reply(s):
    
    try:
        tag_index = s.find(":")
        space_index = s.rfind(' ', 0, tag_index)
        tag = s[space_index + 1:tag_index] + ":"
        
    except:
        tag = '' 


    try:
        s = s.split(tag)[0]
        s = s.replace("[]","")
    except:
        pass

    return s


        
        
def get_reply_finetuned(query):
    try:
        import time
        if torch.cuda.is_available():
            print('running in GPU')
            llm = Llama(
                            model_path="quantized_models/peft_llama_chatbot-q4_k_m.gguf",
                            n_gpu_layers=-1 ,
                            chat_format="llama-2",
                            verbose = False
                        )
        else:
            print('running in CPU')
            llm = Llama(
                            model_path="quantized_models/peft_llama_chatbot-q4_k_m.gguf",
                            chat_format="llama-2",
                            verbose = False
                        )
    except:
        llm = Llama(
                            model_path="quantized_models/peft_llama_chatbot-q4_k_m.gguf",
                            chat_format="llama-2",
                            verbose = False
                        )
        
    # llm = Llama(
    #                         model_path="quantized_models/peft_llama_chatbot-q4_k_m.gguf",
    #                         chat_format="llama-2",
    #                         verbose = False
    #                     )
    start_time = time.time()
    resp = llm(f"Q: {query},  A: ", max_tokens = 100, stop=["Q:","\n"])
    end_time = time.time()
    execution_time = end_time - start_time

    resp = process_reply(resp['choices'][0]['text'])
    return resp , execution_time
