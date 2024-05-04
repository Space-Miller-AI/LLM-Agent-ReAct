#Back to Lesson 2, time flies!
import gradio as gr
import os
from langchain_community.llms import openai
# from dotenv import load_dotenv
# load_dotenv()

# llm = OpenAI(temperature=0.1, model_name='gpt-3.5-turbo', max_tokens=200)
# r = llm.invoke('Who is Lionel Messi?')
# print(r)
# def generate(input, slider):
#     llm = OpenAI(temperature=0.1, model_name='gpt-3.5-turbo', max_tokens=slider)
#     return llm(input)

# demo = gr.Interface(fn=generate, 
#                     inputs=[gr.Textbox(label="Prompt"), 
#                             gr.Slider(label="Max new tokens", 
#                                       value=20,  
#                                       maximum=1024, 
#                                       minimum=1)], 
#                     outputs=[gr.Textbox(label="Completion")])

# gr.close_all()
# demo.launch(share=True, server_port=int(os.environ['PORT1']))