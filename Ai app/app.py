from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI
from dotenv import load_dotenv,find_dotenv
import requests
from PIL import Image
import io
import streamlit as st


load_dotenv(find_dotenv())
import os
openai_key=os.getenv("openai_api")
huggingface_key=os.getenv("huggingface_key")
os.environ["OPENAI_API_KEY"] = openai_key


def generatestory(context=None,story=None):
    template1="""
    you are a story teller;genearate a creative story about the prompt,max  40 words;
    CONTEXT:{context}
    STORY:"""
    if story:
        template2="""
    you are given a story ;genearate a prompt for image generation,image describes the story but dont include story details in prompt;
    CONTEXT:{story}
    prompt:"""
        prompt=PromptTemplate(template=template2,input_variables=["story"])
        prompt_llm=LLMChain(llm=OpenAI(model_name="gpt-3.5-turbo",temperature=1),prompt=prompt,verbose=True)
        prompt=prompt_llm.predict(story=story)
        return prompt
    prompt=PromptTemplate(template=template1,input_variables=["context"])
    story_llm=LLMChain(llm=OpenAI(model_name="gpt-3.5-turbo",temperature=1),prompt=prompt,verbose=True)
    story=story_llm.predict(context=context)
    print(story)
    return story
def text_to_img(text):
    API_URL = "https://api-inference.huggingface.co/models/kr-manish/text-to-image-sdxl-lora-dreemBooth-rashmika"
    headers = {"Authorization": f"Bearer {huggingface_key}"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.content


    
    image_bytes= query({"inputs": text,})
    image = Image.open(io.BytesIO(image_bytes))
    image.save("output_image.jpg")
def text_to_audio(text):
    
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {huggingface_key}",}
    payload={"inputs":text}
    response = requests.post(API_URL, headers=headers, json=payload)
    with open('audio.flac','wb') as file:
        file.write(response.content)

def main():
    st.set_page_config(page_title="story generator",page_icon=":-)")
    st.header("give a sentence it will generate a story with an image")
    user_input=st.text_input("give a sceneria:")
    if user_input is not None:
        story=generatestory(context=user_input,story=None)
        text_to_audio(story)
        prompt=generatestory(context=None,story=story)
        image=text_to_img(prompt)
        with st.expander("story"):
            st.write(story)
        st.image("output_image.jpg",caption=prompt)
        st.audio("audio.flac")
if __name__=='__main__':
    main()








   
