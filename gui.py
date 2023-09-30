import streamlit as st
from sample import Sampler
from omegaconf import OmegaConf
import time
import random
def sample():
    z = time.time()
    print(st.session_state.guidance)
    img, img_list = st.session_state.sampler.sample(st.session_state.prompt,guidance=st.session_state.guidance,
                                         height=st.session_state.height,width=st.session_state.width,
                                         steps=st.session_state.T,x_steps=st.session_state.x_steps,
                                         return_steps=st.session_state.int_res,seed=st.session_state.R)
    st.session_state.img = img
    st.session_state.img_list = img_list
    st.session_state.inf_time = time.time()-z

def rand():

    st.session_state.R = random.randint(0,100000)

if "config" not in st.session_state.keys():
    st.session_state.config = OmegaConf.load("config.yaml")
    st.session_state.sampler = Sampler(st.session_state.config.device)
st.title("Latent Diffusion Demonstration")
st.metric("Device", st.session_state.config.device)
st.text_input("Prompt", value="a kitten with blue cowboy hat", key="prompt")
st.number_input("width", min_value=256,max_value=2048,key="width", value=512)
st.number_input("height", min_value=256,max_value=2048,key="height",value=512)
st.slider("Guidance",min_value=0.0,max_value=30.0,step=0.1,value=7.5, key="guidance")
st.number_input("Inference Steps", min_value=5, max_value=1000,step=1,value=128,key="T")
st.number_input("Seed", value=0,key="R")
st.button("Randomize", on_click=rand)
st.checkbox("Show Intermediate Results", key="int_res")
st.slider("Every n steps", min_value=1,max_value=100,step=1,value=10,key="x_steps")
st.button("Generate!",on_click=sample)
if "img" in st.session_state:
    st.image(st.session_state.img)
    st.metric("Inference Time", round(st.session_state.inf_time,2))
    for i, img in enumerate(st.session_state.img_list):
        st.subheader(f"After {(i+1)*st.session_state.x_steps} steps")
        st.image(img)