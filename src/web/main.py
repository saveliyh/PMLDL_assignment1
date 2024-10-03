import streamlit as st
import requests

st.title('PMLDL assignment 1')

text = st.text_input("Right joke in russian", value="Заходят как-то в бар")

response = requests.get(f'http://localhost:8000/?text={text}')



st.text(f"joke is fun for {response.json()['rating']}")
