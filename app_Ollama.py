import streamlit as st
from htmlTemplates import css, bot_template, user_template
from restCallOllama import getrestcallsdone

def handle_userinput(user_question):
    response = getrestcallsdone(user_question)

    st.session_state.chat_history.append({"user": user_question})
    # Process user input and generate bot response here
    # ...

    st.session_state.chat_history.append({"bot": response})

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message['user']), unsafe_allow_html=True)
        else:
            if(type(message['bot']) == 'str'):
                st.write(bot_template.replace(
                    "{{MSG}}", message['bot']), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace(
                    "{{MSG}}", "We will intimate to our tech support about your query!!! INC will be shared to you in mail"), unsafe_allow_html=True)


def main():
    #load_dotenv()
    st.set_page_config(page_title="Wealth Tech Insight",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Explore Wealth Tech in Citi")
    user_question = st.text_input("Have a doubt, Just Ask Me?:")
    if user_question:
        handle_userinput(user_question)


if __name__ == '__main__':
    main()
