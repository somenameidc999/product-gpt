from dotenv import load_dotenv
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

def main():
    load_dotenv()
    st.title('🦜🔗 Product AutoGPT')
    prompt = st.text_input('Product Prompt')

    # Product Title Prompt
    title_template = PromptTemplate(
        input_variables = ['topic'],
        template = 'Write me a fun, catchy, concise product title about {topic}'
    )

    # Product Description Prompt
    description_template = PromptTemplate(
        input_variables = ['title', 'wikipedia_research'],
        template = """Write me a product description about {title}. 
        Make sure the description is about the product only, not the store.
        Also leverage this Wikipedia Research: {wikipedia_research}"""
    )

    # Sets Memory
    title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
    description_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

    # Sets LLM with Chains
    temperature = 0.9
    llm = OpenAI(temperature=temperature, verbose=True)
    title_chain = LLMChain(
        llm=llm, 
        prompt=title_template, 
        verbose=True, 
        output_key='title',
        memory=title_memory
    )
    description_chain = LLMChain(
        llm=llm, 
        prompt=description_template, 
        verbose=True, 
        output_key='description',
        memory=description_memory
    )
    # sequential_chain = SequentialChain(
    #     chains=[title_chain, description_chain],
    #     input_variables = ['topic'],
    #     output_variables = ['title', 'description'],
    #     verbose=True
    # )
    wiki = WikipediaAPIWrapper()

    if prompt:
        title = title_chain.run(prompt)
        wikipedia_research = wiki.run(prompt)
        description = description_chain.run(title=title, wikipedia_research=wikipedia_research)

        st.subheader('Your Product Title')
        st.write(title)
        st.subheader('Your Product Description')
        st.write(description)

        with st.expander('Title History'):
            st.warning(title_memory.buffer)

        with st.expander('Description History'):
            st.warning(description_memory.buffer)

        with st.expander('Wikipedia Research History'):
            st.warning(wikipedia_research)

        # To be used with SequentialChain
        # response = sequential_chain(prompt)
        # st.write(response['title'])
        # st.write(response['description'])

        # with st.expander('Message History'):
        #     st.warning(memory.buffer)


# if application is imported, it will not run
if __name__ == "__main__":
    main()