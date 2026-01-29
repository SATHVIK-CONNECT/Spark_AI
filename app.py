import streamlit as st
import os
import tempfile
import json
import base64
from groq import Groq

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter # Updated

from langchain_core.prompts import ChatPromptTemplate
from streamlit_lottie import st_lottie

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser



import langchain

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

st.set_page_config(page_title="Spark AI", page_icon="⚡")

canvas = st.markdown("""
    <style>
        header{ visibility: hidden; }   
    </style> """, unsafe_allow_html=True)


def generate(uploaded_image, prompt):
    base64_image = base64.b64encode(uploaded_image.read()).decode('utf-8')
    client = Groq(api_key=st.secrets["api_key"])
    chat_completion = client.chat.completions.create(
        messages=[
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': prompt},
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': f'data:image/jpeg;base64,{base64_image}',
                        },
                    },
                ],
            }
        ],
        model='meta-llama/llama-4-scout-17b-16e-instruct',
    )
    return chat_completion.choices[0].message.content

st.title("⚡Spark AI")

tab_titles = [
    "Home",
    "Vision Instruct",
    "File Query",
    "About",
]

vision = [
    "The vision instruct system utilizes visual instructional RAG to generate interactive and immersive visual instructions. By incorporating computer vision-based instructional RAG, the system can analyze images and generate corresponding instructions. With the integration of multimodal RAG for visual instruction, users can interact with the system using multiple modes of input, enhancing their understanding and retention of complex concepts.\n\nHow to use Vision Instruct\n\n1. Go to Vision Instruct tab.\n\n2. Click on upload option and upload the image.\n\n3. Give the prompt (Question).\n\n4. Click on Generate option."
    ]     
file = [
    "The file query system utilizes document-centric querying with RAG to retrieve relevant information from PDF documents. By incorporating PDF-based knowledge retrieval with RAG, the system can analyze the content of PDF files and generate answers to user queries. With the integration of file-based question answering with RAG, users can ask questions about the content of files and receive accurate and relevant responses.\n\nHow to use File Query\n\n1. Go to File Query tab.\n\n2. Click on upload option and upload the PDF File.\n\n3. Give the prompt (Question).\n\n4. Click on Generate option."
   ]     

tabs = st.tabs(tab_titles)
with tabs[0]:
    def lottie(anime="anime.json"):
        with open(anime, "r", encoding='UTF-8') as animation:
            return json.load(animation)
    animes = lottie()
  


    col1, col2 = st.columns(2, gap="large", vertical_alignment="center")
    with col2:
          st_lottie(animes, width=300, height=300)
    with col1:      
        st.markdown("""<h4>Welcome to Spark AI!</h4>
                    <p style="text-align: justify;">Unlock the power of AI-driven image and file analysis with our innovative application. Sparkis designed to simplify complex tasks, providing accurate and efficient results.</p>""", unsafe_allow_html=True)
    st.markdown("""<hr>""", unsafe_allow_html=True)
    st.image(image="slide1.webp")
    st.markdown("""<h4>Retrieval Augumented Generation</h4>
                        <p style="text-align: justify;">The Retrieval-Augmented Generation (RAG) framework leverages hybrid retrieval-generation techniques to produce more accurate and informative responses. By combining the strengths of retrieval and generation models, RAG enables knowledge-augmented language generation, where relevant facts and information are seamlessly integrated into the generated text. This approach facilitates generative retrieval, 
                        allowing the model to retrieve and generate text in a single, unified framework. Ultimately, RAG has the potential to revolutionize natural language processing and language generation, enabling the development of more sophisticated and knowledgeable AI systems.</p><hr>""", unsafe_allow_html=True)        
    st.markdown("""<h4>Advantages of the Spark AI</h4>
                        <p style="text-align: justify;">It simplifies daily life tasks by using AI, generates the anlyzed data with in a minute. It saves the time by reading all data in files using AI-driven model.</p>""", unsafe_allow_html=True)
    st.image(image="advantage.png")
    st.markdown("""<hr>
                        <h4>Explore Our Features - Get Started</h4>
                        <h5>Vision Instruct</h5>
                        <p style="text-align: justify;">It is used to query with images. It let us analyze the image data by using the llama model.</p>""", unsafe_allow_html=True)
    with st.expander("V I S I O N - I N S T R U C T"):
        st.write(vision[0])

    st.markdown("""
       <h5>File Query</h5>
       <p style="text-align: justify;">It is used to query with files. It let us analyze the files like PDF, TXT and so on by using the llama model.</p>
    """, unsafe_allow_html=True)
    with st.expander("F I L E - Q U E R Y"):
       st.write(file[0])

with tabs[1]:
    # def img_analyze(img_analyze="img_analyze.json"):
    #     with open(img_analyze, "r", encoding='UTF-8') as f:
    #         return json.load(f)
    # img_analyze = img_analyze()
    # st_lottie(img_analyze)

    uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
            st.image(uploaded_file, caption='Uploaded Image')
            prompt = st.text_input('Enter the prompt')

            if st.button('Generate'):
                with st.spinner('Generating output...'):
                    if prompt:
                        output = generate(uploaded_file, prompt)
                    else:
                        output = generate(uploaded_file, 'What is in this picture?')
                st.subheader('Result:')
                st.write(output)
with tabs[2]:        
    if " ":
        st.markdown("""<h4 style='color:#dd1100; text-align:center'>This feature is currently being updated.</h4>""", unsafe_allow_html=True)
        st.markdown("""<h4 style='color:#dd1100; text-align:center'>Please check back later!</h4>""", unsafe_allow_html=True)
        def pdf_analyze(file_path="error.json"):
            with open(file_path, "r", encoding='UTF-8') as f:
                return json.load(f)
        st_lottie(pdf_analyze())   
        
    else:
        llm = ChatGroq(
            groq_api_key=st.secrets["api_key"], 
            model_name="llama-3.1-8b-instant", 
            temperature=0
        )

        rag_prompt = ChatPromptTemplate.from_template("""
        Answer the question based ONLY on the provided context.
        If the answer isn't in the context, say "I don't find that in the document."
        
        <context>
        {context}
        </context>
        
        Question: {input}
        """)

        def process_pdf(pdf_file):
            if "vector_store" not in st.session_state:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(pdf_file.read())
                    tmp_path = tmp.name

                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(docs)

                embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')
                
                st.session_state.vector_store = FAISS.from_documents(splits, embeddings)
                os.remove(tmp_path) 

        pdf_upload = st.file_uploader("Upload PDF", type=['pdf'])

        if pdf_upload:
            process_pdf(pdf_upload)
            st.success("PDF processed and indexed!")

        if "vector_store" in st.session_state:
            st.divider()
            user_query = st.text_input("Ask a question about the PDF:")
            
            if st.button('Generate Analysis', type="primary"):
                if user_query:
                    retriever = st.session_state.vector_store.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 5} 
                    )

                    def format_docs(docs):
                        if not docs:
                            return "EMPTY_CONTEXT"
                        return "\n\n".join(doc.page_content for doc in docs)

                    chain = (
                        {"context": retriever | format_docs, "input": RunnablePassthrough()}
                        | rag_prompt
                        | llm
                        | StrOutputParser()
                    )

                    with st.spinner('Analyzing...'):
                        relevant_docs = retriever.invoke(user_query)
                        
                        if not relevant_docs:
                            st.error("The search returned no matching text from the PDF.")
                        else:
                            response = chain.invoke(user_query)
                            st.markdown("### Answer")
                            st.write(response)

                            with st.expander("Diagnostic: See what the AI 'read'"):
                                st.write(f"**Number of chunks found:** {len(relevant_docs)}")
                                for i, doc in enumerate(relevant_docs):
                                    st.info(f"Chunk {i+1}:\n\n{doc.page_content[:500]}...")
                else:
                    st.warning("Please enter a question.")

with tabs[3]:
    st.markdown("""
        <h4>About Spark AI</h4>
        <p style="text-indent: 60px; text-align: justify;"> Spark is an AI-powered application developed as part of the Applied Artificial Intelligence: Practical Implementations course  by TechSaksham Program, which is a CSR initiative by Microsoft and SAP, implemented by Edunet Foundation</p>
        <hr>""", unsafe_allow_html=True)
    col5, col6 = st.columns(2, gap="large", vertical_alignment="center")
    with col5:
        st.markdown("""        <ul> 
            <h3>Project Development Details</h3>
            <h4>Developer</h4>
            <li>Sathvik Palivela</li>
        </ul>
        <ul>
            <h4>Mentor</h4>
            <li>Abdul Aziz Md</li>
        </ul>
        <br>""", unsafe_allow_html=True)
    with col6:
        def coding(coding = "coding.json"):
            with open(coding, 'r', encoding='UTF-8') as f:
                return json.load(f)
        icon = coding()
        st_lottie(icon, width=350, height=350)
    st.markdown("""<hr>
        <h4>Acknowledgements</h4>
        <p>We would like to extend our gratitude to: </p>
        <ul><li>TechSaksham Program, a CSR initiative by Microsoft and SAP.</li>
            <li>Edunet Foundation for implementing the AI Practical Implementations course.</li>
            <li>Aziz Sir for excellent guidance and mentorship.</li></ul>
        <br>
        <h4>GitHub Repository</h4>
        <p>Check our github repository - <a href='https://github.com/SATHVIK-CONNECT/Project/tree/main'>Git Repo of Spark AI</a></p>
        <br> 
        <h4>Contact Us</h4>
        <p>For any queries or feedback, please reach out to us at <a href='mailto:sathvikpalivela0@gmail.com'>sathvikpalivela0@gmail.com</a>.
    """, unsafe_allow_html=True)
    
