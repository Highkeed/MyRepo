import os
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableMap
import google.generativeai as genai
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv('.env')
# Configure Google AI API 
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Setup embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Extract text from uploaded files
def extract_text_from_resume(file):
    loader = PyPDFLoader(file)
    documents = loader.load()
    text = " ".join([doc.page_content for doc in documents])
    return text

# Text splitting
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.create_documents([text])

# Extract percentage score from analysis text
def extract_suitability_score(text):
    match = re.search(r"Suitability Score: (\d{1,3})%", text)
    if match:
        return int(match.group(1))
    return None


job_requirements = input("Enter job requirements: ")
file = 'C:\\Edureka\\course-Agentic AI\\03-Working with Langchain and LCEL\\demo1\\Resume_Sample 1.pdf'
resume_text = extract_text_from_resume(file)
print("###################### Resume Text #######################\n", resume_text)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=GOOGLE_API_KEY)

prompt_template = PromptTemplate(
            input_variables=["job_requirements", "resume_text"],
            template="""
            You are an expert HR and recruitment specialist. Analyze the resume below against the job requirements.

            Job Requirements:
            {job_requirements}

            Resume:
            {resume_text}

            Provide a structured analysis of how well the resume matches the job requirements. 
            At the end, clearly state a "Suitability Score" as a percentage (0-100%) based on how well the resume aligns with the job.
            Format: Suitability Score: XX%
            """
        )

chain = (
            RunnableMap({
                "job_requirements": lambda x: x["job_requirements"],
                "resume_text": lambda x: x["resume_text"]
            })
            | prompt_template
            | llm
            | StrOutputParser()
        )

analysis = chain.invoke({"job_requirements": job_requirements,"resume_text": resume_text})
print("######################## Analysis #########################\n", analysis)

# Extract and display the suitability score
suitability_score = extract_suitability_score(analysis)
print("######################## Suitablity Score #########################\n", suitability_score)
