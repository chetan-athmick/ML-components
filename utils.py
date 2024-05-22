import os
import re
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
from langchain_openai import AzureOpenAIEmbeddings
from openai import AzureOpenAI

os.environ['AZURE_OPENAI_API_KEY'] = "e3151139857144bb9d56d029fcd44e8f"
os.environ['AZURE_OPENAI_ENDPOINT'] = "https://athmick-openai.openai.azure.com/"
os.environ['OPENAI_API_VERSION'] = "2024-02-15-preview"

client = AzureOpenAI()

def get_pdf_text(file_path):
    text = ''
    pdf = PdfReader(file_path)
    for page_number in range(len(pdf.pages)):
        page = pdf.pages[page_number]
        text += page.extract_text()
    return text

def create_docs(user_pdf_list, unique_id):
    docs = []
    for file in user_pdf_list:
        chunks = get_pdf_text(file)
        docs.append(Document(
            page_content=chunks,
            metadata={"name": file.filename, "unique_id": unique_id},
        ))
    return docs

def create_embeddings_load_data():
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment="AZUREEMBEDDING"
    )
    return embeddings

def opeani_response(resume, job_description):
    chat_completion = client.chat.completions.create(
        model="GPT35",
        messages=[
            {"role": "system", "content": "You are a Detailed Resume Matcher For Given Job description."},
            {"role": "user", "content": f"""
                Given the job description and the resume, assess the given job description and the resume with detailed analysis. provide matching percentage.
                **Job Description:**{job_description}
                **Resume:**{resume}

                **Detailed Analysis:**
                **the result should be in this format:**
                '''Matched Percentage: [matching percentage].
                Reason: [Reasons for why this resume matched and not matched.].
                Skills To Improve: [Mention the skills to improve for the candidate according to the given job description.].
                Irrelevant: [mention the irrelevant skills and experience].
                Keywords: [Return the matched keywords from resume and job description.]'''
            """}
        ],
        max_tokens=500,
        temperature=0
    )
    generated_text = chat_completion.choices[0].message.content
    return generated_text

def get_strip_response(matched_result):
    lines = matched_result.split('\n')
    matched_percentage = None
    reason = []
    skills_to_improve = []
    keywords = []
    irrelevant = []

    section = None

    for line in lines:
        line = line.strip()
        if line.startswith('Matched Percentage:'):
            match = re.search(r"Matched Percentage: (\d+)%", line)
            if match:
                matched_percentage = int(match.group(1))
        elif line.startswith('Reason:'):
            section = 'reason'
            reason.append(line.split(':', 1)[1].strip())
        elif line.startswith('Skills To Improve:'):
            section = 'skills_to_improve'
            skills_to_improve.append(line.split(':', 1)[1].strip())
        elif line.startswith('Keywords:'):
            section = 'keywords'
            keywords.append(line.split(':', 1)[1].strip())
        elif line.startswith('Irrelevant:'):
            section = 'irrelevant'
            irrelevant.append(line.split(':', 1)[1].strip())
        else:
            if section == 'reason':
                reason.append(line)
            elif section == 'skills_to_improve':
                skills_to_improve.append(line)
            elif section == 'keywords':
                keywords.append(line)
            elif section == 'irrelevant':
                irrelevant.append(line)

    reason = ' '.join(reason).strip('- ')
    skills_to_improve = ' '.join(skills_to_improve).strip('- ')
    keywords = ' '.join(keywords).strip('- ')
    irrelevant = ' '.join(irrelevant).strip('- ')

    return matched_percentage, reason, skills_to_improve, keywords, irrelevant

def get_summary(resume):
    chat_completion = client.chat.completions.create(
        model="GPT35",
        messages=[
            {"role": "system", "content": "You are a Resume summarizer."},
            {"role": "user", "content": f"""Summarize the given resume within 60 words. resume : {resume}"""}
        ],
        max_tokens=200,
        temperature=0
    )
    summary = chat_completion.choices[0].message.content
    return summary
