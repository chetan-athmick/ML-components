from flask import Flask, request, jsonify
import uuid
import pandas as pd
from utils import *
from langchain_community.vectorstores import FAISS

app = Flask(__name__)

@app.route('/')
def home():
    return "HR Resume Screening Assistance API"

@app.route('/analyze', methods=['POST', 'GET'])
def analyze():
    job_description = request.files.get('job_description')
    resumes = request.files.getlist('resumes')
    document_count = int(request.form.get('document_count', 5))

    if not job_description or not resumes:
        return jsonify({"error": "Job description and resumes are required."}), 400

    unique_id = uuid.uuid4().hex

    job_description_txt = get_pdf_text(job_description)
    final_docs_list = create_docs(resumes, unique_id)
    
    embeddings = create_embeddings_load_data()
    db = FAISS.from_documents(final_docs_list, embeddings)
    
    relevant_docs = db.similarity_search_with_relevance_scores(job_description_txt, k=document_count)
    
    data = []
    
    for resume in resumes:
        resume_txt = get_pdf_text(resume)
        matched_result = opeani_response(resume_txt, job_description_txt)
        matched_percentage, reason, skills_to_improve, keywords, irrelevant = get_strip_response(matched_result)
        summary = get_summary(resume_txt)
        data.append({
            "File": resume.filename,
            "Matched Score": matched_percentage,
            "Matched Reason": reason,
            "Skills to improve": skills_to_improve,
            "Keywords": keywords,
            "Irrelevant": irrelevant,
            "Summary": summary
        })
    
    df = pd.DataFrame(data)
    df_sorted = df.sort_values(by='Matched Score', ascending=False).reset_index(drop=True)
    
    return df_sorted.to_json(orient='records')

if __name__ == '__main__':
    app.run(debug=True)
