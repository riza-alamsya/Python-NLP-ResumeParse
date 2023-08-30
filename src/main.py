import spacy
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

nlp = spacy.load("en_core_web_sm")
# Data pelatihan (contoh sederhana)
training_data = [
    {"text": "Saya lulus dari Universitas A. Saya memiliki pengalaman sebagai developer.", "label": "experience"},
    {"text": "Saya memiliki pendidikan dari Universitas B.", "label": "education"},
    # ... contoh data lainnya ...
]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform([data["text"] for data in training_data])
y = [data["label"] for data in training_data]
model = MultinomialNB()
model.fit(X, y)
def parse_resume(text):
    doc = nlp(text)
    parsed_info = {"name": "", "education": [], "experience": []}
    
    # Mengekstrak nama
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            parsed_info["name"] = ent.text
            break  
    # Mengekstrak pengalaman
    experience_keywords = ["pengalaman", "experience"]
    for sent in doc.sents:
        lower_sent = sent.text.lower()
        for keyword in experience_keywords:
            if keyword in lower_sent:
                experience_info = lower_sent.split(keyword)[-1].strip()
                parsed_info["experience"].append(experience_info)
                
    # Mengekstrak pendidikan
    education_keywords = ["education", "pendidikan"]
    for sent in doc.sents:
        lower_sent = sent.text.lower()
        for keyword in education_keywords:
            if keyword in lower_sent:
                education_info = lower_sent.split(keyword)[-1].strip()
                parsed_info["education"].append(education_info)
                
    return parsed_info

resume_text = "Riza Alamsya berpengalaman sebagai java developer selama 4 tahun. Saya memiliki pendidikan dari Universitas A."
parsed_info = parse_resume(resume_text)
output_json = json.dumps(parsed_info, indent=4)
print(output_json)



