from flask import Flask, render_template, request, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
import pickle
import re
import nltk
from pypdf import PdfReader
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'txt'}

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load trained model and vectorizer
clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clean_resume(resume_text):
    clean_text = re.sub(r"http\S+", " ", resume_text)
    clean_text = re.sub(r"RT|cc", " ", clean_text)
    clean_text = re.sub(r"@\S+", " ", clean_text)
    clean_text = re.sub(r"#[A-Za-z0-9]+", " ", clean_text)
    clean_text = re.sub(r"[^a-zA-Z\s]", " ", clean_text)
    clean_text = re.sub(r"\s+", " ", clean_text).strip()
    
    tokens = word_tokenize(clean_text)
    stop_words = set(stopwords.words('english'))
    clean_text = " ".join([word.lower() for word in tokens if word.lower() not in stop_words])
    
    return clean_text

def calculate_ats_score(resume_text, category_name):
    category_keywords = {
        "Python Developer": ["python", "flask", "django", "pandas", "numpy", "tensorflow", "scikit-learn"],
        "Java Developer": ["java", "spring", "hibernate", "j2ee", "jsp", "servlet", "jdbc"],
        "Data Science": ["machine learning", "deep learning", "data analysis", "pandas", "numpy", "tensorflow"],
        "Web Developer": ["html", "css", "javascript", "react", "angular", "node.js", "express"],
        "DevOps Engineer": ["aws", "docker", "kubernetes", "ci/cd", "jenkins", "terraform", "ansible"]
    }
    
    keywords = category_keywords.get(category_name, [])
    matched_keywords = [kw for kw in keywords if kw.lower() in resume_text.lower()]
    ats_score = (len(matched_keywords) / len(keywords)) * 100 if keywords else 0
    return ats_score

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            resume_text = ""
            if filename.endswith('.pdf'):
                pdf_reader = PdfReader(filepath)
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text:
                        resume_text += text
            else:
                with open(filepath, 'r') as f:
                    resume_text = f.read()

            if not resume_text.strip():
                flash('Failed to extract text from the resume')
                return redirect(request.url)
            
            cleaned_resume = clean_resume(resume_text)
            input_features = tfidf.transform([cleaned_resume])
            prediction_id = clf.predict(input_features)[0]
            
            category_mapping = {
                15: "Java Developer",
                23: "Testing",
                8: "DevOps Engineer",
                20: "Python Developer",
                24: "Web Designing",
                6: "Data Science",
            }
            
            category_name = category_mapping.get(prediction_id, "Unknown")
            ats_score = calculate_ats_score(resume_text, category_name)
            
            return render_template('result.html', 
                                category=category_name,
                                score=f"{ats_score:.2f}")
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)