from flask import Flask, render_template, request
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.nn.functional import softmax
import torch
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

app = Flask(__name__)

# Load the fine-tuned model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('./results')
model = DistilBertForSequenceClassification.from_pretrained('./results')

# Define English stop words list
stop_words = set(stopwords.words('english'))

# Initialize WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Function to load FAQ data from dataset.txt file and remove stop words from keywords
def load_faq_data(file_path):
    faq_data = {}
    current_category = None
    
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('[') and line.endswith(']'):
                current_category = line[1:-1]  # Remove brackets to get category name
                faq_data[current_category] = []
            elif '|' in line:
                question, answer = line.split('|', 1)
                # Remove stop words from question and split into keywords
                keywords = [lemmatizer.lemmatize(word.lower()) for word in question.split() if word.lower() not in stop_words]
                faq_data[current_category].append({"question": question.strip(), "answer": answer.strip(), "keywords": keywords})
    
    return faq_data

# Load FAQ data from dataset.txt file
file_path = '/Users/aadithyaram/Desktop/Projekts/SRMChatbot/ChatBot-main-2/static/dataset/dataset.txt'
faq_data = load_faq_data(file_path)

# Define fallback responses for each category
fallback_responses = {
    0: "I'm not sure about that. For more information on admissions, please visit our admissions page.",
    1: "I don't have that information right now. You can check the courses page for detailed information.",
    2: "That seems to be out of my scope. For specific details, please contact the support team.",
    3: "I may not have the answer to that. Try browsing our campus facilities page.",
    # Add more categories and their fallback responses as needed
    'default': "I'm not sure about that. Please try rephrasing your question or check our website for more information."
}

# Function to calculate keyword match score
def calculate_match_score(user_query, faq_keywords):
    user_query_words = [lemmatizer.lemmatize(word.lower()) for word in user_query.split()]
    match_score = sum(1 for word in faq_keywords if word in user_query_words)
    return match_score

# Function to find the best matching FAQ question by keyword relevance
def find_best_matching_question(user_query, faqs):
    best_match = None
    highest_score = 0
    
    for faq in faqs:
        match_score = calculate_match_score(user_query, faq["keywords"])
        if match_score > highest_score:
            highest_score = match_score
            best_match = faq
    
    return best_match

# Route for home page
@app.route("/")
def index():
    return render_template('index.html')

# Route to handle user queries
@app.route("/get", methods=["POST"])
def handle_query():
    user_query = request.form["msg"]
    
    response = None
    
    for category, faqs in faq_data.items():
        best_match = find_best_matching_question(user_query, faqs)
        if best_match:
            response = best_match["answer"]
            break
    
    if not response:
        inputs = tokenizer(user_query, return_tensors='pt', padding=True, truncation=True)
        outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=1)
        confidence, predicted_class = torch.max(probs, dim=1)
        predicted_class = predicted_class.item()
        
        # Check confidence level and provide appropriate response
        if confidence.item() > 0.7:  # Adjust threshold as needed
            response = fallback_responses.get(predicted_class, fallback_responses['default'])
        else:
            response = fallback_responses['default']
    
    return response

if __name__ == '__main__':
    app.run(debug=True)
