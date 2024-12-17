import re
import requests
import tldextract
import pandas as pd
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib


# Load the trained model and scaler
model = joblib.load('svm_model.pkl')  # Replace with the path to your trained model
scaler = joblib.load('scaler.pkl')  # Replace with the path to your scaler
# Load the trained model and label encoder
model_path = 'svm_email_classifier.pkl'
encoder_path = 'label_encoder.pkl'
model2 = joblib.load(model_path)  # Changed 'model' to 'model2'
label_encoder = joblib.load(encoder_path)


# Horizontal navigation using tabs
tabs = st.tabs(["Website detector", "Email Detector"])

# Home Tab
with tabs[0]:
    trusted_tlds = ['edu', 'gov', 'org']
    trusted_domains = [
        "youtube.com", "twitch.tv", "facebook.com", "instagram.com", "twitter.com",
        "linkedin.com", "github.com", "reddit.com", "tiktok.com", "snapchat.com",
        "pinterest.com", "whatsapp.com", "vimeo.com", "discord.com", "tumblr.com"
    ]


    # Function to calculate URL Similarity Index based on full URL and domain
    def extract_url_similarity_index(url):
        ext = tldextract.extract(url)
        domain = f"{ext.domain}.{ext.suffix}"
        similarity = SequenceMatcher(None, url.lower(), domain.lower()).ratio()
        return similarity * 100


    # Function to extract features from URL
    def extract_url_features(url):
        features = {}
        ext = tldextract.extract(url)
        domain = f"{ext.domain}.{ext.suffix}"
        features['Domain'] = f"{ext.subdomain}.{ext.domain}.{ext.suffix}".strip('.')

        # Convert TLD to numerical and binary values
        tld_mapping = {
            'com': 0, 'org': 1, 'net': 2, 'edu': 3, 'gov': 4, 'de': 5,
            'co': 6, 'info': 7, 'io': 8, 'biz': 9
        }
        features['TLD'] = tld_mapping.get(ext.suffix, -1)
        features['IsTrustedTLD'] = 1 if ext.suffix in trusted_tlds else 0

        # Special character count and ratio
        special_chars = re.findall(r'[@$&%*!#]', url)
        features['NoOfOtherSpecialCharsInURL'] = len(special_chars)
        features['SpacialCharRatioInURL'] = len(special_chars) / len(url) if url else 0

        # Check HTTPS
        features['IsHTTPS'] = 1 if url.startswith("https") else 0

        # URL Similarity Index
        features['URLSimilarityIndex'] = extract_url_similarity_index(url)

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            html_content = response.text
            soup = BeautifulSoup(html_content, 'html.parser')

            # Extract features from HTML
            features['LineOfCode'] = len(html_content.splitlines())
            title = soup.title.string.strip() if soup.title else "No Title"
            features['Title'] = title

            # Domain-Title and URL-Title Match Scores
            domain_keywords = ext.domain.lower().split('-')
            title_words = title.lower().split() if title != "No Title" else []
            url_keywords = re.split(r'[/:._-]', url.lower())
            features['DomainTitleMatchScore'] = (len([word for word in title_words if word in domain_keywords]) / len(
                domain_keywords) * 100) if domain_keywords else 0
            features['URLTitleMatchScore'] = (len([word for word in title_words if word in url_keywords]) / len(
                url_keywords) * 100) if url_keywords else 0

            # Other features from HTML
            features['NoOfImage'] = len(soup.find_all('img'))
            features['NoOfJS'] = len(soup.find_all('script'))
            features['NoOfSelfRef'] = len([a for a in soup.find_all('a', href=True) if features['Domain'] in a['href']])
            features['IsResponsive'] = 1
            features['HasDescription'] = 1 if soup.find('meta', attrs={'name': 'description'}) else 0
            social_keywords = ['facebook', 'twitter', 'instagram', 'linkedin']
            social_links = [a for a in soup.find_all('a', href=True) if
                            any(keyword in a['href'] for keyword in social_keywords)]
            features['HasSocialNet'] = 1 if social_links else 0
            features['HasSubmitButton'] = 1 if soup.find('button', {'type': 'submit'}) else 0
            features['HasCopyrightInfo'] = 1 if 'copyright' in soup.text.lower() else 0
        except Exception as e:
            print(f"Error fetching or parsing URL {url}: {e}")
            features.update({
                'LineOfCode': 0, 'Title': "No Title", 'DomainTitleMatchScore': 0, 'URLTitleMatchScore': 0,
                'NoOfImage': 0, 'NoOfJS': 0, 'NoOfSelfRef': 0, 'IsResponsive': 0,
                'HasDescription': 0, 'HasSocialNet': 0, 'HasSubmitButton': 0, 'HasCopyrightInfo': 0
            })

        selected_features = [
            'TLD', 'NoOfOtherSpecialCharsInURL', 'SpacialCharRatioInURL', 'IsHTTPS', 'URLSimilarityIndex',
            'LineOfCode', 'DomainTitleMatchScore', 'URLTitleMatchScore', 'NoOfImage', 'NoOfJS', 'NoOfSelfRef',
            'IsResponsive', 'HasDescription', 'HasSocialNet', 'HasSubmitButton', 'IsTrustedTLD'
        ]
        return {key: features[key] for key in selected_features}


    # Streamlit UI
    st.title("URL Feature Extractor and Predictor")
    url_input = st.text_input("Enter a Website URL", "https://www.uni-mainz.de")

    if url_input:
        ext = tldextract.extract(url_input)
        domain = f"{ext.domain}.{ext.suffix}"

        # Check for trusted domains and TLDs
        if ext.suffix in trusted_tlds or domain in trusted_domains:
            st.write(f"### Trusted Website: {domain} (TLD: {ext.suffix})")
            st.write("This website is classified as Good based on its trusted TLD or domain.")
            prediction = 0
            prediction_proba = [0.1, 0.9]
        else:
            st.write(f"### Extracting features for URL: {url_input}")
            features = extract_url_features(url_input)
            df = pd.DataFrame(list(features.items()), columns=["Feature", "Value"])
            st.table(df)

            # Exclude 'IsTrustedTLD' for scaler input
            features_for_scaler = {k: v for k, v in features.items() if k != 'IsTrustedTLD'}
            scaled_features = scaler.transform([list(features_for_scaler.values())])
            prediction = model.predict(scaled_features)[0]
            prediction_proba = model.predict_proba(scaled_features)[0]

            threshold = 0.8 if ext.suffix in trusted_tlds or domain in trusted_domains else 0.5
            prediction = 1 if prediction_proba[1] > threshold else 0

        prediction_text = "Bad Website" if prediction == 1 else "Good Website"
        st.write(f"### Prediction: {prediction_text}")
        st.write(f"Probability of being a Bad Website: {prediction_proba[1]:.4f}")
        st.write(f"Probability of being a Good Website: {prediction_proba[0]:.4f}")



# Email Detector Tab
with tabs[1]:
    st.title("Fake Email Detector")
    st.header("Enter Email Content")
    email_content = st.text_area("Paste the email content below:")

    if st.button("Detect"):
        if email_content.strip():  # Ensure the text area is not empty
            # Predict the label
            prediction = model2.predict([email_content])[0]  # Changed 'model' to 'model2'
            label = label_encoder.inverse_transform([prediction])[0]  # Decode label
            if label == "fake":
                st.error("🚨 This email is likely **FAKE**.")
            else:
                st.success("✅ This email is likely **GENUINE**.")
        else:
            st.warning("Please enter the email content to detect.")
