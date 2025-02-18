import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def clean_text(text):
    """Normalize text for better matching."""
    text = str(text).strip().lower()
    if text in ["-", "no", "nan", "none", ""]:
        return ""
    return text.replace(";", ",").replace("\n", " ").strip()

def calculate_text_similarity(series1, series2):
    """Calculate text similarity using TF-IDF and cosine similarity."""
    vectorizer = TfidfVectorizer()
    combined = pd.concat([series1, series2], ignore_index=True).fillna("")
    tfidf_matrix = vectorizer.fit_transform(combined)
    similarity_matrix = cosine_similarity(tfidf_matrix[:len(series1)], tfidf_matrix[len(series1):])
    return similarity_matrix

def match_mentees_to_mentors(df_mentees, df_mentors):
    """Find best matches between mentees and mentors."""
    # Normalize relevant columns
    for col in ["Languages", "Faculty", "Interests", "Personality Type"]:
        df_mentees[col] = df_mentees[col].apply(clean_text)
        df_mentors[col] = df_mentors[col].apply(clean_text)
    
    # Compute similarity scores
    lang_sim = calculate_text_similarity(df_mentees["Languages"], df_mentors["Languages"])
    faculty_sim = calculate_text_similarity(df_mentees["Faculty"], df_mentors["Faculty"])
    interests_sim = calculate_text_similarity(df_mentees["Interests"], df_mentors["Interests"])
    personality_sim = calculate_text_similarity(df_mentees["Personality Type"], df_mentors["Personality Type"])
    
    # Weighted matching score
    match_score = (0.3 * lang_sim + 0.3 * faculty_sim + 0.2 * interests_sim + 0.2 * personality_sim)
    
    matches = []
    for i, mentee in df_mentees.iterrows():
        best_mentor_idx = np.argmax(match_score[i])
        best_mentor = df_mentors.iloc[best_mentor_idx]
        matches.append({
            "Mentee": mentee["Name"],
            "Mentee Faculty": mentee["Faculty"],
            "Mentee Languages": mentee["Languages"],
            "Mentee Interests": mentee["Interests"],
            "Best Matched Mentor": best_mentor["Name"],
            "Mentor Faculty": best_mentor["Faculty"],
            "Mentor Languages": best_mentor["Languages"],
            "Mentor Interests": best_mentor["Interests"],
            "Match Score": match_score[i, best_mentor_idx]
        })
    
    return pd.DataFrame(matches)

# Streamlit App
st.title("Mentor-Mentee Matching App")

st.sidebar.header("Upload Files")
mentees_file = st.sidebar.file_uploader("Upload Mentees Excel File", type=["xlsx", "csv"])
mentors_file = st.sidebar.file_uploader("Upload Mentors Excel File", type=["xlsx", "csv"])

if mentees_file and mentors_file:
    df_mentees = pd.read_excel(mentees_file) if mentees_file.name.endswith(".xlsx") else pd.read_csv(mentees_file)
    df_mentors = pd.read_excel(mentors_file) if mentors_file.name.endswith(".xlsx") else pd.read_csv(mentors_file)
    
    df_matches = match_mentees_to_mentors(df_mentees, df_mentors)
    
    st.subheader("Best Mentor-Mentee Matches")
    filter_score = st.slider("Filter by Match Score", min_value=0.0, max_value=1.0, value=(0.0, 1.0))
    df_filtered = df_matches[(df_matches["Match Score"] >= filter_score[0]) & (df_matches["Match Score"] <= filter_score[1])]
    
    edited_df = st.data_editor(df_filtered, num_rows="dynamic")
    
    # Download button
    csv = edited_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Matches as CSV", data=csv, file_name="matches.csv", mime="text/csv", key="download-csv")
