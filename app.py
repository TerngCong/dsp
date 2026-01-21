import streamlit as st
from transformers import pipeline
import pandas as pd
import numpy as np
import ast
from dotenv import load_dotenv
import os
import google.generativeai as genai
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

if API_KEY:
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash-lite", 
        system_instruction="You are a SDG and sustainability sentiment analyser. You explain sentiments based on provided text and classifier results."
    )

@st.cache_resource
def load_classifier():
    return pipeline("text-classification", model="distilBERT model")

try:
    classifier = load_classifier()
except Exception as e:
    st.error(f"Could not load model. Check if 'distilBERT model' folder exists. Error: {e}")
    classifier = None

@st.cache_data
def load_data():
    df = pd.read_csv("final df.csv")
    return df

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Sentiment Analysis", "Data Analysis", "About the Dataset"])

if page == "Sentiment Analysis":
    st.title("Sentiment Analysis Model")

    LABEL_MAP = {
        "LABEL_0": "Negative",
        "LABEL_1": "Neutral",
        "LABEL_2": "Positive"
    }

    def explain_prediction(text, raw_result):
        top_result = raw_result[0]
        label_code = top_result['label']
        score = top_result['score']
        human_label = LABEL_MAP.get(label_code, label_code)
        
        prompt = f"""
        You are an SDG and sustainability sentiment explanation assistant.

        Input Text: "{text}"
        Predicted Sentiment: {human_label}
        Confidence Score: {score:.2f}

        Instructions:
        - First, validate the input text and select ONE appropriate response only.

        Validation Rules:
        1. If the text is empty, null, or contains only whitespace:
        Output exactly: "Please enter your text for analysis."

        2. If the text is not written in English:
        Output exactly: "Please enter the text in English for analysis."

        3. If the text does not relate to sustainability, Sustainable Development Goals (SDGs), or environmental, social, or governance topics:
        Output exactly: "The text does not appear to be related to SDGs or sustainability."

        4. If the confidence score is below 0.40:
        Output exactly: "The model is not confident enough to provide an explanation."

        Task:
        - If and only if none of the above conditions apply, explain in exactly 1–2 short sentences why the text was classified as "{human_label}".
        - Base the explanation strictly on the input text.
        - Do not mention confidence scores, validation rules, or assumptions.
        - Do not add recommendations or extra commentary.

        Output:
        - Return only one sentence or two sentences, or one of the exact messages above.
        """
        response = model.generate_content(prompt)
        return response.text.strip().strip('"')
    
    text = st.text_area("Enter text to analyze", height=150)

    if st.button("Analyze"):
        if not text.strip():
            st.warning("Please enter some text before analyzing.")
        elif classifier:
            result = classifier(text) 
            label_code = result[0]['label']
            human_label = LABEL_MAP.get(label_code, label_code)
            confidence = result[0]['score']

            st.subheader("Result")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Sentiment", human_label)
            with col2:
                st.metric("Confidence", f"{confidence:.2%}")
            
            if API_KEY:
                with st.spinner("Asking Gemini for an explanation..."):
                    try:
                        explanation = explain_prediction(text, result)
                        st.write("**Why?**")
                        st.info(explanation)
                    except Exception as e:
                        st.error(f"Gemini Error: {e}")
            else:
                st.warning("Gemini API Key not found. Cannot generate explanation.")
        else:
            st.error("Model not loaded.")

elif page == "Data Analysis":
    st.title("Data Analysis Dashboard")
    
    try:
        df = load_data()
        
        st.subheader("Filter Data")
        faculty_list = df["Which faculty are you from?"].unique().tolist()
        
        with st.expander("Click to Filter by Faculty", expanded=False):
            selected_faculty = st.multiselect(
                "Select Faculty to View", 
                options=faculty_list, 
                default=faculty_list
            )

        if not selected_faculty:
            st.warning("Please select at least one faculty. Currently showing all faculties.")
            filtered_df = df
        else:
            filtered_df = df[df["Which faculty are you from?"].isin(selected_faculty)]
            
        st.markdown("---") 
        st.caption(f"Showing results for {len(filtered_df)} students.")

        # sentiment distribution
        st.subheader("Sentiment Distribution")
        
        sentiment_counts = filtered_df['label'].value_counts()
        
        if not sentiment_counts.empty:
            fig_sent, ax_sent = plt.subplots(figsize=(10, 4))
            
            sentiment_counts.sort_values(ascending=True).plot(
                kind='barh', 
                color='#4CAF50', 
                ax=ax_sent
            )
            
            ax_sent.bar_label(ax_sent.containers[0], fmt='%d', padding=3)
            ax_sent.set_title("Sentiment Breakdown")
            ax_sent.set_xlabel("Number of Students")
            ax_sent.set_ylabel("Label")
            st.pyplot(fig_sent)
        else:
            st.info("No data available.")

        # top discussed sdg topics
        st.subheader("Top 5 Discussed SDG Topics")
        def parse_list(x):
            try:
                return ast.literal_eval(x)
            except:
                return []
            
        sdg_df = filtered_df.copy()
        sdg_df['sdg_real_list'] = sdg_df['sdg_topics_multi'].apply(parse_list)

        exploded_sdgs = sdg_df.explode('sdg_real_list')
        top_sdgs = exploded_sdgs['sdg_real_list'].value_counts().head(5)

        if not top_sdgs.empty:
            fig_sdg, ax_sdg = plt.subplots(figsize=(10, 5))
            
            top_sdgs.sort_values(ascending=True).plot(
                kind='barh', 
                color='#4CAF50', 
                ax=ax_sdg
            )
            
            ax_sdg.bar_label(ax_sdg.containers[0], fmt='%d', padding=3)
            ax_sdg.set_title("Most Common SDG Topics")
            ax_sdg.set_xlabel("Number of Mentions")
            ax_sdg.set_ylabel("SDG topics")
            st.pyplot(fig_sdg)
        else:
            st.info("No SDG data found for this selection.")

        # word cloud
        st.subheader("What are students saying?")

        col_input, col_cloud = st.columns([1, 3])
            
        with col_input:
            st.markdown("**Word Cloud Settings**")
            user_ignored_words = st.text_input(
                "Add words to exclude", 
                placeholder="e.g. campus, goal"
            )

        stopwords = set(STOPWORDS)
        stopwords.update(["comment", "especially", "honestly", "actually", "quite", "dont", "think", "dont know", "um", "using", "lot", "really", "maybe", "know"])
        if user_ignored_words:
            new_words = [word.strip() for word in user_ignored_words.split(",")]
            stopwords.update(new_words)

        text_data = " ".join(filtered_df['wc_text'].dropna().astype(str))

        with col_cloud:
            if text_data:
                wc = WordCloud(
                    width=800, 
                    height=400, 
                    background_color='white',
                    colormap='viridis',
                    stopwords=stopwords
                ).generate(text_data)

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wc, interpolation='bilinear')
                ax.axis("off") 
                st.pyplot(fig)
            else:
                st.info("No text data available for this selection.")
    
    except FileNotFoundError:
        st.error("Could not find 'final df.csv'. Please check the file name.")

elif page == "About the Dataset":
    st.title("About the Dataset")
    
    try:
        df = load_data()
        
        st.write(f"**Total Respondents:** {len(df)}")
        st.markdown("---")

        # faculty distribution
        st.subheader("1. Participants by Faculty")
        
        faculty_counts = df['Which faculty are you from?'].value_counts()
        
        if not faculty_counts.empty:
            fig_fac, ax_fac = plt.subplots(figsize=(10, 6))
            
            faculty_counts.sort_values(ascending=True).plot(
                kind='barh', 
                color='#4CAF50', 
                ax=ax_fac
            )
            
            ax_fac.bar_label(ax_fac.containers[0], fmt='%d', padding=3)
            ax_fac.set_title("Number of Students per Faculty")
            ax_fac.set_xlabel("Count")
            ax_fac.set_xlim(0, faculty_counts.max() * 1.15)
            
            st.pyplot(fig_fac)

        st.markdown("---")

        # Year and rating of um
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("2. Year of Study")
            year_counts = df['What is your current year of study?'].value_counts().sort_index()
            
            if not year_counts.empty:
                fig_year, ax_year = plt.subplots(figsize=(6, 5))
                year_counts.plot(kind='bar', color='#4CAF50', ax=ax_year)
                
                ax_year.bar_label(ax_year.containers[0], fmt='%d', padding=3)
                ax_year.set_title("Participants by Year")
                ax_year.set_ylabel("Count")
                ax_year.tick_params(axis='x', rotation=0)
                ax_year.set_ylim(0, year_counts.max() * 1.15)
                
                st.pyplot(fig_year)
                
        with col2:
            st.subheader("3. Satisfaction Level")
            sat_counts = df['How satisfied are you with UM’s sustainability initiatives?  '].value_counts().sort_index()  

            if not sat_counts.empty:
                fig_sat, ax_sat = plt.subplots(figsize=(6, 5))
                sat_counts.plot(kind='bar', color='#4CAF50', ax=ax_sat)
                
                ax_sat.bar_label(ax_sat.containers[0], fmt='%d', padding=3)
                ax_sat.set_title("Satisfaction Ratings")
                ax_sat.set_ylabel("Count")
                ax_sat.tick_params(axis='x', rotation=0)
                ax_sat.set_ylim(0, sat_counts.max() * 1.15)
                
                st.pyplot(fig_sat)

        st.markdown("---")

        # participation in activities
        st.subheader("4. Participation in Sustainability Activities")
        part_counts = df['Have you ever participated in sustainability-related activities or programs at UM?  '].value_counts()
        
        if not part_counts.empty:
            fig_pie, ax_pie = plt.subplots(figsize=(5, 5))
            
            ax_pie.pie(
                part_counts, 
                labels=part_counts.index, 
                autopct='%1.1f%%', 
                startangle=90, 
                colors=plt.cm.Greens(np.linspace(0.4, 0.8, 3)),
            )
            
            ax_pie.set_title("Have students participated in sustainability-related activities?")
            st.pyplot(fig_pie)

    except FileNotFoundError:
        st.error("Could not find 'final df.csv'.")