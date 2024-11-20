import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from collections import Counter
import re
import matplotlib.pyplot as plt

st.title("Sentiment Analyzer Based On Text Analysis")
st.subheader("adapted from https://github.com/patidarparas13/Sentiment-Analyzer-Tool")
st.write('\n\n')

@st.cache_data
def get_all_data():
    root = "Datasets/"
    with open(root + "imdb_labelled.txt", "r") as text_file:
        data = text_file.read().split('\n')
         
    with open(root + "amazon_cells_labelled.txt", "r") as text_file:
        data += text_file.read().split('\n')

    with open(root + "yelp_labelled.txt", "r") as text_file:
        data += text_file.read().split('\n')

    return data

@st.cache_data
def preprocessing_data(data):
    processing_data = []
    for single_data in data:
        if len(single_data.split("\t")) == 2 and single_data.split("\t")[1] != "":
            # Convert label to int immediately during preprocessing
            text, label = single_data.split("\t")
            processing_data.append([text, int(label)])
    return processing_data

@st.cache_data
def split_data(data):
    total = len(data)
    training_ratio = 0.75
    training_data = []
    evaluation_data = []

    for indice in range(0, total):
        if indice < total * training_ratio:
            training_data.append(data[indice])
        else:
            evaluation_data.append(data[indice])

    return training_data, evaluation_data

@st.cache_data
def preprocessing_step():
    data = get_all_data()
    processing_data = preprocessing_data(data)
    return split_data(processing_data)

@st.cache_resource
def create_and_train_model():
    """Create and train both vectorizer and classifier"""
    training_data, _ = preprocessing_step()
    
    # Get text and labels
    training_text = [data[0] for data in training_data]
    training_result = [data[1] for data in training_data]  # Labels are already integers
    
    # Create and fit vectorizer
    vectorizer = CountVectorizer(binary=True)
    training_text = vectorizer.fit_transform(training_text)
    
    # Train classifier
    classifier = BernoulliNB().fit(training_text, training_result)
    
    return vectorizer, classifier

def analyze_dataset_statistics(data):
    """Analyze statistics of the training dataset"""
    # Extract text and labels (labels are already integers)
    texts = [item[0] for item in data]
    labels = [item[1] for item in data]
    
    # Calculate basic statistics
    total_reviews = len(texts)
    positive_reviews = sum(labels)
    negative_reviews = total_reviews - positive_reviews
    
    # Calculate average text lengths
    text_lengths = [len(text.split()) for text in texts]
    avg_length_positive = np.mean([length for length, label in zip(text_lengths, labels) if label == 1])
    avg_length_negative = np.mean([length for length, label in zip(text_lengths, labels) if label == 0])
    
    return {
        'total': total_reviews,
        'positive': positive_reviews,
        'negative': negative_reviews,
        'avg_length_positive': avg_length_positive,
        'avg_length_negative': avg_length_negative,
        'text_lengths': text_lengths,
        'labels': labels
    }

def create_visualizations(training_data, evaluation_data, vectorizer, classifier):
    """Create and display various visualizations using Plotly"""
    
    st.subheader("Data Analysis and Model Performance Visualizations")
    
    # 1. Dataset Distribution Pie Chart
    stats = analyze_dataset_statistics(training_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Reviews", stats['total'])
        st.metric("Average Words in Positive Reviews", f"{stats['avg_length_positive']:.1f}")
    
    with col2:
        st.metric("Positive/Negative Ratio", f"{stats['positive']/stats['negative']:.2f}")
        st.metric("Average Words in Negative Reviews", f"{stats['avg_length_negative']:.1f}")
    
    fig_distribution = px.pie(
        values=[stats['positive'], stats['negative']],
        names=['Positive', 'Negative'],
        title='Distribution of Sentiments in Training Data',
        color_discrete_sequence=['#00CC96', '#EF553B']
    )
    st.plotly_chart(fig_distribution, use_container_width=True)
    
    # 2. Review Length Distribution
    fig_lengths = px.histogram(
        x=stats['text_lengths'],
        color=[('Positive' if label == 1 else 'Negative') for label in stats['labels']],
        title='Distribution of Review Lengths by Sentiment',
        labels={'x': 'Review Length (words)', 'y': 'Count'},
        nbins=50,
        color_discrete_sequence=['#00CC96', '#EF553B']
    )
    st.plotly_chart(fig_lengths, use_container_width=True)
    
    # 3. Most Common Words Visualization
    def get_top_words(texts, label_filter):
        all_text = ' '.join([text for text, label in zip([item[0] for item in training_data], 
                                                        [item[1] for item in training_data]) 
                            if label == label_filter])
        words = re.findall(r'\w+', all_text.lower())
        word_counts = Counter(words)
        # Remove common English stop words and short words
        stop_words = {'the', 'and', 'is', 'in', 'to', 'a', 'of', 'for', 'it', 'on', 'i', 
                     'this', 'was', 'that', 'with', 'are', 'be', 'as', 'you', 'have'}
        word_counts = {word: count for word, count in word_counts.items() 
                      if word not in stop_words and len(word) > 2}
        return dict(sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:15])

    col1, col2 = st.columns(2)
    
    with col1:
        pos_words = get_top_words(training_data, 1)  # Changed from '1' to 1
        fig_pos_words = go.Figure(data=[
            go.Bar(x=list(pos_words.keys()), 
                  y=list(pos_words.values()), 
                  marker_color='#00CC96')
        ])
        fig_pos_words.update_layout(
            title='Top Words in Positive Reviews',
            xaxis_title='Words',
            yaxis_title='Frequency',
            xaxis_tickangle=45
        )
        st.plotly_chart(fig_pos_words, use_container_width=True)
    
    with col2:
        neg_words = get_top_words(training_data, 0)  # Changed from '0' to 0
        fig_neg_words = go.Figure(data=[
            go.Bar(x=list(neg_words.keys()), 
                  y=list(neg_words.values()), 
                  marker_color='#EF553B')
        ])
        fig_neg_words.update_layout(
            title='Top Words in Negative Reviews',
            xaxis_title='Words',
            yaxis_title='Frequency',
            xaxis_tickangle=45
        )
        st.plotly_chart(fig_neg_words, use_container_width=True)
    
    # 4. Model Performance Metrics
    st.subheader("Model Performance Analysis")
    
    # Calculate predictions on evaluation data
    eval_texts = [item[0] for item in evaluation_data]
    eval_labels = [item[1] for item in evaluation_data]  # Labels are already integers
    eval_vectors = vectorizer.transform(eval_texts)
    predictions = classifier.predict(eval_vectors)
    
    # Create confusion matrix
    cm = confusion_matrix(eval_labels, predictions)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Negative', 'Positive'],
            y=['Negative', 'Positive'],
            title='Confusion Matrix',
            color_continuous_scale='RdBu'
        )
        st.plotly_chart(fig_cm, use_container_width=True)
    
    # 5. Prediction Confidence Distribution
    with col2:
        confidences = classifier.predict_proba(eval_vectors)
        positive_confidences = [conf[1] for conf in confidences]
        
        fig_conf = px.histogram(
            x=positive_confidences,
            title='Model Confidence Distribution',
            labels={'x': 'Confidence Score (Positive)', 'y': 'Count'},
            nbins=50,
            color_discrete_sequence=['#636EFA']
        )
        st.plotly_chart(fig_conf, use_container_width=True)
    
def analyse_text(classifier, vectorizer, text):
    """Analyze the sentiment of a given text"""
    prediction = classifier.predict(vectorizer.transform([text]))
    return text, prediction[0]  # Return integer prediction

def print_result(result):
    """Format the sentiment analysis result"""
    text, analysis_result = result
    sentiment = "Positive" if analysis_result == 1 else "Negative"  # Compare with integer
    return text, sentiment

# Main application flow
all_data = get_all_data()

# Data viewing options
with st.expander("View Dataset"):
    if st.checkbox('Show Raw Dataset'):
        st.write(all_data)
    if st.checkbox('Show Preprocessed Dataset'):
        st.write(preprocessing_data(all_data))

# Create and train model (will be cached)
vectorizer, classifier = create_and_train_model()

# Visualization section
if st.checkbox('Show Data Analysis and Model Performance'):
    training_data, evaluation_data = preprocessing_step()
    create_visualizations(training_data, evaluation_data, vectorizer, classifier)

# Sentiment Analysis Input Section
st.subheader("Analyze Your Text")
review = st.text_area("Enter Text for Analysis", "Write your review here...", height=100)

if st.button('Analyze Sentiment'):
    if review != "Write your review here...":
        result = print_result(analyse_text(classifier, vectorizer, review))
        sentiment = result[1]
        confidence = classifier.predict_proba(vectorizer.transform([review]))[0]
        
        # Calculate the gauge value: transform from [P(neg), P(pos)] to [-100 to 100] scale
        # When P(pos) is high (close to 1), gauge will be close to 100
        # When P(neg) is high (close to 1), gauge will be close to -100
        gauge_value = (confidence[1] - confidence[0]) * 100
        
        confidence_level = abs(gauge_value)
        if confidence_level > 66:
            confidence_text = "High Confidence"
        elif confidence_level > 33:
            confidence_text = "Moderate Confidence"
        else:
            confidence_text = "Low Confidence"
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Sentiment", sentiment)
        with col2:
            st.metric("Confidence", f"{confidence_text}")
        
        # Create sentiment polarity gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = gauge_value,
            title = {'text': "Sentiment Polarity"},
            gauge = {
                'axis': {
                    'range': [-100, 100],
                    'tickmode': 'array',
                    'ticktext': ['Negative', 'Neutral', 'Positive'],
                    'tickvals': [-100, 0, 100],
                    'tickangle': 0,
                },
                'bar': {'color': "#636EFA"},
                'steps': [
                    {'range': [-100, -33], 'color': "#EF553B"},  # Red for negative
                    {'range': [-33, 33], 'color': "#FFA15A"},    # Orange for neutral
                    {'range': [33, 100], 'color': "#00CC96"}     # Green for positive
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 2},
                    'thickness': 0.75,
                    'value': 0
                }
            },
            number = {
                'suffix': '%',
                'font': {'size': 24}
            },
            domain = {'x': [0, 1], 'y': [0, 1]}
        ))
        
        # Update layout for better readability
        fig.update_layout(
            height=300,
            font={'size': 16, 'color': "#444"},
            margin=dict(l=30, r=30, t=100, b=30)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
else:
    st.write("Press the above button to analyze your text.")
