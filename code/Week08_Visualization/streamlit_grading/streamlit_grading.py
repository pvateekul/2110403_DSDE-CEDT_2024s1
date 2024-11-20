import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def get_grade(score, grade_min):
    """Convert score to letter grade based on minimum score criteria"""
    if score >= grade_min['A']: return 'A'
    elif score >= grade_min['B']: return 'B'
    elif score >= grade_min['C']: return 'C'
    elif score >= grade_min['D']: return 'D'
    else: return 'F'

def main():
    # Main page title
    st.title('Simple Grading App')
    
    # Sidebar with grade criteria
    st.sidebar.header('Grade Criteria')
    grade_min = {
        'A': st.sidebar.number_input('Minimum score for A:', value=90),
        'B': st.sidebar.number_input('Minimum score for B:', value=80),
        'C': st.sidebar.number_input('Minimum score for C:', value=70),
        'D': st.sidebar.number_input('Minimum score for D:', value=60)
    }
    
    # Input section
    st.header('Enter Scores')
    input_method = st.radio('Choose input method:', ['Manual Entry', 'File Upload'])
    
    scores = None
    
    if input_method == 'Manual Entry':
        scores_text = st.text_area(
            "Enter scores (one per line or comma-separated):",
            height=100
        )
        if scores_text:
            if ',' in scores_text:
                scores = [float(x) for x in scores_text.split(',')]
            else:
                scores = [float(x) for x in scores_text.split('\n') if x.strip()]
    
    else:
        uploaded_file = st.file_uploader("Upload CSV file with scores:", type='csv')
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            scores = df.iloc[:, 0].tolist()
    
    if scores:
        # Convert to numpy array for calculations
        scores = np.array(scores)
        
        # Display basic statistics
        st.header('Class Statistics')
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average", f"{np.mean(scores):.1f}")
        with col2:
            st.metric("Median", f"{np.median(scores):.1f}")
        with col3:
            st.metric("Count", len(scores))
        
        # Create histogram with adjustable bins
        st.header('Score Distribution')
        
        # Add bin size slider
        bin_size = st.slider(
            "Adjust histogram bin size:",
            min_value=1,
            max_value=10,
            value=5,
            help="Smaller values show more detail, larger values show general trends"
        )
        
        # Calculate number of bins based on bin size
        num_bins = int((np.max(scores) - np.min(scores)) / bin_size) + 1
        
        fig = px.histogram(
            scores,
            nbins=num_bins,
            title='Distribution of Scores',
            labels={'value': 'Score', 'count': 'Number of Students'}
        )
        
        # Add grade boundary lines
        for grade, min_score in grade_min.items():
            fig.add_vline(
                x=min_score,
                line_dash="dash",
                line_color="gray",
                annotation_text=grade
            )
        
        st.plotly_chart(fig)
        
        # Calculate grades using current criteria
        grades = [get_grade(score, grade_min) for score in scores]
        grade_counts = pd.Series(grades).value_counts().sort_index()
        
        # Display grade distribution
        st.header('Grade Distribution')
        st.bar_chart(grade_counts)
        
        # Show detailed results
        st.header('Detailed Results')
        results_df = pd.DataFrame({
            'Score': scores,
            'Grade': grades
        })
        st.dataframe(results_df)
        
        # Export option
        st.download_button(
            "Download Results",
            results_df.to_csv(index=False),
            "grades.csv",
            "text/csv"
        )

if __name__ == '__main__':
    main()
