# Import libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.offsetbox as offsetbox
import numpy as np


# Read in data to be used in the app
responses = pd.read_csv('responses.csv')
question_map = pd.read_csv('question_map.csv')

# Create a title for the app
st.title('Quadrant Analysis Generator')
st.write('Start by selecting two questions to compare the answers.')

# Create a selectbox to choose the first question from the 'Question Text' column of the question_map dataframe
question1 = st.selectbox('Select the first question', question_map['Question Text'].unique())

# Create a selectbox to choose the second question from the 'Question Text' column of the question_map dataframe
question2 = st.selectbox('Select the second question', question_map['Question Text'].unique())

# Create a new dataframe called compared responses. 
# The dataframe should use the 'Name' column of the responses dataframe as the index.
# The dataframe should merge on the first row of the responses dataframe with the first column of the question_map dataframe.
# The dataframe should use each question's 'Question Text' as the column names, and the responses as the values.
# Get the question ID for each selected question
question1_id = question_map[question_map['Question Text'] == question1]['Question ID'].values[0]
question2_id = question_map[question_map['Question Text'] == question2]['Question ID'].values[0]

# Extract the relevant columns from the responses dataframe
compared_responses = responses[['Name', question1_id, question2_id]]

# Rename the columns to the actual question text
compared_responses.columns = ['Name', question1, question2]

# Show the dataframe in the app
st.write(compared_responses)

def identify_overlapping_points(df, x_col, y_col):
    overlaps = {}
    for i, row in df.iterrows():
        coords = (row[x_col], row[y_col])
        if coords in overlaps:
            overlaps[coords].append(i)
        else:
            overlaps[coords] = [i]
    return overlaps


def plot_modified_quadrant_chart(question1, question2, question_map, compared_responses):
    # Extract the 'Subject A' and 'Subject B' values for the two questions
    subject_a_1 = question_map[question_map['Question Text'] == question1]['Subject A'].values[0]
    subject_b_1 = question_map[question_map['Question Text'] == question1]['Subject B'].values[0]
    
    subject_a_2 = question_map[question_map['Question Text'] == question2]['Subject A'].values[0]
    subject_b_2 = question_map[question_map['Question Text'] == question2]['Subject B'].values[0]
    
    plt.figure(figsize=(10, 8))

    # Identify overlapping points
    overlaps = identify_overlapping_points(compared_responses, question1, question2)
    
    # Define a function to compute offsets for overlapping images
    def compute_offsets(num_overlaps):
        if num_overlaps == 1:
            return [(0, 0)]
        angle = np.linspace(0, 2*np.pi, num_overlaps, endpoint=False)
        radius = 0.3  # You can adjust this value as needed
        return [(radius*np.cos(a), radius*np.sin(a)) for a in angle]


    # For each unique coordinate, plot the corresponding images
    for coords, indices in overlaps.items():
        offsets = compute_offsets(len(indices))
        for i, idx in enumerate(indices):
            # Load the image
            img = plt.imread(f'./images/{compared_responses["Name"].iloc[idx]}.png')
            imagebox = offsetbox.OffsetImage(img, zoom=0.035)  # adjust zoom as needed
            ab = offsetbox.AnnotationBbox(imagebox, (coords[0] + offsets[i][0], coords[1] + offsets[i][1]), frameon=False)
            plt.gca().add_artist(ab)
            # Draw a line from the offset image position to the original point
            plt.plot([coords[0], coords[0] + offsets[i][0]], 
                     [coords[1], coords[1] + offsets[i][1]], 
                     color='grey', linestyle='--')
    
    # Add quadrant lines
    plt.axhline(5, color='grey', linestyle='--')
    plt.axvline(5, color='grey', linestyle='--')
    
    # Adjusted tick marks for values 0 through 10
    ticks = list(range(0, 11))
    plt.xticks(ticks)
    plt.yticks(ticks)
    
    # Adjust grid style
    plt.grid(True, which='both', linestyle='-', linewidth=0.5, color='lightgray')
    
    ax = plt.gca()
    ax.tick_params(axis="x", which="both", bottom=True, top=True, labelbottom=True, labeltop=True)
    ax.tick_params(axis="y", which="both", left=True, right=True, labelleft=True, labelright=True)
    
    # Manually place text labels on the top, left, and right sides of the plot
    ax.text(1.07, 0.5, question2, va='center', ha='left', transform=ax.transAxes)
    ax.text(0.5, 1.08, question1, va='bottom', ha='center', transform=ax.transAxes)
    ax.text(-0.07, 0.5, question2, va='center', ha='right', transform=ax.transAxes)
    
    # Place the 'Subject A' and 'Subject B' labels closer to the dashed line
    ax.text(5, 11, subject_b_2, ha='center', va='top', fontsize=15, fontweight='bold', color='red')
    ax.text(5, -1, subject_a_2, ha='center', va='bottom', fontsize=15, fontweight='bold', color='red')
    ax.text(-1, 5.3, subject_a_1, ha='left', va='center', fontsize=15, fontweight='bold', color='red')
    ax.text(11, 5.3, subject_b_1, ha='right', va='center', fontsize=15, fontweight='bold', color='red')
    
    # Set title
    ax.set_title(f"{question1} vs. {question2}", y=1.15)
    
    # Make the outside of the chart thicker/bolder
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    
    plt.xlim(-1, 11)  # Adjusted x-axis limits for added margin
    plt.ylim(-1, 11)  # Adjusted y-axis limits for added margin
    
    # Return the plot
    return plt

plt = plot_modified_quadrant_chart(question1, question2, question_map, compared_responses)
st.pyplot(plt)


# We will define the new component as a function that can be called in your Streamlit app.

def compare_respondents(responses):
    # Step 1: Accept Two 'Name' Inputs
    names = responses['Name'].unique()
    name1 = st.selectbox('Select the first respondent', names)
    name2 = st.selectbox('Select the second respondent', names)
    
    if name1 == name2:
        st.write("Please select two different respondents.")
        return
    
    # Extract responses for the two selected names
    resp1 = responses[responses['Name'] == name1].drop(columns='Name').iloc[0]
    resp2 = responses[responses['Name'] == name2].drop(columns='Name').iloc[0]
    
    # Step 2: Calculate Differences on Each Question
    differences = abs(resp1 - resp2)
    
    # Step 3: Identify the Question with the Smallest Difference
    min_diff_question = differences.idxmin()
    min_diff_value = differences[min_diff_question]
    
    # Display results
    st.write(f"{name1} and {name2} agreed the most on the question '{min_diff_question}' with a difference of {min_diff_value}.")
    st.write("Here are the differences on each question:")
    st.write(differences)

    return differences

# For the sake of demonstration, we'll return the differences. 
# However, in your Streamlit app, you can simply call the function to display the results.

compare_respondents(responses)

st.write("Here are the differences on each question:")
st.write(differences)

