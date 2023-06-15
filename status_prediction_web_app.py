import pickle
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from streamlit_option_menu import option_menu

# Replace 'path_to_dataset' with the actual path to your dataset file
data = pd.read_csv(r'C:\Users\User\Downloads\archive\Placement_Data_Full_Class.csv')

# loading the saved model
loaded_model = pickle.load(open('C:/Users/User/Machine Learning subject/MLProject/SVM WEB/trained_model1.sav', 'rb'))

# function for prediction
def status_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)
    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    if prediction[0] == 0:
        return 'The person is not placed'
    else:
        return 'The person is placed'

# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Campus Placement Prediction System',
                          
                          ['Status Prediction',
                           'Visuals',
                           ],
                          icons=['activity','heart','person'],
                          default_index=0)
    
    
# Status Prediction Page
if selected == 'Status Prediction':
    # Page title
    st.title('Campus Placement Prediction')

    # Input fields
    st.subheader('Enter the details:')
    gender = st.text_input('Gender (Male: 1, Female: 0)')
    ssc_p = st.text_input('Secondary Education Percentage - 10th Grade')
    ssc_b = st.text_input('Board of Education (10th Grade) - Central: 1, Others: 0')
    hsc_p = st.text_input('Higher Secondary Education Percentage - 12th Grade')
    hsc_b = st.text_input('Board of Education (12th Grade) - Central: 1, Others: 0')
    hsc_s = st.text_input('Specialization in Higher Secondary Education - Commerce: 2, Science: 1, Arts: 0')
    degree_p = st.text_input('Degree Percentage')
    degree_t = st.text_input('Under Graduation Degree Type - Sci&tech: 1, Comm&Mgmt: 2, Others: 3')
    workex = st.text_input('Work Experience - Yes: 1, No: 0')
    etest_p = st.text_input('Employability Test Percentage')
    specialisation = st.text_input('Post Graduation Specialization (MBA) - Mkt&Fin: 1, Mkt&HR: 0')
    mba_p = st.text_input('MBA Percentage')

    # Code for prediction
    placement = ''
    if st.button('Predict Placement Status'):
        placement = status_prediction([gender, ssc_p, ssc_b, hsc_p, hsc_b, hsc_s, degree_p, degree_t, workex, etest_p, specialisation, mba_p])
    
    # Display prediction result
    if placement:
        st.subheader('Placement Status Result:')
        st.success(placement)



elif selected == 'Visuals':
    # page title
    st.title('Data Visualization for Campus Placement Prediction')

    # User input for variables and graph type
    x_variable = st.selectbox('Select a variable for the x-axis:', ['gender', 'ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p', 'workex', 'degree_t', 'specialisation'])
    y_variable = st.selectbox('Select a variable for the y-axis:', ['gender', 'ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p', 'workex', 'degree_t', 'specialisation'])
    graph_type = st.selectbox('Select a graph type:', ['Bar Plot', 'Pie Chart', 'Box Plot', 'Scatter Plot', 'Stacked Bar Plot'])

    if graph_type == 'Stacked Bar Plot':
        plt.figure(figsize=(10, 6))
        stacked_data = data.groupby([x_variable, y_variable]).size().unstack()
        stacked_data.plot(kind='bar', stacked=True)
        plt.xlabel(x_variable)
        plt.ylabel('Count')
        plt.title(f'{graph_type} of {y_variable} by {x_variable}')
        st.pyplot(plt)

    elif graph_type == 'Pie Chart':
        plt.figure(figsize=(8, 6))
        x_counts = data[x_variable].value_counts()
        labels = x_counts.index.tolist()
        colors = ['#FF7F0E', '#1F77B4']
        plt.pie(x_counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title(f'{graph_type} of {x_variable}')
        st.pyplot(plt)

    elif graph_type == 'Bar Plot':
        plt.figure(figsize=(10, 6))
        sns.barplot(data=data, x=x_variable, y=y_variable)
        plt.xlabel(x_variable)
        plt.ylabel(y_variable)
        plt.title(f'{graph_type} of {y_variable} by {x_variable}')
        st.pyplot(plt)

    elif graph_type == 'Scatter Plot':
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=data, x=x_variable, y=y_variable)
        plt.xlabel(x_variable)
        plt.ylabel(y_variable)
        plt.title(f'{graph_type} of {x_variable} vs {y_variable}')
        st.pyplot(plt)

    elif graph_type == 'Box Plot':
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=data, x=x_variable, y=y_variable)
        plt.xlabel(x_variable)
        plt.ylabel(y_variable)
        plt.title(f'{graph_type} of {y_variable} by {x_variable}')
        st.pyplot(plt)


    
