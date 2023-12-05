# Importing necessary libraries for creating a Streamlit app
import streamlit as st
import pandas as pd
import pickle
from streamlit.components.v1 import html
import sys
sys.path.insert(1, "C:/past/your/coppied/path/here/streamlit_option_menu")
from streamlit_option_menu import option_menu
import numpy as np
import matplotlib.image as plt
import os
import plotly.figure_factory as ff
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

# Load your trained CatBoost model
model_cb = pickle.load(open('housingmarket.sav', 'rb'))

st.set_page_config(page_title="REMDA",
                   page_icon="🔍",
                   layout="wide")

st.markdown("<h1 style='text-align: center;'>REMDA</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Real Estate Market - Dashboard Analysis</h3>", unsafe_allow_html=True)

# Create a horizontal menu with the 'option_menu' custom component
selected2 = option_menu(None, ["Real Estate Analysis", "Prediction", "Model Evaluation"], 
    icons=['home-fill', 'stars', 'gear-fill'], 
    menu_icon="cast", default_index=0, orientation="horizontal")

if selected2 == "Real Estate Analysis":
    # The HTML content to be embedded
    tableau_html = """
<!DOCTYPE html>
<html>
<head>
    <script type="module" src="https://public.tableau.com/javascripts/api/tableau.embedding.3.latest.min.js"></script>
    <style>
        /* CSS untuk mengatur elemen di tengah */
        .center {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
        }

        /* CSS untuk membuat elemen responsif */
        #tableauViz {
            max-width: 100%; /* Maksimum lebar elemen */
            max-height: 100%; /* Maksimum tinggi elemen */
        }
    </style>
</head>
<body>
    <!-- Membungkus elemen dengan div yang memiliki kelas "center" -->
    <div class="center">
        <tableau-viz id="tableauViz"
        src='https://public.tableau.com/app/profile/shaltsa.nadya/viz/RealestatemarketinAustraliadashboard/Dashboard1'
        device="desktop"
        toolbar="hidden" hide-tabs>
        </tableau-viz>
    </div>
</body>
</html>

    """
    
    with st.columns([1, 100, 1])[1]:
        html(tableau_html, height=670)

elif selected2 == "Prediction":
    # Tenure Months Prediction Page
    st.header("Real Estate Market Price Predictions")
    
    df_sample = pd.read_csv("real_estate.csv")

    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
    # Read the uploaded file into a DataFrame
        try:
            if uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                st.error('This file format is not supported. Please upload a .xlsx or .csv file.')
                st.stop()

            df_price = df['Price']

            # Check for 'Latitude' and 'Longitude' columns and drop if exists
            if 'Lattitude' in df.columns and 'Longtitude' in df.columns and 'Date' in df.columns and 'Postcode' in df.columns and 'Address' in df.columns and 'Suburb' in df.columns:
                df.drop(['Lattitude','Longtitude', 'Date', 'Postcode', 'Address', 'Suburb', 'SellerG'], axis=1, inplace=True)

            # Process the DataFrame as needed for your models
            Cat= df[['Type','Method', 'CouncilArea', 'Regionname']]
            Cat = Cat.apply(LabelEncoder().fit_transform)
            
            df= df.drop(['Type', 'Method', 'CouncilArea', 'Regionname'], axis= 1)
            df= pd.concat([df, Cat], axis= 1)
            
            # Urutkan data sesuai permintaan
            df = df[['Rooms', 'Distance', 'Bedroom2', 'Bathroom', 'Car', 'Landsize',
                     'BuildingArea', 'YearBuilt', 'Propertycount', 'Type', 'Method',
                     'CouncilArea', 'Regionname']]

            # Predict using the CatBoost model
            predictions = model_cb.predict(df)
            
            # Add predictions to the DataFrame
            df['Predicted Price'] = predictions

            df['Predicted Price'] = df['Predicted Price'].apply(lambda x: round(x))
            
            df = pd.concat([df, df_price], axis=1)

            df_disp = df[['Predicted Price', 'Price', 'Rooms', 'Distance', 'Bedroom2', 'Bathroom', 'Car', 'Landsize',
                     'BuildingArea', 'YearBuilt', 'Propertycount', 'Type', 'Method',
                    'CouncilArea', 'Regionname']]

            st.header('Table of Predictions')
            # Display the DataFrame with predictions
            st.dataframe(df_disp)

            # histogram tenure months dan predicted tenure months to churn dengan px histogram
            fig11 = px.histogram(df, x=['Price', 'Predicted Price'],
                                title='House Price vs Predicted House Price')
            
            fig11.update_layout(legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            ))

            st.plotly_chart(fig11, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred: {e}")
    
    if uploaded_file is None:
        st.download_button("Download Sample Data", data=df_sample.to_csv(index=False), file_name='sample_data.csv')

        # Define the form and its fields
        with st.form(key='prediction_form'):
            # Create columns for the input fields
            col1, col2, col3 = st.columns(3)
            
            with col1:
                #type = st.selectbox('Type', ['h', 't', 'u'])
                method = st.selectbox('Method', ['S', 'SP', 'PI', 'VB', 'SA'])
                
    
            with col2:
                region = st.selectbox('Regionname', ['Northern Metropolitan', 'Western Metropolitan', 'Southern Metropolitan', 'Eastern Metropolitan', 'South-Eastern Metropolitan', 'Eastern Victoria', 'Northern Victoria', 'Western Victoria'])
                type = st.radio('Type', ['h', 't', 'u'])
                
            with col3:
                council = st.selectbox('CouncilArea', ['Brimbank', 'Melton', 'Hobsons Bay', 'Banyule', 'Greater Dandenong', 'Moreland', 'Unavailable', 'Frankston', 'Nillumbik', 'Glen Eira', 'Moonee Valley', 'Wyndham', 'Darebin', 'Kingston', 'Whittlesea', 'Hume', 'Manningham', 'Bayside', 'Whitehorse', 'Stonnington', 'Cardinia', 'Boroondara', 'Yarra', 'Macedon Ranges', 'Melbourne', 'Casey', 'Port Phillip', 'Knox', 'Yarra Ranges', 'Maroondah', 'Monash', 'Maribyrnong', 'Moorabool'])
                
            # Create a single column for the numeric inputs below the radio buttons
            col4, col5, col6 = st.columns(3)  # The third column is just to take up the remaining space
            with col4:
                room = st.number_input('Room', min_value=1, value=10)
                distance = st.number_input('Monthly Purchase', min_value=0, value=100)
                bedroom = st.number_input('Bedroom2', min_value=0, value=20)
            
            with col5:
                bathroom = st.number_input('Bathroom', min_value=0, value=10)
                car = st.number_input('Car', min_value=0, value=10)
                landsize = st.number_input('Landsize', min_value=0, value=500000)
            
            with col6:
                building = st.number_input('BuildingArea', min_value=0, value=6000)
                yearbuilt = st.number_input('YearBuilt', min_value=1800, value=2020)
                property = st.number_input('Propertycount', min_value=0, value=30000)

            # Place the submit button in the center below the columns
            submit_button = st.form_submit_button(label='Predict')

        if submit_button:
            # Create a dataframe from the inputs
            input_data = pd.DataFrame({
                'Method' : [method],
                'Regionname' : [region],
                'Type' : [type],
                'CouncilArea' : [council],
                'Rooms': [room],
                'Distance': [distance],
                'Bedroom2': [bedroom],
                'Bathroom': [bathroom],
                'Car': [car],
                'Landsize': [landsize],
                'BuildingArea': [building],
                'YearBuilt': [yearbuilt],
                'Propertycount': [property]
            })  
            Cat= input_data[['Type','Method', 'CouncilArea', 'Regionname']]
            Cat = Cat.apply(LabelEncoder().fit_transform)
            
            input_data= input_data.drop(['Type', 'Method', 'CouncilArea', 'Regionname'], axis= 1)
            input_data= pd.concat([input_data, Cat], axis= 1)

            # Predict using the CatBoost model
            prediction = model_cb.predict(input_data)

            # Display the prediction
            st.metric(label='Predicted Price', value=f"{int(prediction)} AUD")    
elif selected2 == "Model Evaluation":
    
    # load .csv as df
    df = pd.read_csv('df_final.csv')
    
    df.sort_values(by=['Price'], inplace=True, ascending=True)
    
    # rename tenure month names variables to tenure month customer
    df.rename(columns={'Price': 'House Price'}, inplace=True)

    # create index after sort
    df['Index'] = np.arange(len(df))

    # show mae and r2 score and show using st.metric
    mae = round(np.mean(abs(df['Predicted Price'] - df['House Price'])), 2)
    r2 = round(1 - (np.sum((df['House Price'] - df['Predicted Price'])**2) / np.sum((df['House Price'] - np.mean(df['House Price']))**2)), 2)
    mse = round(np.mean((df['House Price'] - df['Predicted Price'])**2), 2)

    st.header('Model Evaluation')

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label='MAE', value=mae)
    with col2:
        st.metric(label='MSE', value=mse)
    with col3:
        st.metric(label='R2 Score', value=r2)
        
    # buat line chart 'Predicted Tenure Months to Churn', 'Tenure Months' dengan plotly
    fig1 = px.scatter(df,x='Index', y=['Predicted Price', 'House Price'],
                        title='Predicted House Price vs House Price')

    fig1.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ))

    # load feature importance
    feature_importance = pd.read_csv('importance.csv')
    # buat bar chart feature importance dengan px horizontal bar dan sort value
    feat= feature_importance.head(3)
    fig2 = px.bar(feature_importance, x='Importances', y='Feature Id',
                     orientation='h', title='Feature Importance')
    fig2.update_layout(yaxis=dict(autorange="reversed"),  yaxis_title='', yaxis_showticklabels= True)
    fig2.update_traces(marker_color= ['#637E76' if i in feat['Feature Id'].tolist() else '#6dbf9c' for i in feature_importance['Feature Id'].tolist()]
                   , showlegend=False)
    
    col1, col2 = st.columns([5, 3])
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.plotly_chart(fig2, use_container_width=True)