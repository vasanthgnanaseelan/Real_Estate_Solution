import streamlit as st
import pandas as pd
from src.data_loader import fetch_dataset
from src.preprocessing import clean_and_encode
from src.data_splitter import partition_data
from src.model_trainer import train_random_forest
from src.model_eval import assess_model

st.title("Real Estate Price Prediction")

# Load dataset (adjust for deployment)
file_path = "data/final.csv"  # Will work when deployed
df = fetch_dataset(file_path)

if df.empty:
    st.error("Dataset not found!")
else:
    st.write("### Dataset Preview", df.head())
    
    # Process the data
    df = clean_and_encode(df)
    X_train, X_test, y_train, y_test = partition_data(df, target="price")

    # Train the model
    model = train_random_forest(X_train, y_train)
    mae, mse = assess_model(model, X_test, y_test)
    
    st.write("### Model Evaluation")
    st.write(f"**MAE**: {mae}")
    st.write(f"**MSE**: {mse}")
    
    # Prediction Form
    st.write("### Enter Property Details for Price Prediction")
    with st.form("prediction_form"):
        year_sold = st.number_input("Year Sold", min_value=2000, max_value=2025)
        property_tax = st.number_input("Property Tax")
        insurance = st.number_input("Insurance")
        beds = st.number_input("Beds", min_value=1)
        baths = st.number_input("Baths", min_value=1)
        sqft = st.number_input("Sqft", min_value=100)
        year_built = st.number_input("Year Built", min_value=1900)
        lot_size = st.number_input("Lot Size")
        
        # Submit button
        submitted = st.form_submit_button("Predict Price")
        
        if submitted:
            # Format the input for prediction
            input_data = pd.DataFrame({
                "year_sold": [year_sold],
                "property_tax": [property_tax],
                "insurance": [insurance],
                "beds": [beds],
                "baths": [baths],
                "sqft": [sqft],
                "year_built": [year_built],
                "lot_size": [lot_size],
            })
            
            # Predict the price
            input_data = input_data.reindex(columns=X_train.columns, fill_value=0)
            predicted_price = model.predict(input_data)[0]
            st.write(f"### Predicted Price: ${predicted_price:,.2f}")
