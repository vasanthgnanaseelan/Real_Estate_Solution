# Real Estate Price Prediction Web App

A machine learning project that predicts real estate property prices based on features like square footage, number of bedrooms/bathrooms, year built, and more. This app is built using Python, scikit-learn, and deployed using Streamlit Cloud.

## Project Overview

This project includes:

- Clean modular Python code using scikit-learn
- Data preprocessing, model training, evaluation, and prediction
- Visualizations of price distribution and feature correlations
- A fully interactive web app built with Streamlit
- Ready for deployment on Streamlit Cloud

## Project Structure

real_estate_ml_app/
│
├── app.py                 # Streamlit Web App
├── main.py                # Main pipeline (load, train, evaluate)
├── visuals.py             # EDA and plotting script
├── requirements.txt       # Required Python libraries
├── README.md              # This file
├── .gitignore             # Excluded files
│
├── data/
│   └── final.csv          # Dataset used for modeling
│
├── notebooks/
│   └── Real_Estate_Solution.ipynb
│
└── src/                   # Modular Python code
    ├── __init__.py
    ├── config.py
    ├── data_loader.py
    ├── eda_tools.py
    ├── preprocessing.py
    ├── data_splitter.py
    ├── model_trainer.py
    ├── model_eval.py
    ├── plot_tools.py
    └── utils.py

## How to Run Locally

1. Clone this repo

git clone https://github.com/vasanthgnanaseelan/real-estate-ml-streamlit.git cd real-estate-ml-streamlit

cpp
Copy
Edit

2. (Optional) Create a virtual environment

python -m venv venv venv\Scripts\activate # On Windows source venv/bin/activate # On macOS/Linux

markdown
Copy
Edit

3. Install dependencies

pip install -r requirements.txt

css
Copy
Edit

4. Run the main ML pipeline

python main.py

markdown
Copy
Edit

5. Run the Streamlit app

streamlit run app.py

markdown
Copy
Edit

## Live Demo (Streamlit Cloud)

https://vasanthgnanaseelan-real-estate-solution-app-bhrviv.streamlit.app/


## Features

- Trains a Random Forest Regression model
- Predicts price based on custom user inputs
- Evaluation metrics (MAE, MSE)
- Histogram, KDE plot, and correlation heatmap
- Clean and modular code with logging and error handling

## Author

Vasanth Gnana Seelan  
GitHub: https://github.com/vasanthgnanaseelan

## License

This project is open-source and free to use under the MIT License.