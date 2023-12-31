# Spam Detection Project

Welcome to the Spam Detection Project repository! This project aims to provide an accurate and user-friendly solution for detecting spam emails using Python and its libraries, including Scikit-learn, NLTK, and Streamlit. The project is also deployed on Heroku for easy access.

## Project Overview

Spam emails have become a persistent nuisance in modern communication. This project utilizes machine learning techniques to create a spam detection model, which is then presented through a web application for user interaction. The project stack includes:

- Python
- Scikit-learn
- NLTK
- Streamlit
- Heroku

## Features

- **Machine Learning Model:** The core of the project employs Scikit-learn for building a robust machine learning model that can distinguish between legitimate and spam emails.

- **Natural Language Processing:** NLTK is used to process and analyze the text content of emails, enabling the model to understand linguistic patterns and cues.

- **Web Application:** The project includes a Streamlit web application that allows users to upload emails and receive real-time predictions on whether they are spam or not.

- **Deployment:** The web application is deployed on Heroku, making it accessible to users worldwide without the need for local installation.

## Getting Started

To set up and run the project locally, follow these steps:

1. Clone the repository: `git clone https://github.com/Mansi0402/Spam-Detection.git`
2. Navigate to the project directory: `cd Spam-Detection`
3. Install the required dependencies: `pip install -r requirements.txt`
4. Run the Streamlit app: `streamlit run app.py`


## Directory Structure

- `app.py`: Contains the Streamlit web application code.
- `model.py`: Implements the spam detection model using Scikit-learn and NLTK.
- `vectorizer.txt`: Implements the TfidfVectorizer vectorizer for the model..
- `requirements.txt`: Lists all the required Python packages.
- `README.md`: Project documentation.

## Contribution

Contributions to the project are welcome! If you'd like to enhance the model, add features to the web app, or improve the documentation, feel free to submit a pull request.

