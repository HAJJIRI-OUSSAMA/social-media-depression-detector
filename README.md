# Social Media Depression Detector

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Framework](https://img.shields.io/badge/Framework-Streamlit-red.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-sklearn-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

An AI-powered mental health analysis tool that leverages machine learning to identify potential signs of depression from social media content. This project combines natural language processing, sentiment analysis, and multiple ML models to provide insights into mental health indicators from text data.

## 🎯 Features

- Real-time social media post analysis
- Multiple ML model support (Logistic Regression, Random Forest, KNN)
- Integrated VADER sentiment analysis
- Twitter/X post compatibility
- User-friendly web interface
- Detailed sentiment visualization

## 🛠️ Technologies Used

- **Python 3.8+**
- **Machine Learning**: scikit-learn
- **NLP**: NLTK, VADER
- **Web Interface**: Streamlit
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn

## 📦 Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/social-media-depression-detector.git
cd social-media-depression-detector
```

2. Create a virtual environment:

```bash
python -m venv venv
.\venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Download NLTK resources:

```python
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt'); nltk.download('stopwords')"
```

## 🚀 Usage

1. Start the Streamlit app:

```bash
streamlit run app.py
```

2. Access the web interface at `http://localhost:8501`

3. Choose analysis method:
   - Enter social media post text directly
   - Provide Twitter/X post URL
   - Input hashtags (optional)

## 📊 Model Performance

Our models have been trained on a curated dataset of social media posts with the following accuracy scores:

- Logistic Regression: ~85%
- Random Forest: ~87%
- KNN: ~83%

## 🗂️ Project Structure

```
social-media-depression-detector/
├── app.py                    # Streamlit web application
├── data_preprocessing.py     # Data cleaning and feature extraction
├── train_models.py          # Model training pipeline
├── predict.py               # Prediction functionality
├── data/                    # Dataset directory
├── models/                  # Saved model files
└── requirements.txt         # Project dependencies
```

## ⚠️ Disclaimer

This tool is designed for research and awareness purposes only. It should not be used as a substitute for professional medical diagnosis or treatment. Always consult qualified mental health professionals for medical advice.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NLTK team for VADER sentiment analysis
- Scikit-learn community
- Streamlit developers
- Contributors and testers

## 📧 Contact

Your Name - [@yourtwitter](https://twitter.com/yourtwitter) - email@example.com

Project Link: [https://github.com/yourusername/social-media-depression-detector](https://github.com/yourusername/social-media-depression-detector)
