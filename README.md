👗 StyleSync - AI Fashion Assistant
An intelligent fashion assistant that analyzes your clothing images and provides personalized outfit recommendations using Google's Gemini AI.
Features

📷 Image Analysis: Upload clothing photos for AI-powered analysis
🤖 Smart Categorization: Automatically categorizes and describes clothing items
👔 Outfit Recommendations: Get personalized outfit suggestions based on occasion, season, and preferences
📊 Wardrobe Management: Track and organize your clothing collection
🎨 Style Insights: Receive styling tips and fashion advice

Live Demo
🌐 Try StyleSync Live
Getting Started
Prerequisites

Python 3.8+
Google Gemini API Key (Get one here)

Local Installation

Clone the repository:

bashgit clone https://github.com/yourusername/StyleSyncc.git
cd StyleSyncc

Install dependencies:

bashpip install -r requirements.txt

Add your API key:

Create .streamlit/secrets.toml
Add: GEMINI_API_KEY = "your-api-key-here"


Run the app:

bashstreamlit run app.py
Deployment
Streamlit Community Cloud

Push your code to GitHub
Go to share.streamlit.io
Connect your repository
Add your GEMINI_API_KEY in the secrets section
Deploy!

Usage

Add Clothing: Upload clear images of your clothing items
Get Analysis: AI analyzes fabric, color, style, and occasion suitability
Build Wardrobe: Items are automatically added to your digital wardrobe
Get Recommendations: Specify occasion and preferences for outfit suggestions
Export Data: Download your wardrobe data as JSON

Technology Stack

Frontend: Streamlit
AI/ML: Google Gemini AI, LangChain
Data: Pydantic, Pandas
Image Processing: Pillow

Contributing

Fork the repository
Create a feature branch
Make your changes
Submit a pull request

License
This project is licensed under the MIT License.
Support
If you encounter any issues or have questions, please open an issue on GitHub.

Made with ❤️ and AI
