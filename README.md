# Annual Report Analyzer

A powerful RAG (Retrieval Augmented Generation) system that analyzes company annual reports and generates concise one-page summaries using LLMs and vector databases.

## Features
- 📄 PDF processing and text extraction
- 🔍 Vector-based document retrieval 
- 🤖 AI-powered report analysis
- 📊 Automated summary generation
- 🌏 Multi-language support (English & Chinese)
- 📱 Memory-optimized for large documents

## Prerequisites
- Python 3.8+
- OpenAI API key

## Installation
1. Clone the repository
```
git clone [your-repo-url]
cd annual-report-analyzer
```
2. Create and activate virtual environment
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
3. Env Config
```
echo "OPENAI_API_KEY=your_openai_api_key" > .env
```

## Usage
1. Place your annual report PDF
```
mv your_annual_report.pdf annual_report.pdf
```

2. Run the analyzer
```
python src/annual_report_analyzer.py
```

## Configuration

Language Settings
You can modify the language setting in the main function:
```
# For English
analyzer = AnnualReportAnalyzer(language='en')

# For Chinese
analyzer = AnnualReportAnalyzer(language='zh')
```