import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from fpdf import FPDF
import logging
from typing import List, Optional
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AnnualReportAnalyzer:
    def __init__(self, language: str = 'en', model_name: str = "gpt-4"):
        """
        Initialize the analyzer with language and model settings
        
        Args:
            language (str): Language code ('en' or 'zh')
            model_name (str): Name of the OpenAI model to use
        """
        try:
            self.embeddings = OpenAIEmbeddings()
            self.llm = ChatOpenAI(model_name=model_name, temperature=0)
            self.language = language
            
            # Font settings for different languages
            self.font_settings = {
                'en': {'font': 'Arial', 'font_path': None},
                'zh': {
                    'font': 'TaipeiSansTCBeta-Regular',
                    'font_path': 'fonts/TaipeiSansTCBeta-Regular.ttf'
                }
            }
        except Exception as e:
            logger.error(f"Failed to initialize AnnualReportAnalyzer: {str(e)}")
            raise

    def load_pdf(self, pdf_path: str) -> List:
        """
        Load and split PDF document with error handling
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            List: List of document chunks
        """
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")

            logger.info(f"Loading PDF from {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            pages = loader.load_and_split()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            chunks = text_splitter.split_documents(pages)
            logger.info(f"Successfully split PDF into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error loading PDF: {str(e)}")
            raise

    def process_large_pdf(self, pdf_path: str, batch_size: int = 100) -> Optional[Chroma]:
        """
        Process large PDF files in batches to optimize memory usage
        
        Args:
            pdf_path (str): Path to the PDF file
            batch_size (int): Size of each batch
            
        Returns:
            Chroma: Vector store object
        """
        try:
            chunks = self.load_pdf(pdf_path)
            vectorstore = None
            
            for i in tqdm(range(0, len(chunks), batch_size), desc="Processing PDF batches"):
                batch = chunks[i:i + batch_size]
                
                if vectorstore is None:
                    vectorstore = Chroma.from_documents(
                        documents=batch,
                        embedding=self.embeddings
                    )
                else:
                    vectorstore.add_documents(batch)
                
                logger.info(f"Processed batch {i//batch_size + 1}/{len(chunks)//batch_size + 1}")
            
            return vectorstore
            
        except Exception as e:
            logger.error(f"Error processing large PDF: {str(e)}")
            raise

    def create_analysis_prompt(self) -> PromptTemplate:
        """Create analysis prompt based on language"""
        templates = {
            'en': """
            You are a professional financial analyst. Please generate a concise one-page summary based on the provided annual report content.

            Please analyze in the following format:

            1. Financial Performance
            - Revenue Analysis
            - Profit Analysis
            - Cash Flow Analysis

            2. Business Highlights
            - Key Achievements
            - Market Performance
            - Product Development

            3. Risk Factors
            - Major Challenges
            - Market Risks
            - Operational Risks

            4. Future Outlook
            - Development Strategy
            - Expected Goals

            Requirements:
            - Summary must be concise, not exceeding 500 words
            - Important data should show year-over-year changes
            - Use clear bullet points
            - Ensure accuracy of content

            Question: {question}
            Context: {context}

            Please generate the summary:
            """,
            'zh': """
            您是一位專業的財務分析師。請根據提供的年報內容，生成一份簡潔的一頁總結。

            請按以下格式進行分析：

            1. 財務表現
            - 收入分析
            - 利潤分析
            - 現金流分析

            2. 業務亮點
            - 主要成就
            - 市場表現
            - 產品發展

            3. 風險因素
            - 主要挑戰
            - 市場風險
            - 營運風險

            4. 未來展望
            - 發展策略
            - 預期目標

            要求：
            - 總結必須精簡，不超過500字
            - 重要數據需要標明同比變化
            - 使用清晰的要點形式
            - 確保內容準確性

            問題：{question}
            相關內容：{context}

            請生成總結：
            """
        }
        
        return PromptTemplate(
            template=templates.get(self.language, templates['en']),
            input_variables=["question", "context"]
        )

    def create_pdf_summary(self, summary: str, output_path: str):
        """
        Generate PDF summary with proper font support
        
        Args:
            summary (str): Summary text to include in PDF
            output_path (str): Path to save the PDF
        """
        try:
            pdf = FPDF()
            pdf.add_page()
            
            # Get font settings for current language
            font_settings = self.font_settings.get(self.language)
            
            # Add custom font if needed
            if font_settings['font_path']:
                pdf.add_font(
                    font_settings['font'], 
                    '', 
                    font_settings['font_path'], 
                    uni=True
                )
            
            # Set font
            pdf.set_font(font_settings['font'], size=12)
            
            # Add title
            pdf.set_font(font_settings['font'], 'B', 16)
            title = "Annual Report Performance Summary" if self.language == 'en' else "年報表現總結"
            pdf.cell(200, 10, title, ln=True, align='C')
            
            # Add content
            pdf.set_font(font_settings['font'], size=12)
            pdf.multi_cell(0, 10, summary)
            
            # Save PDF
            pdf.output(output_path)
            logger.info(f"PDF summary created successfully at {output_path}")
            
        except Exception as e:
            logger.error(f"Error creating PDF summary: {str(e)}")
            raise

def main():
    try:
        # Set file paths
        pdf_path = "annual_report.pdf"
        output_path = "summary.pdf"
        
        # Create analyzer with language preference
        analyzer = AnnualReportAnalyzer(language='en')  # or 'zh' for Chinese
        
        # Process file in batches
        logger.info("Starting PDF processing...")
        vectorstore = analyzer.process_large_pdf(pdf_path, batch_size=100)
        
        logger.info("Generating summary...")
        prompt = analyzer.create_analysis_prompt()
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=analyzer.llm,
            retriever=vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": prompt}
        )
        
        question = "Please generate a comprehensive company performance summary based on the annual report content."
        response = qa_chain.invoke({"query": question})
        
        logger.info("Creating PDF summary...")
        analyzer.create_pdf_summary(response["result"], output_path)
        
        logger.info(f"Process completed successfully. Summary saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"An error occurred in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()