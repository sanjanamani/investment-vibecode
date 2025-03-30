import os
import requests
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from sec_api import QueryApi
from datetime import datetime
from fpdf import FPDF
import yfinance as yf
import plotly.graph_objects as go
from openai import OpenAI
import json
from typing import Dict, List
import schedule
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Investment Research Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize API clients with environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
SEC_API_KEY = os.getenv('SEC_API_KEY')

if not OPENAI_API_KEY or not SEC_API_KEY:
    st.error("Please set the OPENAI_API_KEY and SEC_API_KEY environment variables.")
    st.stop()

openai_client = OpenAI(api_key=OPENAI_API_KEY)
sec_client = QueryApi(api_key=SEC_API_KEY)

# Create reports directory if it doesn't exist
os.makedirs("reports", exist_ok=True)

class DueDiligenceReport:
    def __init__(self, company_name: str):
        self.company_name = company_name
        self.data = {}
        
    def collect_sec_data(self) -> Dict:
        """Collect SEC filing data"""
        query = {
            "query": {"query_string": {
                "query": f"company_name:{self.company_name}"
            }},
            "from": "0",
            "size": "10"
        }
        filings = sec_client.get_filings(query)
        return filings

    def scrape_company_info(self) -> Dict:
        """Scrape company website and social media"""
        try:
            ticker = yf.Ticker(self.company_name)
            info = ticker.info
            return {
                'company_name': info.get('longName', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'website': info.get('website', ''),
                'description': info.get('longBusinessSummary', '')
            }
        except Exception as e:
            print(f"Error scraping company info: {str(e)}")
            return {}

    def analyze_financials(self) -> Dict:
        """Analyze financial metrics and ratios"""
        try:
            ticker = yf.Ticker(self.company_name)
            financials = ticker.financials
            metrics = {
                'revenue_growth': self._calculate_growth_rate(financials.loc['Total Revenue']),
                'profit_margins': self._calculate_margins(financials),
                'valuation_metrics': self._get_valuation_metrics(ticker)
            }
            return metrics
        except Exception as e:
            print(f"Error analyzing financials: {str(e)}")
            return {}

    def _calculate_growth_rate(self, revenue_series) -> float:
        """Calculate year-over-year revenue growth rate"""
        try:
            if len(revenue_series) >= 2:
                current = revenue_series.iloc[0]
                previous = revenue_series.iloc[1]
                return ((current - previous) / previous) * 100
            return 0.0
        except Exception as e:
            print(f"Error calculating growth rate: {str(e)}")
            return 0.0

    def _calculate_margins(self, financials) -> Dict:
        """Calculate various profit margins"""
        try:
            margins = {}
            if 'Net Income' in financials.index and 'Total Revenue' in financials.index:
                net_income = financials.loc['Net Income'].iloc[0]
                revenue = financials.loc['Total Revenue'].iloc[0]
                if revenue != 0:
                    margins['net_margin'] = (net_income / revenue) * 100
            return margins
        except Exception as e:
            print(f"Error calculating margins: {str(e)}")
            return {}

    def _get_valuation_metrics(self, ticker) -> Dict:
        """Get key valuation metrics"""
        try:
            info = ticker.info
            return {
                'pe_ratio': info.get('forwardPE', 'N/A'),
                'price_to_book': info.get('priceToBook', 'N/A'),
                'price_to_sales': info.get('priceToSalesTrailing12Months', 'N/A'),
                'dividend_yield': info.get('dividendYield', 'N/A')
            }
        except Exception as e:
            print(f"Error getting valuation metrics: {str(e)}")
            return {}

    def generate_report(self) -> str:
        """Generate PDF report"""
        try:
            pdf = FPDF()
            pdf.add_page()
            
            # Add header
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 10, f'Due Diligence Report: {self.company_name}', ln=True)
            
            # Collect data
            company_info = self.scrape_company_info()
            financial_metrics = self.analyze_financials()
            sec_data = self.collect_sec_data()
            
            # Add sections with content
            sections = {
                'Company Overview': self._format_company_overview(company_info),
                'Financial Analysis': self._format_financial_analysis(financial_metrics),
                'Market Position': self._format_market_position(sec_data),
                'Risk Assessment': self._format_risk_assessment(financial_metrics, sec_data)
            }
            
            for section, content in sections.items():
                self._add_section(pdf, section, content)
            
            # Create reports directory if it doesn't exist
            os.makedirs("reports", exist_ok=True)
            
            # Save report
            output_path = f"reports/{self.company_name}_{datetime.now().strftime('%Y%m%d')}.pdf"
            pdf.output(output_path)
            return output_path
        except Exception as e:
            print(f"Error generating report: {str(e)}")
            return ""

    def _format_company_overview(self, info: Dict) -> str:
        """Format company overview section"""
        return f"""
        Company Name: {info.get('company_name', 'N/A')}
        Sector: {info.get('sector', 'N/A')}
        Industry: {info.get('industry', 'N/A')}
        Website: {info.get('website', 'N/A')}
        
        Description:
        {info.get('description', 'N/A')}
        """

    def _format_financial_analysis(self, metrics: Dict) -> str:
        """Format financial analysis section"""
        return f"""
        Revenue Growth Rate: {metrics.get('revenue_growth', 'N/A'):.2f}%
        Net Profit Margin: {metrics.get('profit_margins', {}).get('net_margin', 'N/A'):.2f}%
        
        Valuation Metrics:
        P/E Ratio: {metrics.get('valuation_metrics', {}).get('pe_ratio', 'N/A')}
        Price to Book: {metrics.get('valuation_metrics', {}).get('price_to_book', 'N/A')}
        Price to Sales: {metrics.get('valuation_metrics', {}).get('price_to_sales', 'N/A')}
        Dividend Yield: {metrics.get('valuation_metrics', {}).get('dividend_yield', 'N/A')}
        """

    def _format_market_position(self, sec_data: Dict) -> str:
        """Format market position section"""
        return f"""
        Recent SEC Filings:
        {self._format_sec_filings(sec_data)}
        """

    def _format_risk_assessment(self, metrics: Dict, sec_data: Dict) -> str:
        """Format risk assessment section"""
        return f"""
        Financial Risks:
        - Revenue Growth: {metrics.get('revenue_growth', 'N/A'):.2f}%
        - Profit Margins: {metrics.get('profit_margins', {}).get('net_margin', 'N/A'):.2f}%
        
        Market Risks:
        - Beta: {metrics.get('valuation_metrics', {}).get('beta', 'N/A')}
        - Market Cap: {metrics.get('valuation_metrics', {}).get('market_cap', 'N/A')}
        """

    def _format_sec_filings(self, sec_data: Dict) -> str:
        """Format SEC filings data"""
        try:
            if 'filings' in sec_data:
                return "\n".join([
                    f"- {filing.get('form', 'N/A')} filed on {filing.get('filedAt', 'N/A')}"
                    for filing in sec_data['filings'][:5]
                ])
            return "No recent SEC filings found."
        except Exception as e:
            print(f"Error formatting SEC filings: {str(e)}")
            return "Error retrieving SEC filings."

    def _add_section(self, pdf: FPDF, section: str, content: str):
        """Helper to add sections to PDF"""
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, section, ln=True)
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 10, content)
        pdf.ln(10)

class InvestmentResearchAPI:
    def __init__(self):
        self.setup_sidebar()
        self.setup_main_content()
        
    def setup_sidebar(self):
        with st.sidebar:
            st.title("📊 Research Settings")
            st.markdown("---")
            
            # Company Search
            self.company_name = st.text_input('Enter Company Name:', placeholder='e.g., AAPL')
            
            # Report Type Selection
            self.report_type = st.selectbox(
                "Select Report Type",
                ["Full Analysis", "Financial Overview", "Market Analysis", "Risk Assessment"]
            )
            
            # Date Range
            self.date_range = st.date_input(
                "Select Date Range",
                value=(datetime.now().date() - pd.Timedelta(days=365), datetime.now().date())
            )
            
            # Additional Options
            st.markdown("### Additional Options")
            self.include_news = st.checkbox("Include News Analysis", value=True)
            self.include_social = st.checkbox("Include Social Media Sentiment", value=True)
            self.include_competitors = st.checkbox("Include Competitor Analysis", value=True)
            
            st.markdown("---")
            st.markdown("### About")
            st.markdown("This tool provides comprehensive investment research and analysis.")
    
    def setup_main_content(self):
        # Header
        st.title("📈 Investment Research Dashboard")
        st.markdown("---")
        
        if self.company_name:
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["Overview", "Financial Analysis", "Market Analysis"])
            
            with tab1:
                self.show_overview()
            
            with tab2:
                self.show_financial_analysis()
            
            with tab3:
                self.show_market_analysis()
            
            # Generate Report Button
            if st.button('Generate Full Report', type='primary'):
                with st.spinner('Generating comprehensive report...'):
                    dd = DueDiligenceReport(self.company_name)
                    report_path = dd.generate_report()
                    st.success(f'Report generated successfully! 📄\nLocation: {report_path}')
    
    def show_overview(self):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Company Overview")
            # Add company info visualization
            try:
                ticker = yf.Ticker(self.company_name)
                info = ticker.info
                
                st.metric("Current Price", f"${info.get('currentPrice', 'N/A')}")
                st.metric("Market Cap", f"${info.get('marketCap', 'N/A'):,.0f}")
                st.metric("52 Week High", f"${info.get('fiftyTwoWeekHigh', 'N/A')}")
                st.metric("52 Week Low", f"${info.get('fiftyTwoWeekLow', 'N/A')}")
            except Exception as e:
                st.error(f"Error fetching company data: {str(e)}")
        
        with col2:
            st.subheader("Key Statistics")
            # Add key statistics visualization
            try:
                ticker = yf.Ticker(self.company_name)
                hist = ticker.history(period="1y")
                
                fig = go.Figure(data=[go.Candlestick(x=hist.index,
                                                    open=hist['Open'],
                                                    high=hist['High'],
                                                    low=hist['Low'],
                                                    close=hist['Close'])])
                fig.update_layout(title="Stock Price History",
                                yaxis_title="Price",
                                height=400)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating price chart: {str(e)}")
    
    def show_financial_analysis(self):
        st.subheader("Financial Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            try:
                ticker = yf.Ticker(self.company_name)
                financials = ticker.financials
                
                # Revenue Growth Chart
                fig = go.Figure(data=[go.Bar(x=financials.columns,
                                           y=financials.loc['Total Revenue'])])
                fig.update_layout(title="Revenue Over Time",
                                yaxis_title="Revenue",
                                height=300)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating financial charts: {str(e)}")
        
        with col2:
            try:
                ticker = yf.Ticker(self.company_name)
                balance_sheet = ticker.balance_sheet
                
                # Balance Sheet Chart
                fig = go.Figure(data=[go.Bar(x=balance_sheet.columns,
                                           y=balance_sheet.loc['Total Assets'])])
                fig.update_layout(title="Total Assets Over Time",
                                yaxis_title="Assets",
                                height=300)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating balance sheet chart: {str(e)}")
    
    def show_market_analysis(self):
        st.subheader("Market Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            try:
                ticker = yf.Ticker(self.company_name)
                info = ticker.info
                
                # Market Metrics
                st.metric("Beta", f"{info.get('beta', 'N/A')}")
                st.metric("Forward P/E", f"{info.get('forwardPE', 'N/A')}")
                st.metric("Dividend Yield", f"{info.get('dividendYield', 'N/A')*100:.2f}%")
                st.metric("Volume", f"{info.get('volume', 'N/A'):,.0f}")
            except Exception as e:
                st.error(f"Error fetching market metrics: {str(e)}")
        
        with col2:
            try:
                ticker = yf.Ticker(self.company_name)
                recommendations = ticker.recommendations
                
                if recommendations is not None:
                    fig = go.Figure(data=[go.Pie(labels=recommendations['To Grade'].value_counts().index,
                                               values=recommendations['To Grade'].value_counts().values)])
                    fig.update_layout(title="Analyst Recommendations",
                                    height=300)
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating recommendations chart: {str(e)}")

def schedule_updates():
    """Schedule weekly data updates"""
    def update_reports():
        try:
            # Get list of companies from reports directory
            reports_dir = "reports"
            if not os.path.exists(reports_dir):
                return
            
            # Get unique company names from report filenames
            company_files = [f for f in os.listdir(reports_dir) if f.endswith('.pdf')]
            companies = set(f.split('_')[0] for f in company_files)
            
            # Update reports for each company
            for company in companies:
                try:
                    dd = DueDiligenceReport(company)
                    dd.generate_report()
                    print(f"Updated report for {company}")
                except Exception as e:
                    print(f"Error updating report for {company}: {str(e)}")
        except Exception as e:
            print(f"Error in update_reports: {str(e)}")
    
    # Schedule updates every Monday at midnight
    schedule.every().monday.at("00:00").do(update_reports)
    
    while True:
        schedule.run_pending()
        time.sleep(3600)  # Check every hour

# Add rate limiting decorator
def rate_limit(max_requests=100, time_window=3600):
    """Rate limiting decorator"""
    requests = []
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            current_time = time.time()
            
            # Remove old requests
            requests[:] = [req_time for req_time in requests 
                         if current_time - req_time < time_window]
            
            # Check if we've exceeded the rate limit
            if len(requests) >= max_requests:
                raise Exception("Rate limit exceeded. Please try again later.")
            
            # Add current request
            requests.append(current_time)
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Apply rate limiting to API calls
@rate_limit(max_requests=100, time_window=3600)
def get_stock_data(ticker_symbol):
    """Get stock data with rate limiting"""
    return yf.Ticker(ticker_symbol)

@rate_limit(max_requests=50, time_window=3600)
def get_sec_data(query):
    """Get SEC data with rate limiting"""
    return sec_client.get_filings(query)

if __name__ == "__main__":
    api = InvestmentResearchAPI()
