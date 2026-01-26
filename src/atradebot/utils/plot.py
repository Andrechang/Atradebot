# report generation and saving to PDF 

import os
import pickle
import pandas as pd
import re
import datetime
import json
import argparse
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import json
import argparse

# Set pandas option
pd.options.mode.copy_on_write = True

from atradebot.config import OUTPUT_D


def extract_date(filename):
    match = re.search(r'\d{4}-\d{2}-\d{2}', filename)
    if match:
        return datetime.datetime.strptime(match.group(), '%Y-%m-%d')
    return None

def get_latest_report(dir):
    paths = os.listdir(dir)
    latest_date = None
    latest_files = []
    for path in paths:
        file_date = extract_date(path)
        if file_date:
            if latest_date is None or file_date > latest_date:
                latest_date = file_date
                latest_files = [path]
            elif file_date == latest_date:
                latest_files.append(path)
    return latest_files 


# Save report to PDF function
def save_report_pdf(report, pdf_path):
    """
    Save the report data to a PDF file.
    
    Args:
        report_out (list): List of report dictionaries
        pdf_path (str): Path to save the PDF file
    """
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    y = height - 40
    
    c.setFont("Helvetica-Bold", 18)
    c.drawString(40, y, "AI Stock Trader Report")
    y -= 30
    
    # Add a page break if needed
    if y < 100:
        c.showPage()
        y = height - 40
    
    # Report date and config
    c.setFont("Helvetica", 12)
    c.drawString(40, y, f"Date: {report.get('present_date', 'N/A')}")
    y -= 20
    
    config_str = str(report.get('config', {}))
    if len(config_str) > 80:  # Split long config strings
        config_parts = [config_str[i:i+80] for i in range(0, len(config_str), 80)]
        for part in config_parts:
            c.drawString(40, y, part)
            y -= 15
    else:
        c.drawString(40, y, f"Config: {config_str}")
        y -= 20
    
    # Stock Research Buy
    if "stock_research_buy" in report:
        y -= 10
        c.setFont("Helvetica-Bold", 14)
        c.drawString(40, y, "BUY REPORT")
        y -= 20
        c.setFont("Helvetica", 11)
        
        buy_report = str(report.get('stock_research_buy', ''))
        if len(buy_report) > 80:  # Split long buy report strings
            buy_parts = [buy_report[i:i+80] for i in range(0, len(buy_report), 80)]
            for part in buy_parts:
                c.drawString(40, y, part)
                y -= 15
        else:
            c.drawString(40, y, f"Buy Report: {buy_report}")
            y -= 20
            

    
    # Stock Rank
    if "stock_rank" in report:
        if y < 100:  # Add a page break if needed
            c.showPage()
            y = height - 40
            
        y -= 10
        c.setFont("Helvetica-Bold", 14)
        c.drawString(40, y, "Stock Rank Table")
        y -= 20
        c.setFont("Helvetica", 11)
        
        # Column headers
        c.drawString(40, y, "Ticker")
        c.drawString(120, y, "Price Change")
        y -= 20
        
        for stock in report.get("stock_rank", []):
            if y < 50:  # Add a page break if close to bottom
                c.showPage()
                y = height - 40
                c.drawString(40, y, "Ticker")
                c.drawString(120, y, "Price Change")
                y -= 20
                
            c.drawString(40, y, str(stock.get('ticker', '')))
            c.drawString(120, y, str(stock.get('price_change', '')))
            y -= 15
    
    # Portfolio
    if "portfolio" in report:
        if y < 100:  # Add a page break if needed
            c.showPage()
            y = height - 40
            
        y -= 10
        c.setFont("Helvetica-Bold", 14)
        c.drawString(40, y, "Portfolio")
        y -= 20
        c.setFont("Helvetica", 11)
        
        # Column headers
        c.drawString(40, y, "Ticker")
        c.drawString(100, y, "Quantity")
        c.drawString(170, y, "Price")
        y -= 20
        
        portfolio_data = report.get("portfolio", {'Ticker': [], 'Quantity': [], 'Price': []})
        portfolio_data_pd = pd.read_json(portfolio_data)
        # Handle portfolio data (could be string, dict, or list)

        for index, row in portfolio_data_pd.iterrows():
            if y < 50:  # Add a page break if close to bottom
                c.showPage()
                y = height - 40
                c.drawString(40, y, "Ticker")
                c.drawString(100, y, "Quantity")
                c.drawString(170, y, "Price")
                y -= 20

            c.drawString(40, y, str(row['Ticker']))
            c.drawString(100, y, str(row['Quantity']))
            c.drawString(170, y, str(row['Price']))
            y -= 15
    
    # Allocation
    if "allocation" in report:
        if y < 100:  # Add a page break if needed
            c.showPage()
            y = height - 40
            
        y -= 10
        c.setFont("Helvetica-Bold", 14)
        c.drawString(40, y, "Allocation")
        y -= 20
        c.setFont("Helvetica", 11)
        
        # Column headers
        c.drawString(40, y, "Ticker")
        c.drawString(100, y, "Quantity")
        y -= 20
        
        for ticker, quantity in report.get("allocation", {}).items():
            if y < 50:  # Add a page break if close to bottom
                c.showPage()
                y = height - 40
                c.drawString(40, y, "Ticker")
                c.drawString(100, y, "Quantity")
                y -= 20
                
            c.drawString(40, y, str(ticker))
            c.drawString(100, y, str(quantity))
            y -= 15
    
    # Stock Explain
    if "stock_collect" in report:
        if y < 100:  # Add a page break if needed
            c.showPage()
            y = height - 40
            
        y -= 10
        c.setFont("Helvetica-Bold", 14)
        c.drawString(40, y, "Stock Explain")
        y -= 20
        c.setFont("Helvetica", 11)
        
        for item in report.get("stock_collect", []):
            if y < 100:  # Add a page break if close to bottom
                c.showPage()
                y = height - 40
                
            c.drawString(40, y, f"Date: {item.get('date', '')}")
            y -= 15
            
            explanation = item.get('summary', '')
            if len(explanation) > 80:  # Split long explanations
                explanation_parts = [explanation[i:i+80] for i in range(0, len(explanation), 80)]
                for part in explanation_parts:
                    c.drawString(50, y, part)
                    y -= 15
            else:
                c.drawString(50, y, explanation)
                y -= 20
            
    c.save()
    print(f"Report saved to PDF: {pdf_path}")

# Run the app
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dashboard for AI Stock Trader")
    parser.add_argument('--dir','-d', type=str, default=OUTPUT_D, help='Path to report directory to pickle')
    args = parser.parse_args()

    paths = os.listdir(args.dir)

    for path in paths:
        if path.endswith('.pkl'):
            reload = os.path.join(args.dir, path)
            with open(reload, 'rb') as f:
                report_out = pickle.load(f)
            print(f'Processing {reload} ...')
            save_report_pdf(report_out, reload.replace('.pkl', '.pdf'))
