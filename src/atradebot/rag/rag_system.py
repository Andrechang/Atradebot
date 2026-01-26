import os
import json
import pickle
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from functools import partial
from tqdm import tqdm
import asyncio
from atradebot.config import *
from atradebot.data.collect_explain import run_collect_explain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Document:
    """Represents a document in the RAG system"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    source: str = ""
    timestamp: Optional[str] = None

class TradingRAGSystem:
    """
    A Retrieval-Augmented Generation system for trading data analysis.
    Loads and indexes data from pkl files (trading reports) and jsonl files (stock database).
    """
    
    def __init__(self):
        """
        Initialize the RAG system.
        
        Args:
            pkl_dir: Directory containing pickle files (default: sd/output)
            jsonl_dir: Directory containing JSONL files (default: sd/)
        """
        self.pkl_dir = os.path.join(FILES_D, 'output') # stocks.py output 
        self.jsonl_dir = OUTPUT_E # collect_explain.py output
        
        self.documents: List[Document] = []
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.8,
            min_df=2
        )
        self.doc_vectors = None
        self.is_fitted = False
        self.load_all_documents()
        
        logger.info(f"Initialized RAG system with PKL dir: {self.pkl_dir}, JSONL dir: {self.jsonl_dir}")
    
    def load_pkl_files(self) -> List[Document]:
        """Load and parse pickle files from the specified directory."""
        pkl_documents = []
        
        if not os.path.exists(self.pkl_dir):
            logger.warning(f"PKL directory {self.pkl_dir} does not exist")
            return pkl_documents
            
        pkl_files = [f for f in os.listdir(self.pkl_dir) if f.endswith('.pkl')]
        logger.info(f"Found {len(pkl_files)} PKL files")
        
        for pkl_file in tqdm(pkl_files, desc="Loading PKL files"):
            try:
                file_path = os.path.join(self.pkl_dir, pkl_file)
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Extract meaningful content from the pickle data
                content_parts = []
                metadata = {
                    'file_name': pkl_file,
                    'file_type': 'pkl',
                    'file_path': file_path
                }
                
                # Parse different types of data structures
                if isinstance(data, dict):
                    # Extract textual content from various fields
                    for key, value in data.items():
                        metadata[f'pkl_{key}'] = str(value)
                        
                        # Add specific content based on key names
                        if 'stock_research_summary' in key.lower():
                            content_parts.append(f"{key}: {str(value)}")
                        elif 'allocation' in key.lower():
                            content_parts.append(f"Investment allocation: {str(value)}")
                
                # Create document
                if content_parts:
                    content = " | ".join(content_parts)
                    doc_id = pkl_file.replace('.pkl', '')
                    timestamp = self._extract_timestamp_from_filename(pkl_file)
                    
                    doc = Document(
                        id=doc_id,
                        content=content,
                        metadata=metadata,
                        source=file_path,
                        timestamp=timestamp
                    )
                    pkl_documents.append(doc)
                    
            except Exception as e:
                logger.error(f"Error loading PKL file {pkl_file}: {e}")
                continue
        
        logger.info(f"Loaded {len(pkl_documents)} documents from PKL files")
        return pkl_documents
    
    def load_jsonl_files(self) -> List[Document]:
        """Load and parse JSONL files from the specified directory."""
        jsonl_documents = {}
        
        if not os.path.exists(self.jsonl_dir):
            logger.warning(f"JSONL directory {self.jsonl_dir} does not exist")
            return jsonl_documents
            
        jsonl_files = [f for f in os.listdir(self.jsonl_dir) if f.endswith('.jsonl')]
        logger.info(f"Found {len(jsonl_files)} JSONL files")
        
        for jsonl_file in tqdm(jsonl_files, desc="Loading JSONL files"): 
            try:
                file_path = os.path.join(self.jsonl_dir, jsonl_file)
                data_list = []
                with open(file_path, 'r') as f:
                    for line in f:
                        data_list.append(json.loads(line))

                for i, data in enumerate(data_list):
                    # Extract content from JSONL entry
                    metadata = {
                        'file_name': jsonl_file,
                        'file_type': 'jsonl',
                        'file_path': file_path,
                        'line_number': i
                    }
                    
                    # Process common fields
                    if isinstance(data, dict):
                        # Create document
                        content = data['summary']
                        doc_id = f"{jsonl_file}_{i}"
                        timestamp = self._extract_timestamp_from_filename(jsonl_file)
                        doc = Document(
                            id=doc_id,
                            content=content,
                            metadata=metadata,
                            source=file_path,
                            timestamp=timestamp
                        )
                        ticker = data.get('ticker') 
                        jsonl_documents[ticker] = doc

            except Exception as e:
                logger.error(f"Error loading JSONL file {jsonl_file}: {e}")
                continue
        
        logger.info(f"Loaded {len(jsonl_documents)} documents from JSONL files")
        return jsonl_documents
    
    def _extract_timestamp_from_filename(self, filename: str) -> Optional[str]:
        """Extract timestamp from filename patterns like 'openai_2025-12-23_manage_fid.pkl'"""
        try:
            # Look for date patterns in filename
            import re
            date_pattern = r'(\d{4}-\d{2}-\d{2})'
            match = re.search(date_pattern, filename)
            if match:
                return match.group(1)
        except:
            pass
        return None
    
    def load_all_documents(self):
        """Load all documents from both PKL and JSONL files."""
        logger.info("Loading all documents...")
        
        self.pkl_docs = self.load_pkl_files()
        self.jsonl_docs = self.load_jsonl_files()
        
        self.documents = self.jsonl_docs
        logger.info(f"Loaded {len(self.documents)} total documents")
            
    async def get_stock_explain(self, tickers: list) -> List[Document]:
        """Retrieve documents related to stock analysis."""
        analysis_docs = ""
        for ticker in tickers:
            if ticker in self.jsonl_docs:
                analysis_docs += f"{ticker} past news are: " + self.jsonl_docs[ticker].content + "\n\n"
            else:
                summary = await run_collect_explain([ticker])
                summ = summary[0]['summary'] if summary else ""
                analysis_docs += f"{ticker} past news are: " + summ + "\n\n"

        return analysis_docs
    


# Example usage and testing functions
async def main():
    """Test the RAG system with sample queries."""
    rag = TradingRAGSystem()
    rag.load_all_documents()
    
    
    # Test queries
    query = ["META", "TSLA", "AAPL"]
    
    print(f"\n{'='*50}")
    print(f"Query: {query}")
    print(f"{'='*50}")
    result = rag.get_stock_explain(query)
    print(f"Answer: {result}...")


if __name__ == "__main__":
    asyncio.run(main()) 