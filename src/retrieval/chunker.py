import os
import json
import fitz
from bidi.algorithm import get_display
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken

class ChunkRegistryBuilder:
    def __init__(self, metadata_path, output_path):
        self.metadata_path = metadata_path
        self.output_path = output_path
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Token-based chunking configuration (~400 tokens, 50 overlap)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50,
            length_function=lambda x: len(self.tokenizer.encode(x)),
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
    def build_registry(self):
        print(f"Reading metadata from {self.metadata_path}")
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            articles = json.load(f)
            
        chunks = []
        
        for art in articles:
            pdf_path = art["split_pdf_path"]
            if not os.path.exists(pdf_path):
                print(f"Warning: {pdf_path} not found. Skipping.")
                continue
                
            doc = fitz.open(pdf_path)
            for local_page_num in range(len(doc)):
                page = doc.load_page(local_page_num)
                # Naive header/footer strip: crop top 10% and bottom 10%
                rect = page.rect
                clip = fitz.Rect(rect.x0, rect.y0 + rect.height * 0.1, rect.x1, rect.y1 - rect.height * 0.1)
                text = page.get_text("text", clip=clip)
                
                # Bidi Reordering for Hebrew
                lines = text.split('\n')
                reordered_text = '\n'.join([get_display(line) for line in lines])
                
                # Split into chunks
                page_chunks = self.text_splitter.split_text(reordered_text)
                
                global_page_num = art["start_page"] + local_page_num
                
                for i, text_chunk in enumerate(page_chunks):
                    if not text_chunk.strip(): continue
                    chunk_id = f"{art['article_id']}_p{global_page_num}_c{i+1}"
                    
                    chunks.append({
                        "chunk_id": chunk_id,
                        "article_id": art["article_id"],
                        "article_title": art["title"],
                        "split_pdf_path": pdf_path,
                        "source_pdf_path": art["source_pdf"],
                        "global_page_num": global_page_num,
                        "local_page_num": local_page_num + 1,
                        "text": text_chunk
                    })
            doc.close()
            
        # Write JSONL
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                
        print(f"Generated {len(chunks)} chunks in {self.output_path}")
        return chunks

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    meta_path = os.path.join(base_dir, "data", "processed", "split_pdfs", "articles_metadata.json")
    out_path = os.path.join(base_dir, "data", "processed", "chunks_registry.jsonl")
    builder = ChunkRegistryBuilder(meta_path, out_path)
    builder.build_registry()
