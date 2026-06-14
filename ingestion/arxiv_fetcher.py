import arxiv
import requests
import os
from pypdf import PdfReader
from io import BytesIO

class ArxivFetcher:
    def __init__(self, download_dir="data/papers"):
        self.download_dir = download_dir
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)

    def search_and_download(self, query, max_results=10):
        print(f"Searching arXiv for: {query}")
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        papers = []
        for result in client.results(search):
            paper_info = {
                'id': result.get_short_id(),
                'title': result.title,
                'summary': result.summary,
                'pdf_url': result.pdf_url,
                'published': result.published,
                'authors': [author.name for author in result.authors]
            }
            # Download and extract text
            text = self.download_and_extract_pdf(result.pdf_url, result.get_short_id())
            paper_info['text'] = text
            papers.append(paper_info)
            print(f"Downloaded and extracted: {result.title}")
            
        return papers

    def download_and_extract_pdf(self, pdf_url, paper_id):
        try:
            response = requests.get(pdf_url)
            response.raise_for_status()
            
            # Save PDF locally
            pdf_path = os.path.join(self.download_dir, f"{paper_id}.pdf")
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
            
            # Extract text
            reader = PdfReader(BytesIO(response.content))
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            return text
        except Exception as e:
            print(f"Failed to process PDF {pdf_url}: {e}")
            return ""

if __name__ == "__main__":
    fetcher = ArxivFetcher()
    papers = fetcher.search_and_download("LLM reasoning", max_results=2)
    for p in papers:
        print(f"Title: {p['title']}")
        print(f"Text length: {len(p['text'])} chars")
