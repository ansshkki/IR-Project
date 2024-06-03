from fastapi import FastAPI, Query
from search_engine import SearchEngine  # Assuming the SearchEngine class is in a file named search_engine.py
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, you can also specify specific origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)
search_engine = SearchEngine()

@app.get("/search")
def search(dataset_name: str, query: str, num_docs: int = 100, top_n: int = 6):
    if(dataset_name == "wikir") : dataset_name = "wikir/en1k/training"
    if(dataset_name == "lotte") : dataset_name = "lotte/recreation/dev/forum"
    results = search_engine.google_search_engine(dataset_name, query, num_docs, top_n)
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# search_engine = SearchEngine()
# search_engine.google_search_engine("wikir/en1k/training", "durabl good typic character long period", 10000, 5)