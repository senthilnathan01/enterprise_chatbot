![image](https://github.com/user-attachments/assets/55d49e5d-fa11-49b5-a929-dbbd8cc9a39a)



**Project Tree**

multimodal_qa_app/

├── main_app.py         # Main Streamlit application, UI, orchestration

├── config.py           # Configuration (model names, constants)

├── utils.py            # General utility functions

├── file_parsers.py     # Functions for parsing different file types

├── image_processor.py  # Image handling, OCR/VLM calls

├── web_crawler.py      # URL finding and crawling functions

├── vector_store.py     # ChromaDB interactions, embedding, context retrieval

├── qa_engine.py        # Q&A logic, data query handling, suggestions

├── requirements.txt    # Dependencies

└── .streamlit/

    └── secrets.toml    # For storing API keys locally


Set up secrets: Create .streamlit/secrets.toml with your GOOGLE_API_KEY or set it as an environment variable.

**Install dependencies:**

`pip install -r requirements.txt`

**Run the main app:**

`streamlit run main_app.py`

or

`python -m streamlit run main_app.py`
