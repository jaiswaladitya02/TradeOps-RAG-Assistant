\# TradeOps RAG Assistant



A local Retrieval-Augmented Generation (RAG) assistant for answering Trade Operations questions using an enterprise trade operations manual.



\## Overview



TradeOps RAG Assistant combines:



\- PDF document ingestion

\- Text chunking

\- Local semantic embeddings

\- ChromaDB vector search

\- Retrieval-Augmented Generation

\- Local Qwen2.5 7B LLM through Ollama

\- FastAPI backend

\- HTML/CSS/JavaScript frontend

\- Source and page citations



The system retrieves relevant sections from the Trade Operations manual and uses them as context for generating grounded answers.



\## Architecture



```text

Trade Operations PDF

&#x20;       |

&#x20;       v

&#x20;  PDF Loader

&#x20;       |

&#x20;       v

&#x20;   Chunking

&#x20;       |

&#x20;       v

Local Embeddings

&#x20;       |

&#x20;       v

&#x20;   ChromaDB

&#x20;       |

&#x20;       |  User Question

&#x20;       v

Query Embedding

&#x20;       |

&#x20;       v

Semantic Retrieval

&#x20;       |

&#x20;       v

Relevant Context

&#x20;       |

&#x20;       v

Qwen2.5 7B (Ollama)

&#x20;       |

&#x20;       v

Answer + Sources

Technology Stack

Component	Technology

Language	Python

RAG	Custom Python RAG pipeline

Embeddings	Sentence Transformers

Vector Database	ChromaDB

LLM	Qwen2.5 7B

LLM Runtime	Ollama

Backend	FastAPI

Server	Uvicorn

Frontend	HTML, CSS, Vanilla JavaScript

Source Data	PDF

Project Structure

TradeOps-RAG-Assistant/

│

├── app/

│   ├── api.py

│   ├── chat.py

│   ├── chunker.py

│   ├── embeddings.py

│   ├── ingest.py

│   ├── llm.py

│   ├── pdf\_loader.py

│   ├── rag.py

│   ├── retriever.py

│   ├── vector\_store.py

│   └── \_\_init\_\_.py

│

├── data/

│   └── raw/

│       └── TradeOps\_Enterprise\_Manual\_Improved.pdf

│

├── frontend/

│   ├── index.html

│   ├── app.js

│   └── style.css

│

├── tests/

│   ├── test\_rag\_evaluation.py

│   ├── test\_retriever.py

│   ├── test\_vector\_store.py

│   └── ...

│

├── vector\_store/

├── requirements.txt

└── README.md

Setup



Create and activate the virtual environment:



python -m venv venv

.\\venv\\Scripts\\Activate.ps1



Install dependencies:



pip install -r requirements.txt



Make sure Ollama is installed and the Qwen model is available:



ollama pull qwen2.5:7b

Build the Vector Database



Place the source PDF in:



data/raw/TradeOps\_Enterprise\_Manual\_Improved.pdf



Run ingestion:



python -m app.ingest



This extracts the PDF text, creates chunks, generates embeddings, and stores them in ChromaDB.



Run the Backend



Start FastAPI:



uvicorn app.api:app --host 127.0.0.1 --port 8000 --reload



API:



http://127.0.0.1:8000



Swagger documentation:



http://127.0.0.1:8000/docs

Run the Frontend



Open another PowerShell terminal and activate the virtual environment if required.



From the project root:



python -m http.server 5500 --directory frontend



Open:



http://127.0.0.1:5500

CLI Mode



The assistant can also be used directly from the terminal:



python -m app.chat

API Example



Send a POST request to:



POST /ask



Example request:



{

&#x20; "question": "How are trade breaks investigated?"

}



The API returns the generated answer together with the retrieved source information.



Evaluation



The project includes a RAG evaluation suite covering:



Retrieval quality

Answer accuracy

Citation accuracy

Out-of-scope handling



The final evaluation achieved:



Total tests:        21

Retrieval STRONG:   17

Retrieval WEAK:     2

Retrieval POOR:     2



Strong retrieval:   81.0%

Answer accuracy:    100.0%

Citation accuracy:  100.0%

Out-of-scope:       100.0%



Overall RAG score:  100.0%



Run the evaluation with:



python -m tests.test\_rag\_evaluation

Key Features

Semantic Retrieval



User questions are converted into embeddings and compared against document chunks stored in ChromaDB.



Duplicate Filtering



The retriever removes exact duplicate document chunks while preserving the original relevance ordering.



Grounded Generation



The LLM receives retrieved document context instead of answering solely from its general knowledge.



Source Citations



Answers include source document and page information for traceability.



Out-of-Scope Handling



The evaluation suite verifies that questions outside the Trade Operations knowledge base are handled appropriately.



Important Notes



The vector database is generated locally and is excluded from Git tracking.



The local Ollama model is also not included in the repository.



The source PDF is included in the repository so the knowledge base can be recreated.



Git



The project is maintained using Git and hosted on GitHub.



Repository:



TradeOps-RAG-Assistant



Future Improvements



Possible future enhancements include:



Improved hybrid keyword + semantic retrieval

Reranking models

Conversation memory

Authentication

Production deployment

Streaming LLM responses

More extensive automated evaluation

Additional Trade Operations documents

