# NLP Text Analysis and Topic Discovery

This project demonstrates an end-to-end workflow for **text analysis and topic discovery** using Natural Language Processing (NLP).  

---

## Dataset
- A collection of **30 short text documents** related to Artificial Intelligence, Data Science, Machine Learning, and Emerging Technologies.  
- Each document is a sentence/paragraph describing a concept (e.g., machine learning, cloud computing, IoT).  

---

## Text Preprocessing
Before applying any NLP techniques, the raw text undergoes **cleaning and tokenization** to ensure meaningful results:  

- **Lowercasing** → converts all text to lowercase (`Data` → `data`).  
- **Punctuation & number removal** → removes symbols, digits, and special characters.  
- **Tokenization** → splits sentences into individual words (tokens).  
- **Stopword removal** → removes common words (like *the, is, and*) that don’t add value.  

 Result: a cleaned list of words for each document, ready for analysis.  

---

## TF-IDF Analysis
- **TF-IDF (Term Frequency – Inverse Document Frequency)** assigns importance to words based on how often they appear in a document vs. across all documents.  
- Implemented using `TfidfVectorizer` from `scikit-learn`.  
- For each document, the **top 10 words with highest TF-IDF scores** are extracted.  

 Helps identify **unique and important terms** in each document.  

---

## Word2Vec Embeddings
- A **Word2Vec model** is trained on the preprocessed corpus using `gensim`.  
- Word2Vec learns **vector representations** of words based on their surrounding context.  

Features:  
- Find **most similar words** (e.g., *data → statistics, information*).  
- Reduce embeddings to 2D using **PCA** for visualization.  
- Plot words to see semantic clusters (words with similar meaning appear close together).  

---

## Topic Modeling (LDA)
- **Latent Dirichlet Allocation (LDA)** is applied to discover hidden topics.  
- Steps:  
  - Convert documents into a **bag-of-words** representation (`gensim.corpora.Dictionary`).  
  - Train LDA to extract **3–5 topics**.  
  - Each topic is represented by its **top words**.  

Example Output:  
- Topic 1 → {data, analysis, business, decision, predictive}  
- Topic 2 → {ai, learning, machine, deep, model}  
- Topic 3 → {blockchain, security, transaction, ledger, digital}  

Each document is then assigned its **dominant topic**.  

---

## Visualization
- **pyLDAvis** is used for **interactive HTML visualization**:  
  - Left panel → bubble chart showing topic distribution.  
  - Right panel → top keywords per topic.  
- Since the project runs as a `.py` script, the visualization is saved as **`lda_topics.html`** and can be opened in a browser.  

Additionally:  
- **Matplotlib bar plots** are generated for static visualization of top words per topic.  

---

## Summary of Processes
1. **Preprocessing** → Clean + tokenize text (remove noise, keep meaningful words).  
2. **TF-IDF** → Extracts most important words per document.  
3. **Word2Vec** → Learns word embeddings, finds semantic similarities, visualizes them.  
4. **LDA** → Discovers hidden topics and assigns them to documents.  
5. **Visualization** → Interactive (pyLDAvis) + Static (Matplotlib).  

---

## Key Insight
This project combines **keyword extraction, semantic similarity, and topic discovery** into one pipeline, providing both **word-level** and **document-level** insights.  
