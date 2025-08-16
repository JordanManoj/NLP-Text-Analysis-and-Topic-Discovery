
#Text Analysis and Topic Discovery

import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from gensim import corpora, models
from nltk.corpus import stopwords
import pyLDAvis
import pyLDAvis.gensim_models


nltk.download('punkt')
nltk.download('stopwords')

# text Preprocessing

documents = [
    "Data science is an interdisciplinary field of study.",
    "Machine learning enables computers to learn from data.",
    "Deep learning is a subset of machine learning.",
    "Natural language processing deals with text and speech.",
    "Data analysis helps in making business decisions.",
    "Artificial intelligence is transforming healthcare and education.",
    "Big data technologies allow storage and analysis of massive datasets.",
    "Robotics combines engineering, computer science, and AI.",
    "Cloud computing provides scalable resources over the internet.",
    "Cybersecurity ensures protection of data and digital systems.",
    "Reinforcement learning trains agents through rewards and penalties.",
    "Computer vision allows machines to understand images and videos.",
    "Recommendation systems suggest products based on user behavior.",
    "Social media analytics provides insights into customer sentiment.",
    "Autonomous vehicles rely on sensors, AI, and real-time processing.",
    "Speech recognition enables voice assistants like Alexa and Siri.",
    "Blockchain technology secures transactions using distributed ledgers.",
    "Genomics research benefits from AI in analyzing DNA sequences.",
    "Smart cities use IoT devices to improve urban infrastructure.",
    "Edge computing processes data closer to where it is generated.",
    "Virtual reality creates immersive digital environments for users.",
    "Augmented reality overlays digital content onto the physical world.",
    "Quantum computing promises breakthroughs in solving complex problems.",
    "5G networks enable faster communication and connected devices.",
    "Predictive analytics helps organizations forecast future trends.",
    "E-commerce platforms rely on recommendation algorithms for sales.",
    "Digital marketing uses data to target customer preferences.",
    "Chatbots provide instant customer support using NLP techniques.",
    "Wearable devices track health data like heart rate and activity.",
    "Supply chain optimization uses AI to improve logistics efficiency."
]

stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    return tokens

processed_docs = [preprocess(doc) for doc in documents]

print("Preprocessed Documents:", processed_docs)


#TF-IDF Analysis
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)
tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

for i, doc in enumerate(tfidf_df.values):
    top_indices = doc.argsort()[-10:][::-1]
    top_words = [(vectorizer.get_feature_names_out()[j], doc[j]) for j in top_indices if doc[j] > 0]
    print(f"\nTop words in Document {i+1}: {top_words}")

# Word2Vec Embeddings
model = Word2Vec(sentences=processed_docs, vector_size=50, window=5, min_count=1, workers=4)

#finding similar words
print("\nSimilar words to 'data':", model.wv.most_similar("data", topn=5))
print("Similar words to 'analysis':", model.wv.most_similar("analysis", topn=5))

words = list(model.wv.index_to_key)
vectors = [model.wv[w] for w in words]
pca = PCA(n_components=2)
x_vals = pca.fit_transform(vectors)

plt.figure(figsize=(8,6))
plt.scatter(x_vals[:,0], x_vals[:,1])
for i, word in enumerate(words):
    plt.annotate(word, (x_vals[i,0], x_vals[i,1]))
plt.title("Word2Vec Embeddings (PCA)")
plt.show()

# Modeling (LDA)
dictionary = corpora.Dictionary(processed_docs)
corpus = [dictionary.doc2bow(text) for text in processed_docs]

lda_model = models.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=10)

print("\nLDA Topics:")
for idx, topic in lda_model.print_topics(num_words=5):
    print(f"Topic {idx+1}: {topic}")

# assigning the topic to each document
for i, row in enumerate(lda_model[corpus]):
    row = sorted(row, key=lambda x: x[1], reverse=True)
    print(f"Document {i+1} -> Topic {row[0][0]+1}")


# pyLDAvis visualization 
import pyLDAvis
import pyLDAvis.gensim_models
vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)

# This code saves the visualization as HTML (works with VS code)
pyLDAvis.save_html(vis, 'lda_topics.html')

print("LDA visualization is saved as lda_topics.html, this file should be opened in the browser.")

