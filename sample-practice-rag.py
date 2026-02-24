from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss

loader=TextLoader("C:\\Users\\egholok\\OneDrive - Ericsson\\Desktop\\GENAI\\RAG\\notebooks\\nvidea.text")
data=loader.load()
print(data[0].page_content)
print(data[0].metadata)


loader1=CSVLoader("C:\\Users\\egholok\\OneDrive - Ericsson\\Desktop\\GENAI\\RAG\\notebooks\\movies.csv",source_column="title")
data1=loader1.load()
print(data1[1].page_content)
print(data1[1].metadata)


loaders = UnstructuredURLLoader(urls=[
    "https://www.moneycontrol.com/news/business/markets/wall-street-rises-as-tesla-soars-on-ai-optimism-11351111.html",
    "https://www.moneycontrol.com/news/business/tata-motors-launches-punch-icng-price-starts-at-rs-7-1-lakh-11098751.html"
])
data = loaders.load() 
print(len(data))

print(data[0].page_content)  # Print the first 500 characters of the first page content



text = """Interstellar is a 2014 epic science fiction film co-written, directed, and produced by Christopher Nolan. 
It stars Matthew McConaughey, Anne Hathaway, Jessica Chastain, Bill Irwin, Ellen Burstyn, Matt Damon, and Michael Caine. 
Set in a dystopian future where humanity is embroiled in a catastrophic blight and famine, the film follows a group of astronauts who travel through a wormhole near Saturn in search of a new home for humankind.

Brothers Christopher and Jonathan Nolan wrote the screenplay, which had its origins in a script Jonathan developed in 2007 and was originally set to be directed by Steven Spielberg. 
Kip Thorne, a Caltech theoretical physicist and 2017 Nobel laureate in Physics,[4] was an executive producer, acted as a scientific consultant, and wrote a tie-in book, The Science of Interstellar. 
Cinematographer Hoyte van Hoytema shot it on 35 mm movie film in the Panavision anamorphic format and IMAX 70 mm. Principal photography began in late 2013 and took place in Alberta, Iceland, and Los Angeles. 
Interstellar uses extensive practical and miniature effects, and the company Double Negative created additional digital effects.

Interstellar premiered in Los Angeles on October 26, 2014. In the United States, it was first released on film stock, expanding to venues using digital projectors. The film received generally positive reviews from critics and grossed over $677 million worldwide ($715 million after subsequent re-releases), making it the tenth-highest-grossing film of 2014. 
It has been praised by astronomers for its scientific accuracy and portrayal of theoretical astrophysics.[5][6][7] Interstellar was nominated for five awards at the 87th Academy Awards, winning Best Visual Effects, and received numerous other accolades."""

spliter=CharacterTextSplitter(separator="\n",chunk_size=100,chunk_overlap=0)
chunks=spliter.split_text(text)
print(chunks)

for chunk in chunks:
    print(chunk)
    print(len(chunk))
    print("----")



r_spliter=RecursiveCharacterTextSplitter(separators=["\n\n","\n", " "],chunk_size=100,chunk_overlap=0)
r_chunks=r_spliter.split_text(text)
print(r_chunks)

for chunk in r_chunks:
    print(chunk)
    print(len(chunk))
    print("----")


pd.set_option('display.max_colwidth', 100)  # To display full content of the column
data=pd.read_csv("C:\\Users\\egholok\\OneDrive - Ericsson\\Desktop\\GENAI\\RAG\\notebooks\\sample_text.csv")
print(data.head())
print(data.shape)



encoder=SentenceTransformer("all-MiniLM-L6-v2")
vectors=encoder.encode(data.text.tolist())
print(vectors.shape)
print(vectors)



dim=vectors.shape[1]
index=faiss.IndexFlatL2(dim)
index.add(vectors)
query="I want to buy a polo t-shirt"
query_vector=encoder.encode([query])
k=2
svec = np.array(query_vector).reshape(1,-1)
svec.shape
distances, indices=index.search(query_vector, k)
print("Distances:", distances)
print("Indices:", indices)
row_indices = indices.tolist()[0]
print(data.loc[row_indices])
