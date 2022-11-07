import os
import numpy as np
import pandas as pd
from keybert import KeyBERT
    
        
def sumarry_embedding():
    load_path = './data/processed/'
    books = pd.read_csv(load_path + 'books.csv')
    
    books.loc[books['summary'].isna(), 'summary']=''
    emb = []
    kw_model = KeyBERT()
    
    for i in range(len(books)):
        doc = books.iloc[i]['summary'].replace('\n', '') + books.iloc[i]['book_title']
        keywords = kw_model.extract_keywords(doc, top_n=1, stop_words='english')
        if len(keywords)==0:
            emb.append(books.iloc[i]['book_title'].split()[-1])
        else:
            emb.append(keywords[0][0])
    
    books['summary']=emb
    data_path = './data/processed/'
    books.to_csv(data_path + 'books_summary.csv', index=False)
        
if __name__ == "__main__":
    print(f'--------------- summary embedding.csv ---------------')
    sumarry_embedding()