# -*- coding: utf-8 -*-

import pinecone
import openai
from tqdm import tqdm
import time
import re
import tiktoken
import json

#configs
with open('config.json') as json_file:
    config = json.load(json_file)
file_path = config['file_path']
pinecone.init(api_key=config['pinecone_api_key'],environment=config['pinecone_environment'])
openai.organization = config['openai_organization']
openai.api_key = config['openai_api_key']


# Create Vector DB
if 'book' not in pinecone.list_indexes():
    pinecone.create_index('book', dimension=1536)
index = pinecone.Index('book')


# Get embedding of string
def get_embedding(your_text):
  retries = 0
  while retries < 10:
      try:
          embedding = openai.Embedding.create(input=your_text,model="text-embedding-ada-002")
          embedding = embedding['data'][0]['embedding']
          return embedding
      except:
          retries += 1
          print("failed to get embedding, attempt no. ",retries)
  print('failed to get embedding with max retries, aborted')
  return


#count number of tokens in a string
def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding('cl100k_base')
    num_tokens = len(encoding.encode(string))
    return num_tokens


# split file into list of chapters
def read_file():
    with open(config['file_path'], "r")as f:
      f = f.read()
      f = f.replace("\n", " ")
      list_of_paragraphs = re.split(r'\x0c', f)
      return list_of_paragraphs




# main function to create and upload embeddings
def upload_to_pinecone(list_of_paragraphs):
    for i in tqdm(range(56,len(list_of_paragraphs))):
        print("currently on", i)
        to_insert = []
        embedding = get_embedding(list_of_paragraphs[i])
        name = list_of_paragraphs[i].split(" ")[0]
        name = name.replace("/", "")
        unique_id = name + str(i)
        to_insert.append((unique_id, embedding))
        #print("number of tokens:", num_tokens_from_string(list_of_paragraphs[i]))
        #print(unique_id, list_of_paragraphs[i][:100])
        with open("data/"+unique_id, "w") as f:
            f.write(list_of_paragraphs[i])

        x = index.upsert(to_insert)
        print(x)
        if i % 10 == 0:
            time.sleep(1)

list_of_paragraphs = read_file()
upload_to_pinecone(list_of_paragraphs)
