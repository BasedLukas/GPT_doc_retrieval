
import pinecone
import openai
import time
import tiktoken
import json




with open('config.json') as json_file:
    config = json.load(json_file)
file_path = config['file_path']
pinecone.init(api_key=config['pinecone_api_key'],environment=config['pinecone_environment'])
openai.organization = config['openai_organization']
openai.api_key = config['openai_api_key']
index = pinecone.Index('book')



def num_tokens(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding('cl100k_base')
    num_tokens = len(encoding.encode(string))
    return num_tokens


def get_gpt_response(user_query, system_query):
  completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
    {"role": "user", "content": user_query},
    {"role": "system", "content":system_query}   ])

  return completion['choices'][0]['message']['content']
    
def query(user_input, data):
  #user_input = input("")
  tokens = num_tokens(user_input)
  #data = input("")
  data_vector = openai.Embedding.create(input=data,model="text-embedding-ada-002")['data'][0]['embedding']
  vector_matches = index.query(vector=data_vector,top_k=3,include_values=True)['matches']
  context = ""
  for i in vector_matches:
        filename = i['id']
        text = open("data/"+filename, "r").read()
        text_tokens = num_tokens(text)
        if tokens + text_tokens < 4000:
            context += " | " + text
            tokens += text_tokens
  
  system_prompt = f"Answer the users question using the following context: {context}"
  return user_input, system_prompt


