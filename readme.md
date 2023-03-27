This is a proof of concept for using embeddings with the openAI API.
The goal is to answer questions based on the information in the document which the normal GPT cannot answer
The data is converted to embeddings which stored on pinecone. 
Every time a user makes a query it is converted to an embedding and compared with those in the vectro DB.
Then the correct text section is fetched from the data folder.
It is provided to the GPT model together with the query
