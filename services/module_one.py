import re
import os
import pandas as pd
from pinecone import Pinecone
from dotenv import load_dotenv
from pinecone import ServerlessSpec
from sentence_transformers import SentenceTransformer


load_dotenv()

# Step 1: Data Cleaning
def clean_dataset(file_path):
    # Load dataset
    df = pd.read_csv(file_path)  # Assuming tab-separated data
    df = df[:1000]
    # Step 1: Remove unwanted characters and symbols
    def clean_text(text):
        if isinstance(text, str):
            # Remove non-alphanumeric characters (except spaces, dots, and commas)
            text = re.sub(r'[^a-zA-Z0-9\s.,]', '', text)
            # Remove extra spaces
            text = re.sub(r'\s+', ' ', text).strip()
        return text

    # Apply cleaning to all columns
    for col in df.columns:
        df[col] = df[col].apply(clean_text)

    # Step 2: Standardize InvoiceNo and StockCode
    df['InvoiceNo'] = df['InvoiceNo'].str.replace(r'[^0-9]', '', regex=True)
    df['StockCode'] = df['StockCode'].str.replace(r'[^0-9]', '', regex=True)

    # Step 3: Clean and standardize UnitPrice
    df['UnitPrice'] = df['UnitPrice'].str.replace(r'[^0-9.]', '', regex=True)
    df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce')

    # Step 4: Clean and standardize CustomerID
    df['CustomerID'] = df['CustomerID'].str.replace(r'[^0-9]', '', regex=True)
    df['CustomerID'] = pd.to_numeric(df['CustomerID'], errors='coerce')

    # # Step 5: Clean and standardize Country
    # df['Country'] = df['Country'].str.replace(r'[^a-zA-Z\s]', '', regex=True)
    # df['Country'] = df['Country'].str.strip()

    # Step 5: Clean and standardize Country
    def clean_country(country):
        if isinstance(country, str):
            # Remove unwanted prefixes like 'XxY'
            country = re.sub(r'^XxY', '', country)
            # Remove special characters and extra spaces
            country = re.sub(r'[^a-zA-Z\s]', '', country)
            country = country.strip()
        return country

    df['Country'] = df['Country'].apply(clean_country)

    # Step 6: Handle missing values
    df = df.dropna()
    df = df[df['StockCode'].str.strip() != '']
    print(df.iloc[386])
    # Step 7: Remove duplicates
    df = df.drop_duplicates(subset=['InvoiceNo', 'StockCode', 'Description'])

    # Step 8: Save cleaned dataset
    # df.to_csv('cleaned_ecommerce_data.csv', index=True)
    return df

# Step 2: Vector Database Creation
def setup_pinecone(api_key, index_name, dimension=384):
    # Initialize Pinecone
    #  pinecone.init(api_key=api_key, environment="us-west1-gcp")
    pinecone = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    # Create an index if it doesn't exist
    if index_name not in pinecone.list_indexes().names():
        print("I think it's going to create a new index")
        pinecone.create_index(index_name, 
                              dimension=dimension,
                                spec=ServerlessSpec(
                                cloud="aws",
                                region="us-east-1"
                            ))

    # Connect to the index
    index = pinecone.Index(index_name)
    return index

def generate_product_embeddings(df, model):
    # Generate embeddings for product descriptions
    product_embeddings = []
    for _, row in df.iterrows():
        description = row['Description']
        id = row['StockCode']
        embedding = model.encode(description)
        product_embeddings.append({
            "id": id,
            "vector": embedding.tolist(),
            "metadata": {
                "Description": description,
                "UnitPrice": row['UnitPrice'],
                "Country": row['Country']
            }
        })
    return product_embeddings

def insert_into_pinecone(index, product_embeddings):
    # Insert vectors into Pinecone
    print('getting')
    vectors = [(item["id"], item["vector"], item["metadata"]) for item in product_embeddings]
    index.upsert(vectors)

# Step 3: Product Recommendation Service
def recommend_products(index, model, query, top_k=5):
    # Encode the query into a vector
    query_embedding = model.encode(query)

    # Query Pinecone index
    results = index.query(vector=query_embedding.tolist(), top_k=top_k, include_metadata=True)

    # print(results)
    # Extract product matches
    matches = []
    for match in results['matches']:
        matches.append({
            "StockCode": match['id'],
            "Description": match['metadata']['Description'],
            "UnitPrice": match['metadata']['UnitPrice'],
            "Country": match['metadata']['Country'],
        })

    # Generate natural language response
    
    return matches

# Main Execution
def flask_api_route(query):
    # Step 1: Clean the dataset
    file_path = 'data\dataset\dataset.csv'
    cleaned_df = clean_dataset(file_path)

    # Step 2: Set up Pinecone and insert product embeddings
    pinecone_api_key = "pcsk_2o3gSX_C7QaUf2eEL4bM6T1LniMu13G8Qdh5pBJhrsQgKgjTvZAm8PoLC4R75keviZMp7X"
    index_name = "product-recommendations"
    index = setup_pinecone(pinecone_api_key, index_name)

    # Load a pre-trained sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # # Generate embeddings and insert into Pinecone
    product_embeddings = generate_product_embeddings(cleaned_df, model)
    insert_into_pinecone(index, product_embeddings)

    # Step 3: Test the recommendation service
    recommendation = recommend_products(index, model, query)
    return recommendation


# if __name__ == "__main__":
#     query = "blue dress"
#     matches = flask_api_route(query)
#     print(matches)