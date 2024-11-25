import os
import ssl
import certifi
import pandas as pd
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import requests
import warnings
from generic import GenericFunction
from logger import Logger

warnings.filterwarnings("ignore")
os.environ['CURL_CA_BUNDLE'] = './certificates/huggingface.co.pem'
ssl._create_default_https_context = ssl.create_default_context(cafile=certifi.where())

class DataRetriver:
    def __init__(self):
        try:
            self.logger = Logger()
            generic = GenericFunction()
            self.text_embeddings_model = SentenceTransformer(generic.get_value('index_text_model'), trust_remote_code=True)
            self.cross_encoder = CrossEncoder(generic.get_value('text_cross_encoder'))
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.clip_model = CLIPModel.from_pretrained(generic.get_value('index_image_model')).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(generic.get_value('index_image_model'))
            self.collection = self.connect_to_mongo(generic.get_value('mongo_uri'), generic.get_value('mongo_db'), generic.get_value('mongo_collection'))
        except Exception as e:
            self.logger.log_info(f"Error initializing models or connecting to MongoDB: {e}")

    def connect_to_mongo(self, uri, db_name, collection_name):
        try:
            client = MongoClient(uri)
            db = client[db_name]
            return db[collection_name]
        except Exception as e:
            self.logger.log_error(f"Error connecting to MongoDB: {e}")
            return None
            
    def retrieve_similar_texts(self, query_text, top_n=3,pdf_name_filter=None):
        try:
            query_filter = {"PDF Name": pdf_name_filter} if pdf_name_filter else {}
            documents = list(self.collection.find(query_filter))
            text_embeddings = [doc.get("Text Embedding", []) for doc in documents]
            pdf_names = [doc.get("PDF Name", "") for doc in documents]
            page_numbers = [doc.get("Page Number", "") for doc in documents]
            texts = [doc.get("Page Text(Tesseract)", "") for doc in documents]
            table_jsons = [doc.get("Table JSON", "") for doc in documents]
            image_descriptions = [doc.get("Image Description", "") for doc in documents]
            combined_text = [doc.get("Combined Text", "") for doc in documents]

            query_embedding = self.text_embeddings_model.encode([query_text])[0]
            similarities = cosine_similarity([query_embedding], np.vstack(text_embeddings)).flatten()
            top_indices = similarities.argsort()[-(top_n*2):][::-1]

            candidates = pd.DataFrame({
                "PDF Name": [pdf_names[i] for i in top_indices],
                "Page Number": [page_numbers[i] for i in top_indices],
                "Combined Text": [combined_text[i] for i in top_indices],
                "Page Text(Tesseract)": [texts[i] for i in top_indices],
                "Table JSON": [table_jsons[i] for i in top_indices],
                "Image Description": [image_descriptions[i] for i in top_indices],
                "Cosine Similarity": [similarities[i] for i in top_indices]
            })

            pairs = [(query_text, row["Combined Text"]) for _, row in candidates.iterrows()]
            scores = self.cross_encoder.predict(pairs)
            candidates["Cross-Encoder Score"] = scores
            candidates = candidates.sort_values(by="Cross-Encoder Score", ascending=False)

            results = []
            for idx, row in candidates.head(top_n).iterrows():
                results.append({
                    "Document": f"Document {idx + 1}",
                    "PDF Name": row["PDF Name"],
                    "Page Number": row["Page Number"],
                    "Text": row["Page Text(Tesseract)"],
                    "Table JSON": row["Table JSON"],
                    "Image Description": row["Image Description"]
                })

            return {"Context": results}
        except Exception as e:
            self.logger.log_error(f"Error in text retrieval: {e}")
            return None

    def retrieve_similar_images(self, query_image_path, top_n=1, pdf_name_filter=None):
        try:
            query_filter = {"PDF Name": pdf_name_filter} if pdf_name_filter else {}
            documents = list(self.collection.find(query_filter))
            image_embeddings = [doc.get("Image Embeddings", []) for doc in documents]
            pdf_names = [doc.get("PDF Name", "") for doc in documents]
            page_numbers = [doc.get("Page Number", "") for doc in documents]
            texts = [doc.get("Page Text(Tesseract)", "") for doc in documents]
            table_jsons = [doc.get("Table JSON", "") for doc in documents]
            image_descriptions = [doc.get("Image Description", "") for doc in documents]
    
            if query_image_path.startswith("http://") or query_image_path.startswith("https://"):
                response = requests.get(query_image_path)
                query_image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                query_image = Image.open(query_image_path).convert("RGB")
    
            query_inputs = self.clip_processor(images=query_image, return_tensors="pt").to(self.device)
            query_embedding = self.clip_model.get_image_features(**query_inputs).cpu().detach().numpy().flatten()
    
            max_similarity = -1
            best_row_index = -1
            best_image_index = -1
    
            for row_idx, doc_images in enumerate(image_embeddings):
                if not doc_images:
                    continue
    
                doc_embeddings = np.array(doc_images)
                similarities = cosine_similarity(query_embedding.reshape(1, -1), doc_embeddings).flatten()
                max_row_similarity = similarities.max()
                best_image_idx_in_row = similarities.argmax()
    
                if max_row_similarity > max_similarity:
                    max_similarity = max_row_similarity
                    best_row_index = row_idx
                    best_image_index = best_image_idx_in_row
            if best_row_index != -1:
                result = {
                    "Context": {
                        "Document": f"Document {best_row_index + 1}",
                        "PDF Name": pdf_names[best_row_index],
                        "Page Number": page_numbers[best_row_index],
                        "Text": texts[best_row_index],
                        "Table JSON": table_jsons[best_row_index],
                        "Image Description": image_descriptions[best_row_index],
                        "Similarity": max_similarity
                    }
                }
                return result
            else:
                return None
        except Exception as e:
            print(f"Error in image retrieval: {e}")
            return None




