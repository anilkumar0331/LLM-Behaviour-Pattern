import spacy
from scipy.spatial.distance import cosine
import torch
from sentence_transformers import SentenceTransformer, util
import numpy as np


with open('cleaned.txt', 'r', encoding='utf-8') as file:
    # Read the content of the file
    file_content = file.read()

# Load models
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_semantically_different_sentences(original_text, generated_text, similarity_threshold=0.75):
    # Split the texts into sentences
    original_sentences = [sent.text.strip() for sent in nlp(original_text).sents]
    generated_sentences = [sent.text.strip() for sent in nlp(generated_text).sents]

    # Compute embeddings for all sentences in both texts
    original_embeddings = model.encode(original_sentences, convert_to_tensor=True)
    generated_embeddings = model.encode(generated_sentences, convert_to_tensor=True)

    different_sentences = []

    # Compare each generated sentence to all original sentences
    for gen_sent, gen_emb in zip(generated_sentences, generated_embeddings):
        # Compute cosine similarities
        similarities = util.cos_sim(gen_emb, original_embeddings)
        max_similarity = torch.max(similarities).item()
        
        # If max similarity is below the threshold, consider the sentence significantly different
        if max_similarity < similarity_threshold:
            different_sentences.append(gen_sent)
            
    return different_sentences

def calculate_overall_similarity_score(original_text, generated_text):
    # Encode the whole texts to get their embeddings
    original_embedding = model.encode(original_text, convert_to_tensor=True)
    generated_embedding = model.encode(generated_text, convert_to_tensor=True)

    # Compute cosine similarity between the two embeddings
    similarity = util.cos_sim(original_embedding, generated_embedding)
    return similarity.item()

original_text = file_content
llm_generated_answer = "Sundar Pichai was born on July 12, 1972, in Chennai, India. He grew up in a middle-class family and showed an early aptitude for academics. Pichai attended the Indian Institute of Technology in Kharagpur, where he earned his bachelor's degree in metallurgical engineering. He then moved to the United States, where he received a scholarship to attend Stanford University, where he earned his Master's degree in material sciences and engineering. He also holds an MBA from the Wharton School of the University of Pennsylvania.Pichai began his career as a management consultant before joining Google in 2004. He started out as the head of product management for Google Chrome and oversaw the launch of the popular web browser. He then went on to lead the development of other popular products, including Google Drive, Gmail, and Google Maps. In 2015, Pichai was named the CEO of Google, taking over from the company's founders, Larry Page and Sergey Brin. As CEO, Pichai has overseen Google's continued growth and expansion into new areas, including artificial intelligence and cloud computing.Outside of work, Pichai is known for his philanthropy and advocacy for education. He has donated millions of dollars to charitable causes, including providing funding for schools and education programs in his native India."
# Find semantically different sentences

# Calculate and print the overall similarity score
overall_similarity_score = calculate_overall_similarity_score(original_text, llm_generated_answer)
# print(f"Overall Similarity Score between the original and generated texts: {overall_similarity_score:.2f}")

different_sentences = get_semantically_different_sentences(original_text, llm_generated_answer)

# Print them
print("Sentences from the LLM-generated text significantly different from the original:")
for sent in different_sentences:
    print(f"- {sent}")
    
# ======================================================

# def extract_entities_and_sentences(text):
#     doc = nlp(text)
#     entities = [(ent.text, ent.label_) for ent in doc.ents]
#     sentences = [sent.text for ent in doc.ents for sent in doc.sents if ent.start_char >= sent.start_char and ent.end_char <= sent.end_char]
#     return entities, set(sentences)

# def find_hallucinated_content(original_text, generated_text):
#     original_entities, _ = extract_entities_and_sentences(original_text)
#     generated_entities, generated_sentences = extract_entities_and_sentences(generated_text)
    
#     # Find entities in the generated text not present in the original
#     hallucinated_entities = set(generated_entities) - set(original_entities)
    
#     # Find sentences containing these entities
#     hallucinated_sentences = [sent for ent in hallucinated_entities for sent in generated_sentences if ent[0] in sent]
    
#     return list(set(hallucinated_sentences)), list(hallucinated_entities)

# # Example usage
# original_text = file_content
# llm_generated_answer = "Sundar Pichai was born on July 12, 1972, in Chennai, India. He grew up in a middle-class family and showed an early aptitude for academics. Pichai attended the Indian Institute of Technology in Kharagpur, where he earned his bachelor's degree in metallurgical engineering. He then moved to the United States, where he received a scholarship to attend Stanford University, where he earned his Master's degree in material sciences and engineering. He also holds an MBA from the Wharton School of the University of Pennsylvania.Pichai began his career as a management consultant before joining Google in 2004. He started out as the head of product management for Google Chrome and oversaw the launch of the popular web browser. He then went on to lead the development of other popular products, including Google Drive, Gmail, and Google Maps. In 2015, Pichai was named the CEO of Google, taking over from the company's founders, Larry Page and Sergey Brin. As CEO, Pichai has overseen Google's continued growth and expansion into new areas, including artificial intelligence and cloud computing.Outside of work, Pichai is known for his philanthropy and advocacy for education. He has donated millions of dollars to charitable causes, including providing funding for schools and education programs in his native India."

# hallucinated_sentences, hallucinated_entities = find_hallucinated_content(original_text, llm_generated_answer)

# print("Sentences with Potentially Hallucinated Content:")
# for sentence in hallucinated_sentences:
#     print(f"- {sentence}")
    
# ===========================================================

    

    
# data_structure = [
#     {
#         id: 1
#         question: "Question",
#         llm_answer: "llm generated answer",
#         urls_info: [
#             {
#             url: "URL",
#             context: "Original context fetched through webscraping",
#             similarity_score: 23,
#             hallucinated_senteces: []
#         },
#         {
#             url: "URL",
#             context: "Original context fetched through webscraping",
#             similarity_score: 23,
#             hallucinated_senteces: ['sentence 1', 'sentence 1']
#         },
#         {
#             url: "URL",
#             context: "Original context fetched through webscraping",
#             similarity_score: 23,
#             hallucinated_senteces: ['sentence 1', 'sentence 1']
#         }
#         ],
        
        
#         common_hallucinated_senteces = []
        
#     }
    
# ]    
