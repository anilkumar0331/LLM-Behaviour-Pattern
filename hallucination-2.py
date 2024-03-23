from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline
from sentence_transformers import SentenceTransformer, util
import spacy

with open('cleaned.txt', 'r', encoding='utf-8') as file:
    # Read the content of the file
    file_content = file.read()

def generate_questions_for_text(text, model_name="valhalla/t5-small-qg-prepend", num_questions_per_sentence=2):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    questions = []
    
    # Splitting the text into sentences
    input_texts = [f"Ask a validation question about key details in: {text}" for text in text.split('. ') if text]

    for input_text in input_texts:
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        
        # Ensure num_beams is at least as large as num_return_sequences to enable beam search
        num_beams = max(num_questions_per_sentence, 3)

        # Generating questions using beam search
        outputs = model.generate(input_ids, max_length=64, num_return_sequences=num_questions_per_sentence, num_beams=num_beams, early_stopping=True)
        
        for output in outputs:
            question = tokenizer.decode(output, skip_special_tokens=True)
            questions.append(question)
            
    return questions

# Load a pre-trained QA pipeline
qa_pipeline = pipeline("question-answering")

def answer_questions(questions, context):
    answers = []
    for question in questions:
        # Use the QA pipeline to find answers
        answer = qa_pipeline(question=question, context=context)
        answers.append((question, answer['answer'], answer['score']))
    return answers

# Load models
nlp = spacy.load("en_core_web_sm")  # For sentence segmentation
model = SentenceTransformer('all-MiniLM-L6-v2')  # For semantic similarity

def segment_text_into_sentences(text):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    return sentences

def find_hallucinated_sentences(sentences, question_answers):
    hallucinated_sentences = []
    
    for qa in question_answers:
        question, answer, confidence = qa
        highest_similarity = 0
        most_similar_sentence = None
        
        # Compare answer to each sentence in the LLM-generated text
        for sentence in sentences:
            answer_embedding = model.encode(answer, convert_to_tensor=True)
            sentence_embedding = model.encode(sentence, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(answer_embedding, sentence_embedding)
            
            if similarity > highest_similarity:
                highest_similarity = similarity
                most_similar_sentence = sentence
        
        # If the highest similarity score for a sentence with the answer is below a threshold, it might be hallucinated
        # Adjust the threshold based on your criteria (e.g., 0.5 is just an example and might need fine-tuning)
        if highest_similarity < 0.5:
            hallucinated_sentences.append((most_similar_sentence, question, answer, highest_similarity.item()))

    return hallucinated_sentences


# Generate validation questions
llm_generated_answer = "Sundar Pichai was born on July 12, 1972, in Chennai, India. He grew up in a middle-class family and showed an early aptitude for academics. Pichai attended the Indian Institute of Technology in Kharagpur, where he earned his bachelor's degree in metallurgical engineering. He then moved to the United States, where he received a scholarship to attend Stanford University, where he earned his Master's degree in material sciences and engineering. He also holds an MBA from the Wharton School of the University of Pennsylvania.Pichai began his career as a management consultant before joining Google in 2004. He started out as the head of product management for Google Chrome and oversaw the launch of the popular web browser. He then went on to lead the development of other popular products, including Google Drive, Gmail, and Google Maps. In 2015, Pichai was named the CEO of Google, taking over from the company's founders, Larry Page and Sergey Brin. As CEO, Pichai has overseen Google's continued growth and expansion into new areas, including artificial intelligence and cloud computing.Outside of work, Pichai is known for his philanthropy and advocacy for education. He has donated millions of dollars to charitable causes, including providing funding for schools and education programs in his native India."
original_context = file_content

questions = generate_questions_for_text(llm_generated_answer)

# Answer questions using the original context
answered_questions = answer_questions(questions, original_context)

# Divide the LLM-generated text into sentences
sentences = segment_text_into_sentences(llm_generated_answer)
# Find hallucinated sentences
hallucinated_sentences = find_hallucinated_sentences(sentences, answered_questions)

# Print hallucinated sentences and their details
for hallucinated_sentence in hallucinated_sentences:
    sentence, question, answer, similarity = hallucinated_sentence
    print(f"Sentence: {sentence}\nQuestion: {question}\nAnswer: {answer}\nSimilarity: {similarity:.2f}\n")


# =====================================================================
