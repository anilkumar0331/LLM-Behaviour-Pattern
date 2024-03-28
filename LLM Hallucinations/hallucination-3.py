# =================================
# Load dataset
# =================================
import json

with open('gptrc_urls_context_llm_answers.json', 'r') as file:
    # Read the content of the file
    gptrc_urls_context_llm_answers = json.load(file)
    
print(len(gptrc_urls_context_llm_answers))

# =================================
# Load Gradio Client and find hallucinations
# =================================
from gradio_client import Client
client = Client("https://fava-uw-fava.hf.space/--replicas/igsvg/")

gptrc_urls_context_llm_answers_test = gptrc_urls_context_llm_answers

def find_hallucinations(context_llm_answers):
    hallucinations = []
    for item in context_llm_answers:
        llm_answer = item['llm_answer']
        for url_info in item['urls_info']:
            context = url_info['context']
            result = client.predict(
    		llm_answer,	# str  in 'passage' Textbox component
    		context,	# str  in 'reference' Textbox component
    		api_name="/predict")
            hallucinations.append({'question': item['question'],
                                   'prediction_result': result})
            
    return hallucinations

result_1 = find_hallucinations(gptrc_urls_context_llm_answers_test[:5])
result_2 = find_hallucinations(gptrc_urls_context_llm_answers_test[6:10])

final_result = result_1 + result_2        

def write_results_to_file(results, output_file):
    with open(output_file, "w") as json_file:
         json.dump(results, json_file)
    print(f"Results have been saved to {output_file}")
    
# Save the modified data to a new JSON file
write_results_to_file(final_result, "hallucinations.json")

        