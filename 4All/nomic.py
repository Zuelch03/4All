from flask import Flask, request, render_template
from gpt4all import GPT4All
import os
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import openai

app = Flask(__name__)

# Initialize the GPT4All model
model_name = "orca-mini-3b.ggmlv3.q4_0.bin"
model = GPT4All(model_name)

# Specify a folder to store the uploaded files
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the prompt-response mapping from the JSON file
mapping_file = os.path.join('mappings', 'prompt_response.json')
with open(mapping_file, 'r') as file:
    prompt_response_mapping = json.load(file)

# Load the additional scoring mapping from the JSON file
scoring_file = os.path.join('mappings', 'scoring.json')
with open(scoring_file, 'r') as file:
    scoring_data = json.load(file)

# Create an empty scoring mapping
scoring_mapping = {}

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        user_input = request.form['prompt']

        # Load the conversation history from the JSON file
        if os.path.exists('conversation_history.json'):
            with open('conversation_history.json', 'r') as file:
                conversation_data = json.load(file)
                conversation_history = conversation_data["conversations"]
        else:
            conversation_history = []

        # Check if the user_input is in the prompt_response_mapping or conversation history
        if user_input in prompt_response_mapping:
            # If it is in the prompt_response_mapping, use the corresponding response
            response = prompt_response_mapping[user_input]
        else:
            # Check if the input is in the conversation history
            response = next((item['response'] for item in conversation_history if item['prompt'] == f"User: {user_input}"), None)

            if response is None:
                # Check if the user's input is a question about JSON files
                if user_input.startswith("Question:"):
                    # Extract the question from the user's input
                    question = user_input.split(":")[1].strip()

                    # Check if the question has been asked before
                    if question in scoring_mapping:
                        response = "Question already asked. Please provide a different question."
                    else:
                        scoring_data.append({"Question": question, "Response": ""})
                        scoring_mapping[question] = ""
                        response = question
                else:
                    # If the user's input is not in the prompt_response_mapping, conversation history or about JSON files,
                    # generate a response using the GPT model as before
                    N = 10
                    conversation_strings = [f'{conv["prompt"]}\n{conv["response"]}' for conv in conversation_history]
                    prompt = "\n".join(conversation_strings[-N:]) if conversation_strings else ""
                    prompt += f"\nUser: {user_input}"
                    response = model.generate(prompt)

                    # If the model is not able to generate a response, get it from an external source
                    if response is None or response.strip() == '':
                        response = get_answer_from_external_source(user_input)

        # Store the response in the scoring mapping
        scoring_mapping[user_input] = response

        # Append the new conversation to the conversation history
        conversation_history.append({
            "prompt": f"User: {user_input}",
            "response": response
        })

        # Save the updated conversation history to the JSON file
        with open('conversation_history.json', 'w') as file:
            json.dump({"conversations": conversation_history}, file)

        return render_template('home.html', prompt=user_input, response=response)

    return render_template('home.html', prompt=None, response=None)


@app.route('/ask_questions', methods=['GET', 'POST'])
def ask_questions():
    if request.method == 'POST':
        # Initialize the user's responses dictionary
        user_responses = {}

        # Retrieve the user's responses from the form
        for item in scoring_data:
            question = item['Question']
            user_response = request.form.get(question)
            user_responses[question] = user_response

        # Load the scoring mapping from the JSON file
        with open(scoring_file, 'r') as file:
            scoring_mapping = json.load(file)

        # Update the scoring mapping with the user's responses for new questions
        for question, user_response in user_responses.items():
            if question not in scoring_mapping:
                scoring_mapping.append({"Question": question, "Response": user_response})

        # Save the updated scoring mapping to the JSON file
        with open(scoring_file, 'w') as file:
            json.dump(scoring_mapping, file)

        # Calculate scores based on text similarity
        scores = {}
        for item in scoring_mapping:
            question = item['Question']
            response = item['Response']
            user_response = user_responses[question]
            score = calculate_similarity(response, user_response)
            scores[question] = score

        # Generate a list of unique questions
        seen_questions = set()
        unique_scoring_data = []
        for item in scoring_mapping:
            if item['Question'] not in seen_questions:
                seen_questions.add(item['Question'])
                unique_scoring_data.append(item)

        return render_template('ask_questions.html', scoring_data=unique_scoring_data, scores=scores, user_responses=user_responses)

    return render_template('ask_questions.html', scoring_data=scoring_data)



def calculate_similarity(text1, text2):
    # Check for None values and replace with empty strings
    if text1 is None:
        text1 = ""
    if text2 is None:
        text2 = ""

    # Skip calculation if either text is empty
    if text1 == "" or text2 == "":
        return 0

    corpus = [text1, text2]
    vectorizer = CountVectorizer().fit_transform(corpus)
    vectors = vectorizer.toarray()
    similarity = cosine_similarity([vectors[0]], [vectors[1]])[0][0]

    # Scoring scheme
    if similarity < 0.3:
        return 0
    elif similarity < 0.7:
        return 5
    else:
        return 10



openai.api_key = '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'

def get_answer_from_external_source(prompt):
    try:
        response = openai.Completion.create(
          engine="text-davinci-002", # or any other engine you want to use
          prompt=prompt,
          temperature=0.5,
          max_tokens=100
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"Error: {e}")
        return "There was an error while processing your request."


if __name__ == '__main__':
    app.run(debug=True)
