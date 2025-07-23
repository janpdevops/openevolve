import re
import time
import json
from textwrap import dedent

TASK_MODEL_NAME = "mistral-dings"
MAX_RETRIES = 3  # Number of retries for LLM calls

def evaluate_single_example(example, individual_scores, prompt, test_model, total_score):
    input_text = example["text"]
    expected_score = example["label"]
    # Prepare the message for the LLM

    formatpromt = dedent("""
    Make sure your respone is a valid JSON object.
    Return your evaluation as a JSON object with the following format:
{{
    "sentiment_score": [score],
    "reasoning": "[brief explanation of scores]"
}}
    """)
    messages = [
        {"role": "system", "content": formatpromt},
        {"role": "user", "content": prompt.format(input_text=input_text)}
    ]
    # Call the LLM with retry logic
    max_retries = MAX_RETRIES
    for attempt in range(max_retries):
        try:
            response = test_model.chat.completions.create(
                model=TASK_MODEL_NAME,
                messages=messages
            )
            break
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to get response after {max_retries} attempts: {e}")
                raise e
            time.sleep(1)  # Brief pause before retry
    output_text = response.choices[0].message.content.strip()

    # Extract numerical score from the response
    try:
        # parse the output as JSON
        rating_object = json.loads(output_text)
        sentiment_score = rating_object.get("sentiment_score", None)

        if isinstance(sentiment_score, int) or isinstance(sentiment_score, float):
            predicted_score = sentiment_score
        else:
            predicted_score = sentiment_score[0]
            predicted_score = float(predicted_score)

        # Ensure score is within valid range (0-10)
        predicted_score = max(0.0, min(10.0, predicted_score))
        if predicted_score is None:
            predicted_score = 5.0  # Default to neutral score if not provided

        # Calculate accuracy based on how close the prediction is to the expected score
        # Using 1 - (absolute difference / 10), so perfect match = 1.0, worst case = 0.0
        accuracy = 1.0 - (abs(predicted_score - expected_score) / 10.0)
        individual_scores.append(accuracy)
        total_score += accuracy

    except Exception as e:
        print(f"Error processing response '{output_text}': {e}")
        individual_scores.append(0.0)  # Score 0 for failed predictions
    return total_score
