import re
import time



TASK_MODEL_NAME = "adesso-smart"
MAX_RETRIES = 3  # Number of retries for LLM calls

def evaluate_single_example(example, individual_scores, prompt, test_model, total_score):
    input_text = example["text"]
    expected_score = example["label"]
    # Prepare the message for the LLM
    messages = [
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
        # Try to extract a number between 0 and 10
        score_match = re.search(r'(\d+(?:\.\d+)?)', output_text)
        if score_match:
            predicted_score = float(score_match.group(1))

            # Ensure score is within valid range (0-10)
            predicted_score = max(0.0, min(10.0, predicted_score))
        else:
            predicted_score = 5.0  # Default to neutral

        # Calculate accuracy based on how close the prediction is to the expected score
        # Using 1 - (absolute difference / 10), so perfect match = 1.0, worst case = 0.0
        accuracy = 1.0 - (abs(predicted_score - expected_score) / 10.0)
        individual_scores.append(accuracy)
        total_score += accuracy

    except Exception as e:
        print(f"Error processing response '{output_text}': {e}")
        individual_scores.append(0.0)  # Score 0 for failed predictions
    return total_score
