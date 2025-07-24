import re
import time
import json
from textwrap import dedent
from agno.agent import Agent, RunResponse  # noqa
from agno.models.lmstudio import LMStudio
from pydantic import BaseModel, Field

TASK_MODEL_NAME = "mistralai/mistral-small-3.2"
TASK_MODEL_URL = "http://192.168.178.90:1234/v1"
TASK_MODEL_API_KEY = "weDontNeetItOnLocal"
MAX_RETRIES = 3  # Number of retries for LLM calls

class Rating(BaseModel):
    sentiment_score: float = Field(..., description="The score given by the LLM, ranging from 0 to 10.")
    reasoning: str = Field(..., description="A brief explanation of the score given by the LLM.")

def evaluate_single_example(example, individual_scores, prompt,total_score):
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

    test_agent = Agent(model=LMStudio(id=TASK_MODEL_NAME,
                                    base_url=TASK_MODEL_URL),
                  markdown=True,
                  system_message=formatpromt,
                    response_model=Rating)

    # Call the LLM with retry logic
    max_retries = MAX_RETRIES
    for attempt in range(max_retries):
        try:
            runResponse: RunResponse  = test_agent.run(prompt.format(input_text=input_text))
            rating_object = runResponse.content
            # response = test_model.chat.completions.create(
            #     model=TASK_MODEL_NAME,
            #     messages=messages
            # )
            break
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to get response after {max_retries} attempts: {e}")
                raise e
            time.sleep(1)  # Brief pause before retry

    # Extract numerical score from the response
    try:

        rating_object = runResponse.content
        # parse the output as JSON
        # rating_object = json.loads(output_text)
        predicted_score = rating_object.sentiment_score # Default to neutral score if not provided

        # Ensure score is within valid range (0-10)
        predicted_score = max(0.0, min(10.0, predicted_score))
         # Calculate accuracy based on how close the prediction is to the expected score
        # Using 1 - (absolute difference / 10), so perfect match = 1.0, worst case = 0.0
        accuracy = 1.0 - (abs(predicted_score - expected_score) / 10.0)
        individual_scores.append(accuracy)
        total_score += accuracy

    except Exception as e:
        print(f"Error processing response '{runResponse}': {e}")
        individual_scores.append(0.0)  # Score 0 for failed predictions
    return total_score
