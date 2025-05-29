import os
import time
import sys
from openai import OpenAI, APIError, APIConnectionError, RateLimitError
from dotenv import load_dotenv

# --- Configuration ---
TRAIN_FILE_PATH = "medical_train.jsonl"
VALID_FILE_PATH = "medical_valid.jsonl"  # Optional, can be None
BASE_MODEL = "gpt-3.5-turbo-0125"  # Recommended model for fine-tuning
MODEL_SUFFIX = "med-cot-v1"  # Max 18 chars, helps identify your model
# --- End Configuration ---


def load_api_key():
    """Loads OpenAI API key from .env file."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in .env file or environment variables.")
        print(
            "Please create a .env file with your API key (e.g., OPENAI_API_KEY='sk-...')"
        )
        sys.exit(1)
    return api_key


def upload_file_to_openai(client, file_path, purpose="fine-tune"):
    """Uploads a file to OpenAI and returns its ID."""
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        print(f"File not found or is empty: {file_path}. Skipping upload.")
        return None
    try:
        with open(file_path, "rb") as f:
            response = client.files.create(file=f, purpose=purpose)
        print(f"Successfully uploaded {file_path}. File ID: {response.id}")
        return response.id
    except APIError as e:
        print(f"OpenAI API Error during upload of {file_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during upload of {file_path}: {e}")
    return None


def create_fine_tuning_job(client, training_file_id, validation_file_id=None):
    """Creates a fine-tuning job."""
    if not training_file_id:
        print("Error: Training file ID is required to create a fine-tuning job.")
        return None
    try:
        params = {
            "training_file": training_file_id,
            "model": BASE_MODEL,
            "suffix": MODEL_SUFFIX,
        }
        if validation_file_id:
            params["validation_file"] = validation_file_id

        # For gpt-3.5-turbo, hyperparameters like n_epochs are often automatically determined.
        # If you were using older models like babbage-002, you might add:
        # "hyperparameters": {"n_epochs": 3}

        job_response = client.fine_tuning.jobs.create(**params)
        print(
            f"Fine-tuning job created successfully. Job ID: {job_response.id}, Status: {job_response.status}"
        )
        return job_response.id
    except APIError as e:
        print(f"OpenAI API Error during job creation: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during job creation: {e}")
    return None


def monitor_job_status(client, job_id):
    """Monitors the fine-tuning job and returns the fine-tuned model ID if successful."""
    if not job_id:
        return None
    print(f"\nMonitoring fine-tuning job: {job_id}")
    try:
        while True:
            job_status_response = client.fine_tuning.jobs.retrieve(job_id)
            status = job_status_response.status
            print(
                f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}, Job Status: {status}"
            )

            if status == "succeeded":
                fine_tuned_model_id = job_status_response.fine_tuned_model
                print(f"\nJob finished successfully!")
                print(f"Fine-tuned model ID: {fine_tuned_model_id}")
                return fine_tuned_model_id
            elif status in ["failed", "cancelled"]:
                print(f"\nJob {status}.")
                if job_status_response.error:
                    print(
                        f"Error details: Code: {job_status_response.error.code}, Message: {job_status_response.error.message}, Param: {job_status_response.error.param}"
                    )
                return None

            # Optional: List recent events for more details
            # try:
            #     events_response = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=1)
            #     for event in events_response.data:
            #         print(f"  Latest event: {event.message}")
            # except Exception as event_e:
            #     print(f"  Could not fetch job events: {event_e}")

            time.sleep(60)  # Check every 60 seconds

    except APIConnectionError:
        print("Network error. Retrying in 60 seconds...")
        time.sleep(60)
    except RateLimitError:
        print("Rate limit hit. Retrying in 60 seconds...")
        time.sleep(60)
    except APIError as e:
        print(f"OpenAI API Error during job monitoring: {e}")
    except KeyboardInterrupt:
        print(
            "\nMonitoring interrupted by user. To resume, you can manually check the job status on OpenAI's platform."
        )
        print(f"Job ID was: {job_id}")
    except Exception as e:
        print(f"An unexpected error occurred during job monitoring: {e}")
    return None


def test_fine_tuned_model(client, model_id, sample_question):
    """Sends a sample question to the fine-tuned model."""
    if not model_id:
        print("No fine-tuned model ID available to test.")
        return

    print(f"\nTesting fine-tuned model: {model_id}")
    # ★ Prompt Design: Use the same system prompt structure as during training for consistency.
    system_prompt_for_inference = (
        "You are an expert medical AI assistant. Given a question, first provide a "
        "detailed chain of thought (Complex_CoT) explaining your reasoning step-by-step. "
        "After the chain of thought, conclude with a concise final Response."
    )
    try:
        completion_response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt_for_inference},
                {"role": "user", "content": sample_question},
            ],
            temperature=0.7,  # Adjust for creativity/determinism
            max_tokens=1000,  # Adjust as needed for expected CoT + Response length
        )
        print("\n--- Sample Model Response ---")
        print(f"Question: {sample_question}")
        print(f"Answer:\n{completion_response.choices[0].message.content}")
        print("--- End Sample Model Response ---")
    except APIError as e:
        print(f"OpenAI API Error during model test: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during model test: {e}")


def main():
    """Main function to orchestrate the fine-tuning process."""
    api_key = load_api_key()
    client = OpenAI(api_key=api_key)

    print("Step 1: Uploading data files...")
    training_file_id = upload_file_to_openai(client, TRAIN_FILE_PATH)
    if not training_file_id:
        print("Failed to upload training file. Exiting.")
        sys.exit(1)

    validation_file_id = None
    if os.path.exists(VALID_FILE_PATH) and os.path.getsize(VALID_FILE_PATH) > 0:
        validation_file_id = upload_file_to_openai(client, VALID_FILE_PATH)
        if not validation_file_id:
            print("Warning: Failed to upload validation file. Proceeding without it.")
    else:
        print(
            f"Validation file '{VALID_FILE_PATH}' not found or empty. Proceeding without validation data."
        )

    print("\nStep 2: Creating fine-tuning job...")
    job_id = create_fine_tuning_job(client, training_file_id, validation_file_id)
    if not job_id:
        print("Failed to create fine-tuning job. Exiting.")
        sys.exit(1)

    print("\nStep 3: Monitoring job status...")
    fine_tuned_model_id = monitor_job_status(client, job_id)

    if fine_tuned_model_id:
        print(
            f"\nFine-tuning process completed. Your model ID is: {fine_tuned_model_id}"
        )
        # You can save this model ID for later use.
        with open("fine_tuned_model_id.txt", "w") as f:
            f.write(fine_tuned_model_id)
        print(f"Model ID saved to fine_tuned_model_id.txt")

        # Optional: Test the model
        sample_question = "What are the common symptoms of influenza and how do they differ from a common cold?"
        test_fine_tuned_model(client, fine_tuned_model_id, sample_question)
    else:
        print("\nFine-tuning job did not succeed or was interrupted.")


if __name__ == "__main__":
    # Before running, ensure you have python-dotenv and openai packages installed:
    # pip install python-dotenv openai
    main()
