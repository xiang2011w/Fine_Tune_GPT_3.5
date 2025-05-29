import json
import random
import os


def convert_to_jsonl(
    input_file_path,
    output_train_file_path,
    output_valid_file_path,
    train_split_ratio=0.9,
    min_valid_samples=5,  # Minimum samples for validation if data is scarce
):
    """
    Converts the input JSON array to JSONL format suitable for OpenAI fine-tuning,
    splitting into training and validation sets.
    """
    if not os.path.exists(input_file_path):
        print(f"Error: Input file not found at {input_file_path}")
        return False

    try:
        with open(input_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {input_file_path}: {e}")
        return False
    except Exception as e:
        print(f"Error reading {input_file_path}: {e}")
        return False

    if not isinstance(data, list):
        print(
            f"Error: Expected a list of objects in {input_file_path}, got {type(data)}"
        )
        return False
    if not data:
        print(f"Error: Input file {input_file_path} is empty.")
        return False

    # ★ Prompt Design: This system prompt clearly instructs the model on the desired output format,
    # including the chain-of-thought and the final response. It's crucial for guiding the
    # fine-tuned model's behavior.
    system_prompt = (
        "You are an expert medical AI assistant. Given a question, first provide a "
        "detailed chain of thought (Complex_CoT) explaining your reasoning step-by-step. "
        "After the chain of thought, conclude with a concise final Response."
    )

    formatted_data = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            print(
                f"Warning: Skipping item at index {i} as it's not a dictionary: {item}"
            )
            continue

        question = item.get("Question")
        complex_cot = item.get("Complex_CoT")
        response = item.get("Response")

        if question and complex_cot and response:
            assistant_content = f"{complex_cot}\n\n{response}"
            formatted_data.append(
                {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": assistant_content},
                    ]
                }
            )
        else:
            print(f"Warning: Skipping item at index {i} due to missing fields: {item}")

    if not formatted_data:
        print(
            "Error: No valid data entries found after formatting. Cannot create training/validation files."
        )
        return False

    random.shuffle(formatted_data)  # Shuffle for better split

    total_samples = len(formatted_data)
    split_index = int(total_samples * train_split_ratio)

    train_data = formatted_data[:split_index]
    valid_data = formatted_data[split_index:]

    # Ensure validation set has a minimum number of samples if possible,
    # or is empty if the dataset is too small.
    if total_samples <= min_valid_samples * 2:  # If dataset is very small
        print(
            f"Warning: Dataset size ({total_samples}) is small. Using all for training and none for validation."
        )
        train_data = formatted_data
        valid_data = []
    elif len(valid_data) < min_valid_samples and len(train_data) > min_valid_samples:
        # If validation is too small but train has enough, move some over
        needed_for_valid = min_valid_samples - len(valid_data)
        can_move = min(
            needed_for_valid, len(train_data) - 1
        )  # Keep at least 1 for train
        if can_move > 0:
            valid_data.extend(train_data[-can_move:])
            train_data = train_data[:-can_move]
            print(
                f"Adjusted split: Moved {can_move} samples to validation for a minimum of {min_valid_samples}."
            )

    if not train_data:
        print("Error: No training data after split. Check data and split ratio.")
        return False

    try:
        with open(output_train_file_path, "w", encoding="utf-8") as f_train:
            for entry in train_data:
                f_train.write(json.dumps(entry) + "\n")
        print(
            f"Training data written to {output_train_file_path} with {len(train_data)} examples."
        )

        if valid_data:
            with open(output_valid_file_path, "w", encoding="utf-8") as f_valid:
                for entry in valid_data:
                    f_valid.write(json.dumps(entry) + "\n")
            print(
                f"Validation data written to {output_valid_file_path} with {len(valid_data)} examples."
            )
        else:
            # Delete validation file if it exists and is now empty
            if os.path.exists(output_valid_file_path):
                os.remove(output_valid_file_path)
            print(
                "No validation data generated (dataset might be too small or split resulted in empty validation)."
            )
        return True
    except IOError as e:
        print(f"Error writing output files: {e}")
        return False


if __name__ == "__main__":
    INPUT_JSON_FILE = "medical_o1_sft_mix.json"
    TRAIN_JSONL_FILE = "medical_train.jsonl"
    VALID_JSONL_FILE = "medical_valid.jsonl"

    print(f"Starting data preparation from {INPUT_JSON_FILE}...")
    success = convert_to_jsonl(INPUT_JSON_FILE, TRAIN_JSONL_FILE, VALID_JSONL_FILE)
    if success:
        print("Data preparation completed successfully.")
    else:
        print("Data preparation failed.")
