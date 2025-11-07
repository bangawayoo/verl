# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess JSONL files to parquet format
"""

import argparse
import json
import os
from pathlib import Path

import datasets
from datasets import Dataset

try:
    from verl.utils.hdfs_io import copy, makedirs
    HAS_VERL = True
except ImportError:
    HAS_VERL = False
    print("Warning: verl module not available. HDFS functionality will be disabled.")


def load_jsonl_files(directory_path):
    """Load all JSONL files from the specified directory."""
    data = []
    jsonl_files = list(Path(directory_path).glob("*.jsonl"))
    
    print(f"Found {len(jsonl_files)} JSONL files in {directory_path}")
    
    for jsonl_file in jsonl_files:
        print(f"Processing {jsonl_file.name}...")
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num} in {jsonl_file.name}: {e}")
                    continue
    
    print(f"Loaded {len(data)} entries total")
    return data


def extract_relevant_fields(entry):
    """Extract relevant fields from a JSONL entry."""
    # Common fields across different formats
    processed = {
        "uid": entry.get("uid", ""),
        "data_source": entry.get("source_metadata", {}).get("generated_file", "jsonl"),
    }
    
    # Extract message/conversation data
    if "reduced_message" in entry:
        processed["messages"] = entry["reduced_message"]
    else:
        AssertionError("reduced_message not found")
    # Extract metadata
    metadata = {}
    
    # From source_metadata
    if "source_metadata" in entry:
        metadata.update(entry["source_metadata"])
    
    # From metadata field
    if "metadata" in entry:
        metadata.update(entry["metadata"])
    
    # From teacher_model
    if "teacher_model" in entry:
        metadata["teacher_model"] = entry["teacher_model"]
    
    # From adversarial_metadata
    if "adversarial_metadata" in entry:
        metadata["adversarial_metadata"] = entry["adversarial_metadata"]
    
    processed["metadata"] = metadata
    
    # Extract any generated questions if present
    if "generated_questions" in entry:
        processed["generated_questions"] = entry["generated_questions"]
    
    # Extract additional system messages if present
    if "additional_system_message" in entry:
        processed["additional_system_messages"] = entry["additional_system_message"]
    
    return processed


def create_prompt_format(messages):
    """Convert messages to a prompt format similar to GSM8K processing."""
    if not messages:
        return []
    
    # If messages is already in the correct format, return as is
    if isinstance(messages, list) and all(isinstance(m, dict) and "role" in m and "content" in m for m in messages):
        return messages[:2]
    else:
        AssertionError("Messages are not in the correct format")


def process_data(data, split="train"):
    """Process the raw data into the desired format."""
    processed_data = []
    
    for idx, entry in enumerate(data):
        try:
            extracted = extract_relevant_fields(entry)
            
            # Create the final processed entry
            processed_entry = {
                "data_source": extracted.get("data_source", "jsonl"),
                "prompt": create_prompt_format(extracted.get("messages", [])),
                "ability": "conversation",  # Default ability type
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "uid": extracted.get("uid", ""),
                }
            }
            
            # Only add metadata if it's not empty
            metadata = extracted.get("metadata", {})
            if metadata:
                processed_entry["extra_info"]["metadata"] = metadata
            
            # Add reward model info if needed (placeholder for now)
            processed_entry["reward_model"] = {
                "style": "none",
                "ground_truth": None
            }
            
            processed_data.append(processed_entry)
            
        except Exception as e:
            print(f"Error processing entry {idx}: {e}")
            continue
    
    return processed_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess JSONL files to parquet format")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="~/kiyoon/pubg-cpc/workflows/data/ko/teacher_proprietary/train",
        help="Directory containing JSONL files to process"
    )
    parser.add_argument(
        "--local_save_dir",
        type=str,
        default="~/data/jsonl_processed",
        help="The save directory for the preprocessed dataset"
    )
    parser.add_argument(
        "--hdfs_dir",
        type=str,
        default=None,
        help="HDFS directory to copy the processed data to"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test", "validation"],
        help="Data split type"
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum number of files to process (for testing)"
    )
    
    args = parser.parse_args()
    
    # Expand paths
    input_dir = os.path.expanduser(args.input_dir)
    local_save_dir = os.path.expanduser(args.local_save_dir)
    
    # Create output directory
    os.makedirs(local_save_dir, exist_ok=True)
    
    # Load and process data
    print(f"Loading JSONL files from {input_dir}...")
    raw_data = load_jsonl_files(input_dir)
    
    if args.max_files and len(raw_data) > args.max_files:
        print(f"Limiting to {args.max_files} entries for testing...")
        raw_data = raw_data[:args.max_files]
    
    print(f"Processing {len(raw_data)} entries...")
    processed_data = process_data(raw_data, split=args.split)
    
    # Create dataset and save to parquet
    print(f"Creating dataset from {len(processed_data)} processed entries...")
    dataset = Dataset.from_list(processed_data)
    
    output_file = os.path.join(local_save_dir, f"{args.split}.parquet")
    print(f"Saving to {output_file}...")
    dataset.to_parquet(output_file)
    
    print(f"Successfully saved {len(processed_data)} entries to {output_file}")
    
    # Copy to HDFS if specified
    if args.hdfs_dir:
        if HAS_VERL:
            print(f"Copying to HDFS at {args.hdfs_dir}...")
            makedirs(args.hdfs_dir)
            copy(src=local_save_dir, dst=args.hdfs_dir)
            print("HDFS copy completed")
        else:
            print("Warning: HDFS copy skipped because verl module is not available")
    
    print("Processing complete!")
