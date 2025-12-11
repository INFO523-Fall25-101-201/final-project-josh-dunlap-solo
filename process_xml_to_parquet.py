import os
import sys
import re
import pandas as pd
from lxml import etree
from glob import glob
from pathlib import Path
import argparse

def clean_whitespace(s):
    if s is None:
        return ""
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

def extract_fields_from_xml(file_path):
    try:
        tree = etree.parse(file_path)
        root = tree.getroot()
        data = {
            'pub_date': None,
            'object_type': None,
            'raw_text': None
        }

        # Extract dates
        date_element = root.find('.//NumericPubDate')
        if date_element is not None and date_element.text:
            data['pub_date'] = date_element.text.strip()
        
        # extract object types
        object_types = root.findall('.//ObjectType')
        if object_types:
            unique_types = set(o.text.strip() for o in object_types if o.text)
            data['object_type'] = " | ".join(sorted(list(unique_types)))
        
        # extract article text
        text_element = root.find('.//FullText')
        if text_element is not None and text_element.text:
            raw_text = text_element.text
            data['raw_text'] = clean_whitespace(raw_text)

        # Check for mandatory fields before returning
        if not data['pub_date'] or not data['raw_text']:
            print(f"Warning: Skipping {file_path} due to missing date or text.", flush=True)
            return None
        
        return data

    except Exception as e:
        print(f"Error processing file {file_path}: {e}", flush=True)
        return None

def process_folder(xml_directory, output_parquet_file):
    """Process a single folder of XML files"""
    all_xml_files = glob(os.path.join(xml_directory, '*.xml'))
    print(f"Found {len(all_xml_files)} XML files to process in {xml_directory}", flush=True)
    
    if len(all_xml_files) == 0:
        print(f"No XML files found in {xml_directory}, skipping.", flush=True)
        return
    
    records = []
    
    for i, file_path in enumerate(all_xml_files):
        record = extract_fields_from_xml(file_path)
        if record:
            records.append(record)
        
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(all_xml_files)} files...", flush=True)

    if len(records) == 0:
        print(f"No valid records extracted from {xml_directory}", flush=True)
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(records)
    print(f"\nSuccessfully extracted data from {len(df)} records.", flush=True)
    
    # Write to Parquet with compression
    df.to_parquet(output_parquet_file, engine='pyarrow', index=False, compression='snappy')
    
    print(f"Data written successfully to {output_parquet_file}", flush=True)
    print(f"Final DataFrame Shape: {df.shape}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process XML files to Parquet')
    parser.add_argument('input_folder', type=str, help='Input folder containing XML files')
    parser.add_argument('output_file', type=str, help='Output Parquet file path')
    
    args = parser.parse_args()
    
    print(f"Starting processing at {pd.Timestamp.now()}", flush=True)
    print(f"Input folder: {args.input_folder}", flush=True)
    print(f"Output file: {args.output_file}", flush=True)
    
    process_folder(args.input_folder, args.output_file)
    
    print(f"Completed at {pd.Timestamp.now()}", flush=True)