import os
import sys
import requests
import json
import pandas as pd
import time
from tqdm import tqdm
import random

# Base URL for NMR prediction API
base_url = "https://nmr-prediction.service.zakodium.com/v1/predict/carbon"

def process_smiles(smiles, idx, source="unknown"):
    """
    Process a single SMILES string to get NMR prediction
    """
    # Request headers
    headers = {
        "accept": "*/*",
        "Content-Type": "application/json"
    }

    # Request payload for NMR prediction
    data = {
        "molfile": None,  # No molfile uploaded
        "smiles": smiles,  # SMILES string for prediction
        "includeJDX": False,  # Don't include JDX format
        "frequency": 100,
        "nbPoints": 65536,
        "lineWidth": 1,
        "from": -10,
        "to": 230,
        "maxSphereSize": 5
    }
    
    # Retry mechanism with maximum 10 attempts
    count = 0
    while count < 10:
        count += 1
        
        try:
            response = requests.post(base_url, headers=headers, json=data)
            
            # Check if request was successful
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Successfully processed SMILES from {source} at index {idx}")
                return result, idx
            elif response.status_code == 429:
                # Rate limited - wait longer
                wait_time = 5 + random.uniform(0, 3)  # 5-8 seconds random wait
                print(f"⏰ Rate limited for {source} index {idx}, waiting {wait_time:.1f}s")
                time.sleep(wait_time)
            else:
                print(f"❌ Request failed for {source} index {idx}, status code: {response.status_code}")
                print(f"Error message: {response.text}")
                time.sleep(2)
        except Exception as e:
            print(f"Exception occurred for {source} index {idx}: {str(e)}")
            time.sleep(2)
            pass
    
    print(f"❌ Failed to process SMILES from {source} at index {idx} after 10 attempts")
    return None, idx


def main():
    """
    Main function to process SMILES from both files and get NMR predictions
    """
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define file paths
    csv_file = "/home/lingjiebao/project/NMR_Core_Mol_Detection/data/test_core_mol.csv"
    json_file = "/home/lingjiebao/project/NMR_Core_Mol_Detection/data/data_valid_test_1020/ex_test.json"
    output_file = os.path.join(script_dir, "nmr_predictions_combined.json")
    
    # Check for existing results to resume
    existing_results = []
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                existing_results = json.load(f)
            print(f"📂 Found existing results with {len(existing_results)} entries")
        except:
            existing_results = []
    
    # Storage for results
    all_results = existing_results.copy()
    
    # Track processed indices
    processed_csv_indices = {r['index'] for r in all_results if r['source'] == 'CSV'}
    processed_json_indices = {r['index'] for r in all_results if r['source'] == 'JSON'}
    
    print(f"📊 Previously processed CSV indices: {len(processed_csv_indices)}")
    print(f"📊 Previously processed JSON indices: {len(processed_json_indices)}")
    
    # Process CSV file
    if os.path.exists(csv_file):
        print(f"📂 Reading CSV file: {csv_file}")
        try:
            df_csv = pd.read_csv(csv_file)
            print(f"📊 Loaded {len(df_csv)} molecules from CSV")
            
            # Process each SMILES from CSV (skip already processed)
            for idx, row in df_csv.iterrows():
                if idx in processed_csv_indices:
                    print(f"⏭️  Skipping CSV SMILES {idx+1}/{len(df_csv)} (already processed)")
                    continue
                    
                smiles = row['SMILES']  # SMILES column
                print(f"Processing CSV SMILES {idx+1}/{len(df_csv)}: {smiles}")
                
                result, result_idx = process_smiles(smiles, idx, "CSV")
                
                # Store result
                result_data = {
                    "source": "CSV",
                    "index": idx,
                    "smiles": smiles,
                    "nmr_result": result,
                    "status": "success" if result is not None else "failed"
                }
                all_results.append(result_data)
                
                # Save results periodically
                if len(all_results) % 10 == 0:
                    with open(output_file, 'w') as f:
                        json.dump(all_results, f, indent=2)
                    print(f"💾 Saved {len(all_results)} results to {output_file}")
                
                # Rate limiting - wait 2-3 seconds between requests (randomized)
                wait_time = 2 + random.uniform(0, 1)
                time.sleep(wait_time)
                
        except Exception as e:
            print(f"❌ Error reading CSV file: {str(e)}")
    else:
        print(f"❌ CSV file not found: {csv_file}")
    
    # Process JSON file
    if os.path.exists(json_file):
        print(f"📂 Reading JSON file: {json_file}")
        try:
            with open(json_file, 'r') as f:
                data_json = json.load(f)
            print(f"📊 Loaded {len(data_json)} molecules from JSON")
            
            # Process each cano_smiles from JSON (skip already processed)
            for idx, item in enumerate(data_json):
                if idx in processed_json_indices:
                    if idx % 100 == 0:
                        print(f"⏭️  Skipping JSON SMILES {idx+1}/{len(data_json)} (already processed)")
                    continue
                    
                smiles = item['cano_smiles']
                if idx % 50 == 0:
                    print(f"Processing JSON SMILES {idx+1}/{len(data_json)}: {smiles}")
                
                result, result_idx = process_smiles(smiles, idx, "JSON")
                
                # Store result
                result_data = {
                    "source": "JSON",
                    "index": idx,
                    "smiles": smiles,
                    "original_key": item.get('key', idx),
                    "nmr_result": result,
                    "status": "success" if result is not None else "failed"
                }
                all_results.append(result_data)
                
                # Save results periodically
                if len(all_results) % 10 == 0:
                    with open(output_file, 'w') as f:
                        json.dump(all_results, f, indent=2)
                    if idx % 50 == 0:
                        print(f"💾 Saved {len(all_results)} results to {output_file}")
                
                # Rate limiting - wait 2-3 seconds between requests (randomized)
                wait_time = 2 + random.uniform(0, 1)
                time.sleep(wait_time)
                
        except Exception as e:
            print(f"❌ Error reading JSON file: {str(e)}")
    else:
        print(f"❌ JSON file not found: {json_file}")
    
    # Save final results
    try:
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"💾 Final results saved to: {output_file}")
        
        # Print summary statistics
        successful_predictions = sum(1 for r in all_results if r['status'] == 'success')
        total_predictions = len(all_results)
        success_rate = (successful_predictions / total_predictions) * 100 if total_predictions > 0 else 0
        
        print(f"\n📊 Summary:")
        print(f"   Total molecules processed: {total_predictions}")
        print(f"   Successful predictions: {successful_predictions}")
        print(f"   Failed predictions: {total_predictions - successful_predictions}")
        print(f"   Success rate: {success_rate:.2f}%")
        
        # Breakdown by source
        csv_results = [r for r in all_results if r['source'] == 'CSV']
        json_results = [r for r in all_results if r['source'] == 'JSON']
        
        print(f"\n📈 CSV Results:")
        print(f"   Total: {len(csv_results)}")
        print(f"   Successful: {sum(1 for r in csv_results if r['status'] == 'success')}")
        
        print(f"\n📈 JSON Results:")
        print(f"   Total: {len(json_results)}")
        print(f"   Successful: {sum(1 for r in json_results if r['status'] == 'success')}")
        
    except Exception as e:
        print(f"❌ Error saving results: {str(e)}")


if __name__ == "__main__":
    main()