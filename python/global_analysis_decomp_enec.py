import pandas as pd
import os
from pathlib import Path

def calculate_global_decompression_metrics(csv_path, output_txt_path):
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found.")
        return False

    df = pd.read_csv(csv_path)

    # 1. Filter: Only keep rows where 'OP Type' contains 'decomp'
    df_decomp = df[df['OP Type'].str.contains('decomp', case=False, na=False)].copy()

    if df_decomp.empty:
        print(f"No 'decomp' related operator data found in CSV: {csv_path}")
        return False

    # --- Core Logic: Adjusted Time Calculation ---
    # Formula: (Avg * 4 - Max) / 3, aimed at removing the maximum outlier from 4 test runs
    df_decomp['adjusted_time_us'] = (df_decomp['Avg Time(us)'] * 4 - df_decomp['Max Time(us)']) / 3
    
    # Calculate total adjusted execution time (seconds)
    total_avg_time_s = df_decomp['adjusted_time_us'].sum() / 1e6
    # --------------------------------

    total_output_size_mb = df_decomp['datasize_MB'].sum()
    total_output_size_gb = total_output_size_mb / 1024.0
    
    # Global Decompression Throughput = Total Output Data Size / Adjusted Total Time
    global_decomp_throughput = total_output_size_gb / total_avg_time_s if total_avg_time_s != 0 else 0

    output_lines = [
        "=" * 45,
        " Model Global Decompression Summary (Adj. Time) ",
        "-" * 45,
        f" Total Layers Decompressed : {len(df_decomp)}",
        f" Total Output Data Size    : {total_output_size_gb:.3f} GB",
        f" Total Execution Time (Adj): {total_avg_time_s:.4f} seconds",
        f" Global Decomp Throughput  : {global_decomp_throughput:.2f} GB/s",
        "=" * 45
    ]
    output_text = "\n".join(output_lines)

    print(f"\nProcessing file: {csv_path}")
    print(output_text)

    try:
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(output_text)
        print(f"[INFO] Adjusted decompression analysis saved to: {output_txt_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save file: {e}")
        return False
    return True

def main():
    results_root = './results_enec'
    if not os.path.isdir(results_root):
        print(f"Results root directory not found: {results_root}")
        return

    for dtype in os.listdir(results_root):
        dtype_path = os.path.join(results_root, dtype)
        if not os.path.isdir(dtype_path):
            continue
        for model_name in os.listdir(dtype_path):
            model_path = os.path.join(dtype_path, model_name)
            if not os.path.isdir(model_path):
                continue
            
            # Look for decompression result CSV
            csv_file = os.path.join(model_path, f"{model_name}_decompress.csv")
            if os.path.exists(csv_file):
                output_txt = os.path.join(model_path, f"{model_name}_decompress_summary.txt")
                calculate_global_decompression_metrics(csv_file, output_txt)
            else:
                print(f"Skipping {model_path}, could not find {model_name}_decompress.csv")

if __name__ == '__main__':
    main()