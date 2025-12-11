import os
import pandas as pd
from pathlib import Path
from collections import defaultdict
import argparse

def extract_decade(pub_date_str):
    """
    Extract decade from pub_date string (YYYYMMDD format)
    Returns decade as string like '1850s', '1860s', etc.
    """
    try:
        year = int(pub_date_str[:4])
        decade_start = (year // 10) * 10
        return f"{decade_start}s"
    except (ValueError, TypeError):
        return None

def process_parquet_files(input_dir, output_dir):
    """
    Read all parquet files, group by decade, and save decade-specific files
    """
    print(f"Processing parquet files from: {input_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Find all parquet files
    parquet_files = list(Path(input_dir).glob("*.parquet"))
    print(f"Found {len(parquet_files)} parquet files")
    print()
    
    # Dictionary to hold DataFrames by decade
    decade_dfs = defaultdict(list)
    
    # Statistics tracking
    total_articles = 0
    skipped_articles = 0
    
    # Process each parquet file
    for i, parquet_file in enumerate(parquet_files, 1):
        if i % 50 == 0:
            print(f"Processing file {i}/{len(parquet_files)}...")
        
        try:
            df = pd.read_parquet(parquet_file)
            
            # Add decade column
            df['decade'] = df['pub_date'].apply(extract_decade)
            
            # Track statistics
            total_articles += len(df)
            skipped = df['decade'].isna().sum()
            skipped_articles += skipped
            
            # Group by decade
            for decade in df['decade'].dropna().unique():
                decade_df = df[df['decade'] == decade].copy()
                # Drop the decade column before storing
                decade_df = decade_df.drop(columns=['decade'])
                decade_dfs[decade].append(decade_df)
                
        except Exception as e:
            print(f"Error processing {parquet_file}: {e}")
            continue
    
    print(f"\nProcessed {total_articles:,} total articles")
    print(f"Skipped {skipped_articles:,} articles with invalid dates")
    print()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Merge and save each decade
    print("=" * 60)
    print("DECADE STATISTICS AND EXPORT")
    print("=" * 60)
    
    decade_stats = []
    
    for decade in sorted(decade_dfs.keys()):
        # Concatenate all DataFrames for this decade
        decade_data = pd.concat(decade_dfs[decade], ignore_index=True)
        
        # Calculate statistics
        num_articles = len(decade_data)
        avg_text_length = decade_data['raw_text'].str.len().mean()
        date_range = (decade_data['pub_date'].min(), decade_data['pub_date'].max())
        
        # Save to parquet
        output_file = os.path.join(output_dir, f"nyt_{decade}.parquet")
        decade_data.to_parquet(output_file, engine='pyarrow', index=False, compression='snappy')
        
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        
        print(f"\n{decade}:")
        print(f"  Articles: {num_articles:,}")
        print(f"  Date range: {date_range[0]} to {date_range[1]}")
        print(f"  Avg text length: {avg_text_length:.0f} characters")
        print(f"  File size: {file_size_mb:.2f} MB")
        print(f"  Saved to: {output_file}")
        
        decade_stats.append({
            'decade': decade,
            'num_articles': num_articles,
            'date_range_start': date_range[0],
            'date_range_end': date_range[1],
            'avg_text_length': avg_text_length,
            'file_size_mb': file_size_mb
        })
    
    # Save summary statistics
    stats_df = pd.DataFrame(decade_stats)
    stats_file = os.path.join(output_dir, "decade_statistics.csv")
    stats_df.to_csv(stats_file, index=False)
    print(f"\n{' ' * 0}Summary statistics saved to: {stats_file}")
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    
    return stats_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge parquet files by decade')
    parser.add_argument('--input-dir', type=str, 
                       default='/xdisk/cjgomez/joshdunlapc/parquet_output',
                       help='Directory containing parquet files')
    parser.add_argument('--output-dir', type=str,
                       default='/xdisk/cjgomez/joshdunlapc/parquet_by_decade',
                       help='Directory for decade-specific parquet files')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("NYT PARQUET FILES - DECADE GROUPING")
    print("=" * 60)
    print()
    
    stats = process_parquet_files(args.input_dir, args.output_dir)
    
    print("\nFinal Summary:")
    print(stats.to_string(index=False))