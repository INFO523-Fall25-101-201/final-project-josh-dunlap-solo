import pandas as pd
import re
from gensim.models import Word2Vec
import multiprocessing
import argparse
import os
from datetime import datetime
import gc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

def simple_preprocess(text, min_word_length=3):
    """Minimal preprocessing for historical text"""
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [t for t in text.split() if len(t) >= min_word_length]
    
    return ' '.join(tokens)

def preprocess_parquet_chunked(parquet_file, text_column='raw_text', 
                               min_tokens=10):
    print("="*70)
    print("PREPROCESSING (ROW-GROUP SAFE MODE)")
    print("="*70)

    pf = pq.ParquetFile(parquet_file)
    total_rows = pf.metadata.num_rows
    num_row_groups = pf.num_row_groups

    print(f"Total articles: {total_rows:,}")
    print(f"Row groups: {num_row_groups}")

    processed_texts = []
    total_kept = 0
    total_removed = 0
    processed_count = 0

    for rg_index in range(num_row_groups):
        print(f"\nReading row group {rg_index+1}/{num_row_groups}...")
        rg_table = pf.read_row_group(rg_index, columns=[text_column])
        df_chunk = rg_table.to_pandas()

        for text in df_chunk[text_column]:
            processed = simple_preprocess(text)
            token_count = len(processed.split())
            if token_count >= min_tokens:
                processed_texts.append(processed)
                total_kept += 1
            else:
                total_removed += 1

        processed_count += len(df_chunk)
        print(f"  Processed {processed_count:,}/{total_rows:,}")

        del df_chunk, rg_table
        gc.collect()

    pct_removed = 100 * total_removed / total_rows if total_rows > 0 else 0
    print(f"\nFinal: Kept {total_kept:,}, Removed {total_removed:,} "
          f"({pct_removed:.1f}%)")

    return processed_texts

def prepare_sentences(processed_texts):
    """Convert list of texts to sentences for Word2Vec"""
    print("\n" + "="*70)
    print("PREPARING SENTENCES FOR WORD2VEC")
    print("="*70)
    
    sentences = []
    for text in processed_texts:
        if text:
            tokens = text.split()
            if tokens:
                sentences.append(tokens)
    
    total_tokens = sum(len(s) for s in sentences)
    print(f"Prepared {len(sentences):,} sentences")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Average tokens per sentence: {total_tokens/len(sentences):.1f}")
    
    return sentences

def train_word2vec_model(sentences, 
                         vector_size=300,
                         window=5,
                         min_count=15,
                         workers=None,
                         epochs=5,
                         sg=1,
                         negative=5,
                         sample=1e-3):
    """Train Word2Vec model"""
    if workers is None:
        workers = multiprocessing.cpu_count()
    
    print("\n" + "="*70)
    print("TRAINING WORD2VEC MODEL")
    print("="*70)
    print(f"Configuration:")
    print(f"  Vector size: {vector_size}")
    print(f"  Window: {window}")
    print(f"  Min count: {min_count}")
    print(f"  Workers: {workers}")
    print(f"  Epochs: {epochs}")
    print(f"  Algorithm: Skip-gram with negative sampling")
    
    start_time = datetime.now()
    print(f"\nTraining started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
        sg=sg,
        negative=negative,
        sample=sample,
        seed=42
    )
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"Training completed: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration}")
    print(f"\nFinal vocabulary size: {len(model.wv):,} words")
    
    return model

def analyze_and_save_model(model, output_dir, decade_name, test_words=None):
    """Analyze and save model"""
    print("\n" + "="*70)
    print("MODEL ANALYSIS AND EXPORT")
    print("="*70)
    
    if test_words is None:
        test_words = ["government", "war", "peace", "president", "nation"]
    
    # Get word frequencies
    word_counts = {}
    for word in model.wv.index_to_key:
        word_counts[word] = model.wv.get_vecattr(word, "count")
    
    most_common = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nVocabulary Statistics:")
    print(f"  Total words: {len(model.wv):,}")
    print(f"  Vector dimensions: {model.wv.vector_size}")
    print(f"\nTop 20 most frequent words:")
    for word, count in most_common[:20]:
        print(f"  {word:20s} {count:>10,}")
    
    # Test similarities
    print(f"\nWord Similarity Analysis:")
    for word in test_words:
        if word in model.wv:
            similar = model.wv.most_similar(word, topn=5)
            print(f"\n'{word}':")
            for sim_word, score in similar:
                print(f"  {sim_word:20s} {score:.4f}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, f"word2vec_{decade_name}.model")
    model.save(model_path)
    print(f"\n✓ Model saved to: {model_path}")
    
    # Save word vectors
    wv_path = os.path.join(output_dir, f"word_vectors_{decade_name}.kv")
    model.wv.save(wv_path)
    print(f"✓ Word vectors saved to: {wv_path}")
    
    # Save vocabulary CSV
    vocab_path = os.path.join(output_dir, f"vocabulary_{decade_name}.csv")
    vocab_df = pd.DataFrame([
        {'word': word, 'frequency': count}
        for word, count in most_common
    ])
    vocab_df.to_csv(vocab_path, index=False)
    print(f"✓ Vocabulary saved to: {vocab_path}")
    
    return model_path, wv_path, vocab_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Word2Vec on decade data')
    parser.add_argument('input_parquet', type=str)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--vector-size', type=int, default=300)
    parser.add_argument('--window', type=int, default=5)
    parser.add_argument('--min-count', type=int, default=15)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--workers', type=int, default=None)
    
    args = parser.parse_args()
    
    decade_name = os.path.basename(args.input_parquet).replace('.parquet', '').replace('nyt_', '')
    
    print("="*70)
    print(f"WORD2VEC TRAINING - {decade_name.upper()}")
    print("="*70)
    print(f"Input: {args.input_parquet}")
    print(f"Output: {args.output_dir}")
    print()
    
    # 1. Process parquet in chunks (memory efficient)
    processed_texts = preprocess_parquet_chunked(args.input_parquet)
    
    # 2. Prepare sentences
    sentences = prepare_sentences(processed_texts)
    
    # Clear processed_texts from memory
    del processed_texts
    gc.collect()
    
    # 3. Train model
    model = train_word2vec_model(
        sentences,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        workers=args.workers,
        epochs=args.epochs
    )
    
    # 4. Analyze and save
    model_path, wv_path, vocab_path = analyze_and_save_model(
        model, args.output_dir, decade_name
    )
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Vectors: {wv_path}")
    print(f"Vocabulary: {vocab_path}")