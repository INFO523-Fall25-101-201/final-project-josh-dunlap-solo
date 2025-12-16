# project-final

Final project repo for INFO 523 - Summer 2025. I couldn't get quarto to run after importing my files from my research repository, so I am including what would be on my webpage in this README.

## Abstract

This project builds decade-by-decade static word embedding models on historical NYT data and finds a temporality dimension within these models. In this sense, the project engages the question of temporality in two ways. First, it demonstrates that static word embedding models trained on historical corpora embed not just linguistic and syntactic information, but also historical information; the static embedding of the 1880s differs meaningfully from the static embedding of the 1930s, in part due to the social, cultural, political, and technological changes of the intervening decades. Second, by finding a dimension of temporality within each static word embedding model through averaging the distance between temporal antonym pairs (e.g. future/past) the project also allows for the mapping of language along the continuum of temporality _from the historical perspective of a given decade_.

## Motivation/Broader Context

My higher level research goals are in the sub-field of Natural Language Processing and Computational Linguistics called “Lexical Semantic Change,” “Semantic Change Detection,” or “Diachronic Linguistics.” The idea is studying how language use changes over time. This happens both as a natural consequence of the fact that languages are always evolving, but also for social/cultural/technological reasons (which I'm more interested in). I'm working on the early stages of a research project right now where I'm trying to demonstrate that in the early 20th century the cultural perception of the idea of "utopia" was that it was something very futuristic, but my suspicion is that this has changed over time and now "utopia" is a pastoral, agrarian—even backward-looking—concept. I'm pursuing this from a number of different angles: literary analysis of utopian fiction from different times, social theory about the shifting cultural perception of utopianism, but also through Natural Language Processing. One thing I know I'll need is a way to analyze “utopia” and related concepts at distinct points in time. In order to ask, for example: “what are the differences between a word embedding for the word ‘utopia’ as it was used in the 1960s, as opposed to how it was used in the 2010s?”

## Data

The dataset that I ended up using for this project is the ProQuest Historical Newspapers Collections to which I have access through the U of A library—specifically The New York Times (1851–1936). I had originally intended to use the Corpus of Historical American English (COHA), since it is explicitly designed for diachronic linguistic analysis, is balanced across genres, and has reliable decade-level metadata, but, unfortunately, we didn’t have full paid access through the U of A library. Thankfully, there were meaningful benefits to using the NYT dataset as well. 

1. I got valuable experience preparing a dataset to be used for LSC research, a task I expect to repeat during my studies. The data were delivered as a giant list of unsorted, twenty-five thousand article apiece XML files, 5.6 million articles in total. In the preparation phases, I transformed these into decade size parquet files and then batch preprocessed them before training word2vec models on decade-by-decade data.
2. To my knowledge, in direct contrast to COHA, no one has ever used this dataset for LSC research before. While there are some issues with the data quality (see Error Analysis section), my ultimate hope will be to make both my models and my approach public so that other researchers can pursue projects on this dataset.

The dataset can be accessed here: https://arizona.figshare.com/articles/dataset/ProQuest_Historical_Newspapers_Collections_The_New_York_Times_1851-1936_and_The_Washington_Post_1877-1934_/17003149

## Reproducibility

### Environment Setup

The project uses micromamba and python 3.10.

Create the environment from the provided .yml file and active it as follows:

```
module load micromamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate xml-processing
```

### XML to Parquet Conversion
The dataset delivers the data in 338 folders, each with 25k files in them. For processing efficiency, I convert these XML files to Parquet. Make a list of the unzipped xml folders (folders.txt) and run process_xml_to_parquet.py on them as follows:

```
# Read the folder path for this array task
FOLDERS_FILE="${PROJECT_DIR}/folders.txt"
INPUT_FOLDER=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $FOLDERS_FILE)

# Define output directory and filename
OUTPUT_DIR="/file/path/here"
mkdir -p $OUTPUT_DIR

# Extract folder name for output file
FOLDER_NAME=$(basename $INPUT_FOLDER)
OUTPUT_FILE="${OUTPUT_DIR}/${FOLDER_NAME}.parquet"

echo "========================================="
echo "Processing Details"
echo "========================================="
echo "Processing folder: $INPUT_FOLDER"
echo "Output file: $OUTPUT_FILE"
echo ""

# Run the Python script
python ${PROJECT_DIR}/process_xml_to_parquet.py "$INPUT_FOLDER" "$OUTPUT_FILE"
```

### Merge by Decade

The XML files are unsorted, so the newly created Parquet files are as well. merge_by_decade.py creates a single Parquet file for each decade. Go to project folder and run merge_by_decade.py.

### Train Decade-Specific Models

Now with a Parquet file for each decade, the next step is text processing and model training. The relevant script is train_embeddings.py, and it can run iteratively over the decade files (while avoiding recreating models) as follows

```
for DECADE in 1850s 1860s 1870s 1880s 1890s 1900s 1910s 1920s 1930s; do
    INPUT_FILE="parquet_by_decade/nyt_${DECADE}.parquet"
    OUTPUT_DIR="/output/folder/goes/here/${DECADE}"
    MODEL_FILE="${OUTPUT_DIR}/word2vec_${DECADE}.model"

    if [ -f "$MODEL_FILE" ]; then
        echo "Model already exists for $DECADE — skipping."
        continue
    fi

    mkdir -p "$OUTPUT_DIR"
    echo "Running Word2Vec for $DECADE..."

    python train_embeddings.py "$INPUT_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --vector-size 300 \
        --window 5 \
        --min-count 15 \
        --epochs 5 \
        --workers 16
done
```
Note! _This script does not currently work for 1880s-1920s due to a pyarrow error!_ Hope to update soon!

### Analysis

Current run_analysis.py file is rather minimal and tailored to my specific research interests, however, the gender and geography analogy tests are worthwhile for demonstrating the viability of the models. The nyt_word2vec_temporal.ipynb notebook implements a method for finding and visualizing a temporality dimension.

## Results

### Robustness check
As a first step after training the models, I wanted to demonstrate that they did, in fact, encode semantic information. I started with a few famous examples, first proving that the models could solve gender analogies by taking the vector for “father,” subtracting the vector for “man,” and adding the vector for “woman,” and then asking for the closest vector to that position. I found that in all the models, “mother” was either the closest (or functionally tied for the closest) vector. 
Next, having demonstrated that simple semantic information was encoded, I sought to also show that historical developments would be reflected in the decade-by-decade encodings. Using the same analogy test, but for geographic information, I took the vector for “France,” subtracted the vector for “Paris,” and added the vector for “Berlin,” functionally asking “what country is Berlin the capital of?” Notably, however, Germany was only a proposed national concept, not a nation, until German unification in 1871. Here are the answers to this analogy over time:

=== Results for 1850s ===
Geography analogy:
  prussia: 0.614
  germany: 0.556
  russia: 0.523

=== Results for 1860s ===
Geography analogy:
  prussia: 0.711
  austria: 0.678
  russia: 0.648

=== Results for 1870s ===
Geography analogy:
  germany: 0.726
  austria: 0.664
  prussia: 0.664

=== Results for 1880s ===
Geography analogy:
  germany: 0.757
  austria: 0.648
  gormany: 0.643

Until German unification, Prussia is listed as the top result, which is historically accurate: Berlin was the capital of Prussia until the beginning of the German Empire in 1871. In the 1870s we still see Prussia in the top 3 results, but by the 1880s it’s not even in the top 20. The analogy example is responsive to historical developments in this diachronic approach.

### Temporality dimension

I find the temporality dimension by subtracting paired temporal antonyms and taking the average of the difference between the vectors. The antonym pairs are checked against the vocabulary of each decade to ensure the words were in use at that time (e.g. "futuristic" not in wide usage in the 19th century). I demonstrate that the temporality dimension encodes meaning relevant to the distinct decades by showing how concepts shift along the temporality dimension over time: "phonograph" becomes less future-oriented in the decades after its invention, "highway" becomes more future-oriented as the highway system for automobiles is instituted in the United States. See nyt_word2vec_temporal.ipynb or my final presentation (https://drive.google.com/file/d/1-SA-XpELFMVBWACbW83Cp4eeGPLPKY7D/view) for visualizations of this temporal axis. 

#### Disclosure:
Derived from the original data viz course by Mine Çetinkaya-Rundel @ Duke University
