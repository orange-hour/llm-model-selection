# Weighted Voting with Disagreement-based Random Sampling (WVDRS)
This repository contains the implementation of the Weighted Voting with Disagreement-based Random Sampling (WVDRS) algorithm, which enhances LLM-based query processing over relational databases.
## Datasets
<ol>
<li>Stanford Natural Language Inference (SNLI)</li>
<li>Fact Extraction and Verification (FEVER)</li>
<li>AG News</li>
<li>subsampled 385 data points from IMDB movie reviews</li>
</ol>

## Dependencies
- Pandas
- Numpy
- Matplotlib

## Running code
Our code is tested with Python 3.10.14, Numpy 1.26.4, Pandas 2.2.2, Matplotlib 3.9.2. To reproduce accuracy improvement plots for each dataset, run

```
cd weighted_voting

# SNLI
process_nli_final.ipynb

# FEVER
process_fever_final.ipynb

# AG News
process_ag_news_final.ipynb

# subsampled 385 data points from IMDB movie reviews
process_imdb_final.ipynb
```

### Citation

```
```
</br>
Please contact <a href="https://trung6.github.io/">Trung Nguyen</a> for questions or comments.
