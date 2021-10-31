# Letter images clustering
This is an ML university assignment to cluster a set of very low quality letter images.

## Method
I used KMeans algorithm, with silhouette score to find an optimal K.
Images are converted to greyscale, scaled to 10x10 px and sharpened.
I also take the height:width ratio into account as a separate feature.

## How to run

```python venv -m env
source env/bin/activate.sh
pip install -r requirements.txt
python main.py <path_to_filelist>```

## Execution time for training set:

```real	3m54,843s
user	11m20,421s
sys	3m3,642s```

