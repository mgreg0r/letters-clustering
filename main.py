import sys
import os
from sklearn.cluster import KMeans
from skimage.util import img_as_float
from skimage.filters import unsharp_mask
from skimage.transform import resize
from sklearn.metrics import silhouette_score
import skimage.io
import numpy as np

listfile = sys.argv[1]

with open(listfile) as f:
    filenames = f.read().splitlines()

N = len(filenames)

images = [skimage.io.imread(filename, as_gray=True) for filename in filenames]

def transform_image(image):
    resized = resize(image, (10, 10))
    sharp = unsharp_mask(resized, radius=5, amount=2)
    return np.array([(len(image) / len(image[0])) * 2] + [x for r in sharp for x in r])

processed_images = np.array([transform_image(image) for image in images])

sil = []
for i in range(25, 35):
    kmeans = KMeans(n_clusters=i, n_init=10)
    kmeans.fit(processed_images)
    labels = kmeans.labels_
    sil.append(silhouette_score(processed_images, labels, metric = 'euclidean'))

K = sil.index(max(sil)) + 25

kmeans = KMeans(n_clusters=K, n_init=50)
kmeans.fit(processed_images)
Z = kmeans.predict(processed_images)

def process_results(results, filenames):
    by_cluster = [[] for i in range(K)]
    for i in range(1, N):
        by_cluster[results[i]].append(filenames[i])

    return by_cluster


def generate_html(files_by_cluster):
    htmlstrings = []
    htmlstrings.append("<body>")

    for cluster in files_by_cluster:
        for image in cluster:
            htmlstrings.append('<img src="{}"/>'.format(image))
        htmlstrings.append("<HR/>")


    htmlstrings.append("</body>")
    with open("output.html", "w") as f:
        f.write("".join(htmlstrings))


def generate_txt(files_by_cluster):
    txt = "\n".join([" ".join([os.path.basename(path) for path in cluster]) for cluster in files_by_cluster])
    with open("output.txt", "w") as f:
        f.write(txt)

results = process_results(Z, filenames)
generate_txt(results)
generate_html(results)

