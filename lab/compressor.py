# %%
from sklearn.cluster import KMeans
from PIL import Image
import numpy as np

# %%
def compress(Img_path,n_color):
    n_color=min(n_color,256*256)
    img=Image.open(Img_path)
    img=np.array(img)
    height, width, channels = img.shape
    shape=np.array([height,width,channels],dtype=np.int32)
    pixels = img.reshape(-1, channels)

    km=KMeans(n_clusters=n_color,init='k-means++',)
    km.fit(pixels)
    centroids=km.cluster_centers_
    centroids=centroids.astype(np.uint8)
    labels=km.labels_

    if n_color<257:
        labels=labels.astype(np.int8)
    else:
        labels=labels.astype(np.int16)
    
    np.savez_compressed(f"{Img_path.split('.')[0]}.npz",labels=labels,centroids=centroids,shape=shape)

# %%
compress(Img_path="image2.png",n_color=256)

# %%
def extract(img_path):
    img=np.load(img_path)
    labels,centroids,shape=img['labels'],img['centroids'],img['shape']
    print(centroids)
    extracted_img=np.empty((shape[0]*shape[1],shape[2]))
    for ind,label in enumerate(labels):
        extracted_img[ind]=centroids[label]
    extracted_img=extracted_img.reshape(shape)
    print(extracted_img.shape)
    extracted_img=extracted_img.astype(np.uint8)
    ex_img=Image.fromarray(extracted_img)
    ex_img.save("extracted.png")
    

# %%
extract("image2.npz")

# %%



