#!/usr/bin/env python
from os import path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS

from datetime import datetime

# visualizing Top_K words in the final figure
Top_K = 200

# get data directory (using getcwd() is needed to support running example in console)
d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()

# Read the whole text generated from wiki
# an introduction to Avengers series films
text = open(path.join(d, 'Avengers.txt')).read()

# read the mask / color image taken from
coloring = np.array(Image.open(path.join(d, "avengers_sample.jpg")))

# get the stop words to ignore resource waste
stopwords = set(STOPWORDS)
stopwords.add("said")

# using the wordcloud model by call the wordcloud package
# backgroud color was setted as white
# using mask
# maximum font size was setted as 100

# method explain
# Image-colored wordcloud method
# You can color a word-cloud by using an image-based coloring strategy
# implemented in ImageColorGenerator. It uses the average color of the region
# occupied by the word in a source image. You can combine this with masking -
# pure-white will be interpreted as 'don't occupy' by the WordCloud object when
# passed as mask.
# If you want white as a legal color, you can just pass a different image to
# "mask", but make sure the image shapes line up.

wc = WordCloud(background_color="white", max_words=Top_K, mask=coloring,
               stopwords=stopwords, max_font_size=100, random_state=42)

# generate word cloud
wc.generate(text)

# create coloring from image
image_colors = ImageColorGenerator(coloring)

# show
fig, axes = plt.subplots(1, 2, figsize=(20, 16) )

#axes[0].imshow(wc, interpolation="bilinear")
# recolor wordcloud and show


axes[0].imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
axes[1].imshow(coloring, cmap=plt.cm.gray, interpolation="bilinear")
for ax in axes:
    ax.set_axis_off()

plt.savefig("word_cloud_result_At_{}".format(datetime.utcnow().isoformat().split('.')[0].replace(':', '-')), dpi=400)
plt.show()
