import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random
import umap
def feature_visualize(img_embd, txt_embd):

    features = np.concatenate((img_embd,txt_embd))
    embed= TSNE(n_components=2).fit_transform(features)
    #embed = umap.UMAP(n_components=2).fit_transform(features)
    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = embed[:, 0]
    ty = embed[:, 1]

    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)
    # initialize a matplotlib plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors=['#0000ff', '#F10000', '#F1EB00', '#4EAA00', '#384A00', '#384A72',
    '#ccccff', '#eeeeff', '#ffeeee', '#ffcccc', '#ffaaaa', '#ff8888', '#ff6666',
    '#ff4444', '#ff2222', '#ff0000']
    # for every class, we'll add a scatter plot separately
    labels=["1","2",'3','4','5','6','7','8','9','10']
    indices = random.sample(range(img_embd.shape[0]),10)
    indices_txt = [i+img_embd.shape[0] for i in indices]
    #print(indices)
    #print(indices_txt)
    for i in range(10):
        inx=[indices[i],indices_txt[i]]
        print(inx)
        x = np.take(tx, inx)
        y = np.take(ty, inx)
    # convert the class color to matplotlib format
        ax.plot(x,y,zorder=1)
        ax.scatter(x, y, c=colors[i],label=labels[i],zorder=2)

    # build a legend using the labels we set previously
    ax.legend(loc='best')

    # finally, show the plot
    plt.savefig('test.png')
# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range
