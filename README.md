# Introduction
Image segmentation is a process in computer vision where an image is divided into multiple segments, each of which corresponds to different objects or parts of objects. This process allows for more precise analysis and understanding of images by focusing on individual components rather than the image as a whole. In essence, image segmentation transforms a generic image into a collection of regions of pixels with the same attributes, such as color, intensity, or texture. This technique is widely used in various fields, including medical imaging, autonomous driving, and surveillance systems. 
## Methods of Segmentation
I have used 2 methods to perform image segmentation tasks.
### 1.) K-Means Based Clustering Approach:
In my experimentation with image segmentation, I've found the K-Means clustering method to be highly valuable. This technique functions by dividing an image into K separate regions or clusters, where K is a predetermined value.
Initially, each pixel in the image is randomly assigned to a cluster. Then, the centroid (or central point) of each cluster is calculated. Throughout the algorithm's iterations, pixels are reassigned to the cluster with the nearest centroid. This iterative process continues until there's no further change in pixel assignments or until a specified maximum iteration limit is reached.
Consequently, the image is segmented into K regions, each represented by the color of its corresponding centroid. While this approach effectively distinguishes between different regions within an image, it does have limitations, especially when the number of clusters is unknown beforehand or varies significantly.
### 2.) Ratio-Cut based Clustering Technique:
This method relies on graph theory, where an image is depicted as a graph with individual pixels as nodes and the similarity between pixels as the edge weight.
The objective is to divide the graph into multiple subgraphs, each representing a segment in the image. The basis for this division is the "ratio cut," calculated as the total weight of the cut edges divided by the sum of the sizes of the various subgraphs. The aim is to minimize this ratio cut, leading to segments that are internally coherent yet distinct from one another.
## Approach
For this task, I first calculated the pixel intensity and position differences and using it I calculated the affinity matrix (similarity score) utilizing the equations mentioned below.
Then I calculated the degree matrix, which is a diagonal matrix, by summing up elements of the row and giving the diagonal positions that value.
#### D = np.diag(np.sum(s, axis=1))
Then for calculating Laplacian matrix, I used the formula,
#### L = D - s
For ratio- cut problem:
Solving the above equation we get the answer of the H matrix as the top K min eigenvalue vectors. I first calculated the eigen values and eigen vectors. After calculating the eigen values and eigen vector, I sorted the eigen values in ascending order and return the corresponding indices .Then I selected the first k smallest eigen values and then selected the columns of their eigen vectors and took them as clusters to be used in K-means clustering.
