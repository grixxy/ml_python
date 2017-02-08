import matplotlib.pyplot as plt

def plotProgresskMeans(X, centroids, previous, idx, K, i):
# PLOTPROGRESSKMEANS is a helper function that displays the progress of
# k-Means as it is running. It is intended for use only with 2D data.
#    PLOTPROGRESSKMEANS(X, centroids, previous, idx, K, i) plots the data
#    points with colors assigned to each centroid. With the previous
#    centroids, it also plots a line between the previous locations and
#    current locations of the centroids.


# Plot the examples
    plotDataPoints(X, idx, K)

# Plot the centroids as black x's
    plt.plot(centroids[:,0], centroids[:,1],'x')
    move_lines_x = [[c1[0],c2[0]] for (c1,c2) in zip(previous, centroids) ]
    move_lines_y = [[c1[1],c2[1]] for (c1,c2) in zip(previous, centroids) ]
# Plot the history of the centroids with lines
    for j in range(len(move_lines_x)):

        plt.plot(move_lines_x[j], move_lines_y[j])

    plt.show()


def plotDataPoints(X, idx, K):
#PLOTDATAPOINTS plots data points in X, coloring them so that those with the same
#index assignments in idx have the same color
#   PLOTDATAPOINTS(X, idx, K) plots data points in X, coloring them so that those
#   with the same index assignments in idx have the same color

# Create palette
    #todo palette
    #palette = hsv(K + 1);
    #colors = palette(idx, :);

# Plot the data
    plt.scatter(X[:,0], X[:,1])


