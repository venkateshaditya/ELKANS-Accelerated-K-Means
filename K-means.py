import pandas
import numpy
import os
import math
from sklearn.model_selection import train_test_split


def initialize_centroids(data, k):
    numpy.random.shuffle(data)
    return data[:k]


def dist(cluster, point):
    global dist_calls, distance_func
    dist_calls += 1
    distance = 0
    if distance_func == 0:  # Euclidean
        distance = numpy.linalg.norm(cluster - point)
    elif distance_func == 1:  # Manhattan
        distance = numpy.sum(numpy.abs(cluster - point))
    elif distance_func == 2:  # Cosine
        distance = 1 - numpy.sum(numpy.dot(cluster, point)) / (numpy.linalg.norm(cluster) * numpy.linalg.norm(point))
    elif distance_func == 3:  # dP Equation 1
        temp = cluster - point
        distance = math.sqrt((numpy.sum((temp > 0) * temp) ** 2) + (numpy.sum((temp < 0) * temp) ** 2))
    elif distance_func == 4:  # dPN Equation 2
        temp = cluster - point
        distance = math.sqrt((numpy.sum((temp > 0) * temp) ** 2) + (numpy.sum((temp < 0) * temp) ** 2))
        temp = numpy.concatenate((abs(cluster), abs(point), abs(temp)))
        temp = numpy.amax(temp, axis=0)
        distance = distance / numpy.sum(temp)
    return distance


def closest_centroid(point, centroids):
    closest_cluster, cluster = 0, centroids[0]
    distance = dist(cluster, point)
    if distance == 0:
        closest_cluster, distance = 1, dist(centroids[1], point)
    for i in range(0, centroids.shape[0]):
        cluster = centroids[i]
        temp = dist(cluster, point)
        if temp < distance:
            closest_cluster = i
            distance = temp
    return closest_cluster, distance


def recompute_centroids(data, cluster_assignment, centroids, k):
    for i in range(k):
        subset = data[cluster_assignment['cluster'] == i]
        centroids[i] = subset.mean(axis=0)
    return centroids


def assign_cluster(data, cluster_assignment, centroids):
    for i in range(0, data.shape[0]):
        cluster, distance = closest_centroid(data[i], centroids)
        cluster_assignment.iloc[i, 1] = cluster
        cluster_assignment.iloc[i, 2] = distance
    # cluster_assignment.index = range(0, cluster_assignment.shape[0])
    return cluster_assignment


def kmeans(data, k, centroids):
    cluster_assignment = pandas.DataFrame(columns=list(['point', 'cluster', 'distance']))
    iteration = 1
    for i in range(0, data.shape[0]):
        cluster_assignment = cluster_assignment.append(
            pandas.DataFrame([[i, None, None]], columns=list(['point', 'cluster', 'distance'])))
    cluster_assignment = assign_cluster(data, cluster_assignment.copy(), centroids)
    centroids = recompute_centroids(data, cluster_assignment, centroids.copy(), k)
    SSE = 0
    while True:
        iteration += 1
        # print("Kmeans Iteration=%d    SSE=%9.5f" % (iteration, SSE))
        cluster_assignment = assign_cluster(data, cluster_assignment.copy(), centroids)
        new_centroids = recompute_centroids(data, cluster_assignment, centroids.copy(), k)
        # no_changed_centroids = sum((centroids != new_centroids).any(1))
        SSE = numpy.sum(numpy.square(centroids - new_centroids))
        if SSE < 0.0001 or iteration > 1000:
            break
        centroids = new_centroids
    cluster_assignment.to_csv("basic_kmeans.csv", mode='w', header=True)
    print("\nLoyds's K-means iterations=", iteration)
    return centroids, cluster_assignment['cluster'], SSE


def elkan_initial_cluster(data, cluster_assignment, centroids, s, cluster_dm, lower_bounds, upper_bounds):
    for i in range(0, data.shape[0]):
        cluster_assignment.iloc[i, 2] = lower_bounds[i, 0] = upper_bounds[i] = dist(data[i], centroids[0])
        cluster_assignment.iloc[i, 1] = 0
        for j in range(1, centroids.shape[0]):
            if cluster_dm[j, cluster_assignment.iloc[i, 1]] >= 2 * cluster_assignment.iloc[i, 2]:
                continue
            temp = dist(centroids[j], data[i])
            if temp < cluster_assignment.iloc[i, 2]:
                cluster_assignment.iloc[i, 2] = lower_bounds[i, j] = upper_bounds[i] = temp
                cluster_assignment.iloc[i, 1] = j
    return cluster_assignment, lower_bounds, upper_bounds


def self_distance_matrix(a):
    dist_mat = numpy.zeros(shape=(a.shape[0], a.shape[0]))
    for i in range(0, a.shape[0]):
        for j in range(i + 1, a.shape[0]):
            if i != j:
                dist_mat[i][j] = dist_mat[j][i] = dist(a[i], a[j])
    return dist_mat


def distance_matrix(a, b):
    dist_mat = numpy.zeros(shape=(a.shape[0], b.shape[0]))
    for i in range(0, a.shape[0]):
        for j in range(0, b.shape[0]):
            dist_mat[i][j] = dist(a[i], b[j])
    return dist_mat


def Elkans(data, k, centroids):
    cluster_assignment = pandas.DataFrame(columns=list(['point', 'cluster', 'distance']))
    n = data.shape[0]
    lower_bounds = numpy.zeros(shape=(n, k))
    upper_bounds = numpy.zeros(n)
    cluster_dm = self_distance_matrix(centroids)
    s = 0.5 * numpy.sort(cluster_dm)[:, 1]
    iteration = 0
    SSE = 0
    for i in range(n):
        cluster_assignment = cluster_assignment.append(
            pandas.DataFrame([[i, None, None]], columns=list(['point', 'cluster', 'distance'])))
    cluster_assignment.index = range(0, cluster_assignment.shape[0])
    cluster_assignment, lower_bounds, upper_bounds = elkan_initial_cluster(data, cluster_assignment.copy(), centroids,
                                                                           s, cluster_dm, lower_bounds.copy(),
                                                                           upper_bounds.copy())
    # new_centroids = recompute_centroids(data, cluster_assignment, centroids.copy(), k)

    while True:
        iteration += 1
        # print("Elkans Iteration=%d    SSE=%9.5f" % (iteration, SSE))
        cluster_dm = self_distance_matrix(centroids.copy())
        s = 0.5 * numpy.sort(cluster_dm.copy())[:, 1]
        for i in range(n):
            centroid_i = cluster_assignment.iloc[i, 1]
            if upper_bounds[i] <= s[centroid_i]:
                continue
            r = True
            for j in range(k):
                if centroid_i == j:
                    continue
                if (upper_bounds[i] > lower_bounds[i, j]) and (upper_bounds[i] > 0.5 * cluster_dm[j, centroid_i]):

                    if r:
                        r = False
                        upper_bounds[i] = dist(data[i], centroids[centroid_i])
                    if upper_bounds[i] > lower_bounds[i, j] or upper_bounds[i] > 0.5 * cluster_dm[j, centroid_i]:
                        lower_bounds[i, j] = temp2 = dist(data[i], centroids[j])
                        if temp2 < upper_bounds[i]:
                            cluster_assignment.iloc[i, 1] = centroid_i = j
                            cluster_assignment.iloc[i, 2] = upper_bounds[i] = temp2

        new_centroids = recompute_centroids(data, cluster_assignment, centroids.copy(), k)
        SSE = numpy.sum(numpy.square(centroids - new_centroids))
        if SSE < 0.0001 or iteration > 1000:
            break

        delta = numpy.zeros(k)
        for j in range(k):
            delta[j] = dist(centroids[j], new_centroids[j])
        for i in range(n):
            for j in range(k):
                lower_bounds[i, j] = max(lower_bounds[i, j] - delta[j], 0)
        for i in range(n):
            upper_bounds[i] = upper_bounds[i] + delta[cluster_assignment.iloc[i, 1]]

        centroids = new_centroids.copy()

    print("\nElkans iterations=", iteration)

    cluster_assignment.to_csv("Elkan_kmeans.csv", mode='w', header=True)
    return centroids, cluster_assignment['cluster'], SSE


if __name__ == "__main__":
    global distance_func, dist_calls
    distance_func = 0
    # filepath = input("Enter Complete file path: \n")
    # filepath = r"ionosphere.csv"
    # filepath = r"iris_data1.csv"
    # filepath = r"covtype.csv"
    filepath = r"seeds.csv"
    print("Dataset : ", os.path.splitext(filepath)[0])
    extension = os.path.splitext(filepath)[1]
    if extension == ".csv":
        data = pandas.read_csv(filepath, sep=",", header=None, engine='c')
        columns = numpy.asarray(data.columns)
        labels = data.pop(columns[-1])
        k = len(numpy.unique(labels))
        if data.shape[0] > 1000:
            rows = int(input("Large data set with %d rows,Choose number of rows-" % data.shape[0]))
            data, test, train_labels, test_labels = train_test_split(data, labels, train_size=rows, test_size=10,
                                                                     stratify=labels)
            labels=train_labels
            k = len(numpy.unique(labels))
            data = data.as_matrix()
            print("Stratified sampling of %d rows" % data.shape[0])
        else:
            data = data.as_matrix()
    elif extension == ".xlsx" or extension == '.xls':
        data = pandas.read_excel(filepath, sheet_name=None)
        sheet = str(list(data.keys())[0])
        data = data[sheet]
        data = data.dropna()
        labels = data.pop()
        if data.shape[0] > 1000:
            train, test = train_test_split(data, labels, train_size=1000, stratify=labels)
            data = train.as_matrix()
    elif extension == ".txt":
        data = pandas.read_table(filepath, sep=",", header='infer', engine='c')
        labels = data.pop()
        if data.shape[0] > 1000:
            train, test = train_test_split(data, labels, train_size=1000, stratify=labels)
            data = train.as_matrix()

    # data = pandas.DataFrame(numpy.random.randint(5, size=(5, 5)))
    distance_func = int(input("Select Distance function:\n0.Euclidean\n1.City block\n2.Cosine\n3.dP\n4.dPN\nYour "
                              "choice:"))
    print("Suggested number of clusters-", k)
    k = int(input("Enter number of clusters K: "))
    centroids = initialize_centroids(data.copy(), k)
    dist_calls = 0
    c1, as1, SSE_Kmeans = kmeans(data, k, centroids.copy())
    print("K-means dist calls=%d  SSE=%9.5f" % (dist_calls, SSE_Kmeans))
    dist_calls = 0
    c2, as2, SSE_Elkans = Elkans(data, k, centroids.copy())
    print("Elkans dist calls=%d  SSE=%9.5f" % (dist_calls, SSE_Elkans))

    # print("\nDifference between Elkans and K-means centroids\n")
    # print(c1 - c2)
    # print("\nNo of mismatch between K-means and Elkan's cluster assignment")
    # print(sum((numpy.array(as1) != numpy.array(as2))))
    if (k == len(numpy.unique(labels))):
        print("\nAccuracy")
        total=0
        for i in range(k):
            indices_class_i = (pandas.Categorical(labels).codes == i)
            accuracy = (as2.iloc[indices_class_i]).value_counts().iloc[0] / sum(indices_class_i)
            total += accuracy
            print("Class %d = %d" % (i, accuracy * 100), '%')
        print("Total Accuracy = %d" % (total/k * 100), '%')

    print("\nDistribuition across clusters")
    print(as2.value_counts())
