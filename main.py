def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
import numpy as np

#load data
data_batch_1 = unpickle('./cifar-10-batches-py/data_batch_1')
data_batch_2 = unpickle('./cifar-10-batches-py/data_batch_2')
data_batch_3 = unpickle('./cifar-10-batches-py/data_batch_3')
data_batch_4 = unpickle('./cifar-10-batches-py/data_batch_4')
data_batch_5 = unpickle('./cifar-10-batches-py/data_batch_5')
test_batch = unpickle('./cifar-10-batches-py/test_batch')


#Split x y
train_1 = data_batch_1[b'data']
train_2 = data_batch_2[b'data']
train_3 = data_batch_3[b'data']
train_4 = data_batch_4[b'data']
train_5 = data_batch_5[b'data']

y_1 = data_batch_1[b'labels']
y_2 = data_batch_2[b'labels']
y_3 = data_batch_3[b'labels']
y_4 = data_batch_4[b'labels']
y_5 = data_batch_5[b'labels']

test_x = test_batch[b'data']
test_y = np.array(test_batch[b'labels']).reshape(-1)



#All training sets are combined
import numpy as np
train_x = np.vstack((train_1, train_2, train_3, train_4, train_5))
train_y = np.hstack((y_1, y_2, y_3, y_4, y_5))


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#dimension reduction
pca = PCA(n_components=300)
train_x = pca.fit_transform(train_x)
test_x = pca.transform(test_x)

#normalize

ss = StandardScaler()

train_x = ss.fit_transform(train_x)
test_x = ss.transform(test_x)


from sklearn.model_selection  import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score, recall_score
import matplotlib.pyplot as plt
 
k_range = range(2,20)
a = []
r = []
f = []
#loop，from k=1 to k=31，view error effect
from tqdm import tqdm

best_acc = 0
for k in tqdm(k_range):
    
    print('K={}'.format(k))
    
    knn = KNeighborsClassifier(n_neighbors=k, p=1, n_jobs=-1)  #p=1  this is equivalent to using manhattan_distance (l1)
    knn.fit(train_x, train_y)
    pred = knn.predict(test_x)
    
    acc = accuracy_score(test_y, pred)
    recall = recall_score(test_y, pred, average='micro')
    f1 = f1_score(test_y, pred, average='micro')
    
    a.append(acc)
    r.append(recall)
    f.append(f1)

    print(classification_report(test_y, pred))
    
    print('#######################')
    #The cv parameter determines the data set division ratio, here is the 5:1 division of training set and test set
    #knn.fit(train_x, train_y)
    #score = knn.score(train_x, train_y)
    #scores = cross_val_score(knn, train_x[:10000], train_y[:10000], cv=5, scoring='accuracy', verbose=1)
    #k_error.append(score)
    
    


figure = plt.figure(figsize=(10, 8))
plt.plot(k_range, f)
plt.xlabel('Value of K for KNN')
plt.ylabel('F1')
plt.savefig('f1.png', dpi=300)


