# %%
import numpy as np
from sklearn.model_selection import train_test_split

#全部資料
raw_data_cnt = 1095
used_data = []
y_train = []
raw_data = []

#讀取資料
for x in range(raw_data_cnt):
    with open('./PA1-data/'+str(x+1)+'.txt', 'r') as file:
        file_contents = file.read()
        raw_data.append(file_contents)

#讀取訓練資料
with open("./training_new.txt", 'r') as file:
    training_data = file.read()
    training_data = training_data.split('\n')
    training_data = [x.split(' ') for x in training_data]
    for x in range(len(training_data)):
        training_data[x] = training_data[x][0:-1]
        training_data[x] = [int(y) for y in training_data[x]]

#分割資料
x_train = []
for x in range(len(training_data)):
    for y in range(1, len(training_data[x])):
        x_train.append(raw_data[training_data[x][y]-1])
        y_train.append(training_data[x][0])

x_train = np.array(x_train)
y_train = np.array(y_train)


# %%
from keras_bert import extract_embeddings
from keras_bert import load_vocabulary
from keras_bert import Tokenizer

model_path = './uncased_L-12_H-768_A-12'
dict_path = model_path+'/vocab.txt'

bert_token_dict = load_vocabulary(dict_path)
bert_tokenizer = Tokenizer(bert_token_dict)

for x in raw_data:
    tokens = bert_tokenizer.tokenize(x)
    indices, segments = bert_tokenizer.encode(first=x)

embeddings = extract_embeddings(model_path, x_train)


# %%
print(len(embeddings))
X_convert = []

for x in range(len(embeddings)):
    X_convert.append(embeddings[x][0])

x_train, x_test, y_train, y_test = train_test_split(X_convert, y_train, test_size=0.1, random_state=3, stratify = y_train)


# %%
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier

# Create a one-vs-all classifier using SVM with a linear kernel
model_linear = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True))
model_linear.fit(x_train, y_train)

# %%
from sklearn.metrics import classification_report

y_pred_linear = model_linear.predict(x_test)
report_linear = classification_report(y_test, y_pred_linear, zero_division=1)

print("Linear SVM Model:")
print(report_linear)

# %%
import numpy as np
from sklearn.model_selection import train_test_split

#全部資料
raw_data_cnt = 1095
used_data = []
y_train = []
raw_data = []

#讀取資料
for x in range(raw_data_cnt):
    with open('./PA1-data/'+str(x+1)+'.txt', 'r') as file:
        file_contents = file.read()
        raw_data.append(file_contents)

#讀取訓練資料
with open("./training_new.txt", 'r') as file:
    training_data = file.read()
    training_data = training_data.split('\n')
    training_data = [x.split(' ') for x in training_data]
    for x in range(len(training_data)):
        training_data[x] = training_data[x][0:-1]
        training_data[x] = [int(y) for y in training_data[x]]

#分割資料
x_train = []
for x in range(len(training_data)):
    for y in range(1, len(training_data[x])):
        x_train.append(raw_data[training_data[x][y]-1])
        y_train.append(training_data[x][0])

training_data = np.array(training_data)
training_data = np.delete(training_data, 0, 1)
training_data = training_data.flatten()
training_data.sort()

predict = np.delete(raw_data, training_data-1)
predict = extract_embeddings(model_path, predict)




# %%
real_predict = []
for x in range(len(predict)):
    real_predict.append(predict[x][0])

y_pred = model_linear.predict(real_predict)
# save the prediction to a csv
import csv
with open('./prediction2.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Id', 'Value'])
    i = 0
    for x in range(raw_data_cnt):
        if x+1 not in training_data:
            writer.writerow([x+1, y_pred[i]])
            i+=1


