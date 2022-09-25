# %%
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import pandas as pd
import skimage
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# %%
images = load_digits()['images']
target = load_digits()['target']

# %%
print('image size is',images.shape[1:])

# %%
plt.figure(figsize=(10, 10))
for idx, val in enumerate([100, 200, 300, 1500]):
    plt.subplot(2, 2, idx + 1)
    plt.imshow(images[val])
    plt.axis('off')
    plt.title('original')
plt.show()

# %%
shape = (len(images), images.shape[1] * images.shape[2])
reshaped_images = images.reshape(shape)


xtrain, xtest, ytrain, ytest = train_test_split(reshaped_images, target, random_state=1, 
                                                stratify=target, test_size=0.2)
xtrain, xvalid, ytrain, yvalid = train_test_split(xtrain, ytrain, random_state=1, stratify=ytrain,
                                                  test_size=0.6)

xtrain.shape, xvalid.shape, xtest.shape



C  = [0.1, 1, 10, 30, 50, 70,100] 
gamma =  [0.0001,0.001,0.01,1,10,50,100]

C_list = []
gamma_list = []
train_acc_list = []
valid_acc_list = []

for c in C:
    for gm in gamma:
        model = SVC(C = c, gamma = gm)
        model.fit(xtrain, ytrain)
        train_acc = model.score(xtrain,ytrain )
        valid_acc = model.score(xvalid, yvalid)
        
        C_list.append(c)
        gamma_list.append(gm)
        train_acc_list.append(train_acc)
        valid_acc_list.append(valid_acc)


df = pd.DataFrame({'C' : C_list, 'gamma' : gamma_list, 'train_accuracy' : train_acc_list, 
                   'dev_accuracy' : valid_acc_list})
#display(df)

max_score = df['dev_accuracy'].max()
best_params = df[df['dev_accuracy'] == max_score]

best_c = best_params['C'].values[0]
best_gamma = best_params['gamma'].values[0]



model = SVC(C = best_c, gamma = best_gamma)
model.fit(xtrain, ytrain)


train_acc = model.score(xtrain, ytrain)
valid_acc = model.score(xvalid, yvalid)
test_acc = model.score(xtest, ytest)   


df_final = pd.DataFrame({'train_accuracy' : [train_acc],'dev_accuracy' : [valid_acc], 
                         'test_accuracy' : [test_acc]})
df_final.index = ['C ' + str(best_c) + ' gamma ' + str(best_gamma)]

#display(df_final)

# %%
"""
### Accuracy scores for different Gamma & C 
"""

# %%
display(df)

# %%
"""
### Max Dev Accuracy
"""

# %%
print("Best Accuracy for dev is :",round(max_score, 2))

# %%
"""
### Best Parameters
"""

# %%
print("Value for C for Max Dev Accuracy :",best_c)
print("Value for Gamme for Max Dev Accuracy :",best_gamma)

# %%
"""
### Scores for Dev, Test & Train with Best Parameters
"""

# %%
display(df_final)

# %%
"""
### Orignal Image Size
"""

# %%
print('image size is',images.shape[1:])

# %%
"""
## Resolution -1
"""

# %%
ros1_image = skimage.transform.resize(images, (len(images), 20, 20))
shape = (len(ros1_image), ros1_image.shape[1] * ros1_image.shape[2])
reshaped_images = ros1_image.reshape(shape)


xtrain, xtest, ytrain, ytest = train_test_split(reshaped_images, target, random_state=1, 
                                                stratify=target, test_size=0.2)
xtrain, xvalid, ytrain, yvalid = train_test_split(xtrain, ytrain, random_state=1, stratify=ytrain,
                                                  test_size=0.6)


C  = [0.1, 1, 10, 30, 50, 70,100] 
gamma =  [0.0001,0.001,0.01,1,10,50,100]

C_list = []
gamma_list = []
train_acc_list = []
valid_acc_list = []

for c in C:
    for gm in gamma:
        model = SVC(C = c, gamma = gm)
        model.fit(xtrain, ytrain)
        train_acc = model.score(xtrain, ytrain)
        valid_acc = model.score(xvalid, yvalid)
        
        C_list.append(c)
        gamma_list.append(gm)
        train_acc_list.append(train_acc)
        valid_acc_list.append(valid_acc)
        
df = pd.DataFrame({'C' : C_list, 'gamma' : gamma_list, 'train_accuracy' : train_acc_list, 
                   'dev_accuracy' : valid_acc_list})

max_score = df['dev_accuracy'].max()
best_params = df[df['dev_accuracy'] == max_score]

best_c = best_params['C'].values[0]
best_gamma = best_params['gamma'].values[0]

#print('resolution :',(20,20))
#print(best_c, best_gamma)


model = SVC(C = best_c, gamma = best_gamma)
model.fit(xtrain, ytrain)


train_acc = model.score(xtrain, ytrain)
valid_acc = model.score(xvalid, yvalid)
test_acc = model.score(xtest, ytest)   

#print('test accuracy is',round(test_acc, 2))

df_final = pd.DataFrame({'train_accuracy' : [train_acc],'dev_accuracy' : [valid_acc], 
                         'test_accuracy' : [test_acc]})
df_final.index = ['C ' + str(best_c) + ' gamma ' + str(best_gamma)]

#df_final

# %%
plt.figure(figsize=(10, 10))
for idx, val in enumerate([100, 200, 300, 1500]):
    plt.subplot(2, 2, idx + 1)
    plt.imshow(ros1_image[val])
    plt.axis('off')
    plt.title('original')
plt.show()

# %%
"""
### Accuracy scores for different Gamma & C for Resolution 20*20
"""

# %%
display(df)

# %%
"""
### Max Dev Accuracy for Resolution 20*20
"""

# %%
print("Best Accuracy for dev is :",round(max_score, 2))

# %%
"""
### Best Parameters  for Resolution 20*20
"""

# %%
print("Value for C for Max Dev Accuracy :",best_c)
print("Value for Gamme for Max Dev Accuracy :",best_gamma)

# %%
"""
### Scores for Dev, Test & Train with Best Parameters for Resolution 20*20
"""

# %%
display(df_final)

# %%
"""
## Resoltuion - 2
"""

# %%
ros2_image = skimage.transform.resize(images, (len(images), 35, 35))
shape = (len(ros2_image), ros2_image.shape[1] * ros2_image.shape[2])
reshaped_images = ros2_image.reshape(shape)


xtrain, xtest, ytrain, ytest = train_test_split(reshaped_images, target, random_state=1, 
                                                stratify=target, test_size=0.2)
xtrain, xvalid, ytrain, yvalid = train_test_split(xtrain, ytrain, random_state=1, stratify=ytrain,
                                                  test_size=0.6)


C  = [0.1, 1, 10, 30, 50, 70,100] 
gamma =  [0.0001,0.001,0.01,1,10,50,100]

C_list = []
gamma_list = []
train_acc_list = []
valid_acc_list = []

for c in C:
    for gm in gamma:
        model = SVC(C = c, gamma = gm)
        model.fit(xtrain, ytrain)
        train_acc = model.score(xtrain, ytrain)
        valid_acc = model.score(xvalid, yvalid)
        
        C_list.append(c)
        gamma_list.append(gm)
        train_acc_list.append(train_acc)
        valid_acc_list.append(valid_acc)
        
df = pd.DataFrame({'C' : C_list, 'gamma' : gamma_list, 'train_accuracy' : train_acc_list, 
                   'dev_accuracy' : valid_acc_list})

max_score = df['dev_accuracy'].max()
best_params = df[df['dev_accuracy'] == max_score]

best_c = best_params['C'].values[0]
best_gamma = best_params['gamma'].values[0]

#print('resolution :',(20,20))
#print(best_c, best_gamma)


model = SVC(C = best_c, gamma = best_gamma)
model.fit(xtrain, ytrain)


train_acc = model.score(xtrain, ytrain)
valid_acc = model.score(xvalid, yvalid)
test_acc = model.score(xtest, ytest)   

#print('test accuracy is',round(test_acc, 2))

df_final = pd.DataFrame({'train_accuracy' : [train_acc],'dev_accuracy' : [valid_acc], 
                         'test_accuracy' : [test_acc]})
df_final.index = ['C ' + str(best_c) + ' gamma ' + str(best_gamma)]

df_final

# %%
plt.figure(figsize=(10, 10))
for idx, val in enumerate([100, 200, 300, 1500]):
    plt.subplot(2, 2, idx + 1)
    plt.imshow(ros2_image[val])
    plt.axis('off')
    plt.title('original')
plt.show()

# %%
"""
### Accuracy scores for different Gamma & C for Resolution 35*35
"""

# %%
display(df)

# %%
"""
### Max Dev Accuracy for Resolution 35*35
"""

# %%
print("Best Accuracy for dev is :",round(max_score, 2))

# %%
"""
### Best Parameters for Resolution 35*55
"""

# %%
print("Value for C for Max Dev Accuracy :",best_c)
print("Value for Gamme for Max Dev Accuracy :",best_gamma)

# %%
"""
### Scores for Dev, Test & Train with Best Parameters for Resolution 35*35
"""

# %%
display(df_final)

# %%
"""
## Resolution-3
"""

# %%
ros3_image = skimage.transform.resize(images, (len(images), 15, 45))
shape = (len(ros3_image), ros3_image.shape[1] * ros3_image.shape[2])
reshaped_images = ros3_image.reshape(shape)


xtrain, xtest, ytrain, ytest = train_test_split(reshaped_images, target, random_state=1, 
                                                stratify=target, test_size=0.2)
xtrain, xvalid, ytrain, yvalid = train_test_split(xtrain, ytrain, random_state=1, stratify=ytrain,
                                                  test_size=0.6)


C  = [0.1, 1, 10, 30, 50, 70,100] 
gamma =  [0.0001,0.001,0.01,1,10,50,100]

C_list = []
gamma_list = []
train_acc_list = []
valid_acc_list = []

for c in C:
    for gm in gamma:
        model = SVC(C = c, gamma = gm)
        model.fit(xtrain, ytrain)
        train_acc = model.score(xtrain, ytrain)
        valid_acc = model.score(xvalid, yvalid)
        
        C_list.append(c)
        gamma_list.append(gm)
        train_acc_list.append(train_acc)
        valid_acc_list.append(valid_acc)
        
df = pd.DataFrame({'C' : C_list, 'gamma' : gamma_list, 'train_accuracy' : train_acc_list, 
                   'dev_accuracy' : valid_acc_list})

max_score = df['dev_accuracy'].max()
best_params = df[df['dev_accuracy'] == max_score]

best_c = best_params['C'].values[0]
best_gamma = best_params['gamma'].values[0]

print('resolution :',(20,20))
print(best_c, best_gamma)


model = SVC(C = best_c, gamma = best_gamma)
model.fit(xtrain, ytrain)


train_acc = model.score(xtrain, ytrain)
valid_acc = model.score(xvalid, yvalid)
test_acc = model.score(xtest, ytest)   

print('test accuracy is',round(test_acc, 2))

df_final = pd.DataFrame({'train_accuracy' : [train_acc],'dev_accuracy' : [valid_acc], 
                         'test_accuracy' : [test_acc]})
df_final.index = ['C ' + str(best_c) + ' gamma ' + str(best_gamma)]

#df_final

# %%
plt.figure(figsize=(10, 10))
for idx, val in enumerate([100, 200, 300, 1500]):
    plt.subplot(2, 2, idx + 1)
    plt.imshow(ros3_image[val])
    plt.axis('off')
    plt.title('original')
plt.show()

# %%
"""
### Accuracy scores for different Gamma & C for Resolution 15*45
"""

# %%
display(df)

# %%
"""
### Max Dev Accuracy for Resolution 15*45
"""

# %%
print("Best Accuracy for dev is :",round(max_score, 2))

# %%
"""
### Best Parameters for Resolution 15*45
"""

# %%
print("Value for C for Max Dev Accuracy :",best_c)
print("Value for Gamme for Max Dev Accuracy :",best_gamma)

# %%
"""
### Scores for Dev, Test & Train with Best Parameters for Resolution 15*45
"""

# %%
display(df_final)

# %%
