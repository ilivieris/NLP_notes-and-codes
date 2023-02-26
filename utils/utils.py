import matplotlib.pyplot             as plt
import numpy as np
from sklearn                         import metrics


def plot_history(history):
    acc = [100.0*x for x in history.history['accuracy']]
    val_acc = [100.0*x for x in history.history['val_accuracy']]
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Testing acc')
    plt.title('Training and Testing accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Testing loss')
    plt.title('Training and Testing loss')
    plt.legend()
    plt.show()


def evaluate_model(testY, pred):
    
    pred = np.array([1 if x > 0.5 else 0 for x in pred ])

    CM = metrics.confusion_matrix(testY, pred)
    
    Accuracy             = metrics.accuracy_score(testY, pred)
    F1                   = metrics.f1_score(testY, pred)
    fpr, tpr, thresholds = metrics.roc_curve(testY, pred)
    AUC                  = metrics.auc(fpr, tpr)

    # Print results
    print('Accuracy = %.2f%%' % (100*Accuracy))
    print('AUC      = %.5f'   % AUC)
    print('F1       = %.5f'   %  F1)
    print(CM)
    print('\n')


def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    

    with open(filepath, encoding="utf8") as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word] 
                embedding_matrix[idx] = np.array(vector, dtype=np.float32)[:embedding_dim]
                
                
    return embedding_matrix