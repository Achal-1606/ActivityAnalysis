"""
Given a training log file, plot something.
"""
import csv
import matplotlib.pyplot as plt


def main(training_log):
    with open(training_log) as fin:
        reader = csv.reader(fin)
        next(reader, None)  # skip the header
        accuracies = []
        train_accuracies = []
        loss_val = []
        cnn_benchmark = []  # this is ridiculous
        for epoch, acc, loss, val_acc, val_loss in reader:
            accuracies.append(float(val_acc))
            train_accuracies.append(float(acc))
            loss_val.append(float(val_loss))
            # top_5_accuracies.append(float(val_top_k_categorical_accuracy))
            cnn_benchmark.append(0.65)  # ridiculous

        plt.plot(accuracies, label='Val Acc')
        plt.plot(loss_val, label='Val Loss')
        plt.plot(train_accuracies, label='Train Acc')
        # plt.plot(top_5_accuracies)
        plt.plot(cnn_benchmark, label='CNN Benchmark')
        plt.legend(loc='best')
        plt.grid()
        plt.show()


if __name__ == '__main__':
    training_log = 'data/logs/lstm-training-1555207078.5300581.log'
    main(training_log)
