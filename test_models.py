from keras.models import load_model
import numpy as np
from numpy import array
from sklearn import preprocessing
from random import randint
import pickle
import time
import heapq

# call tests, record data, display and save
def main():
    # load pickled training and test data
    x_train, y = pickle.load(open("piece_count_data.p", "rb"))
    # have to scale y data to -1, 1 range
    num_values = len(y)
    y_min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    y_true = y_min_max_scaler.fit_transform(y)

    # first test results from piece count model
    piece_count_model = load_model("piece_count_5_layer.h5")
    scores = piece_count_model.evaluate(x_train, y_true)

    y_pred = piece_count_model.predict(x_train)

    times = time_move_evaluation(piece_count_model, x_train)
    # check 5000 slices
    pct_correct = percent_correct_move(y_true, y_pred, 5000)
    pct_top_three = top_three(y_true, y_pred, 5000)
    print("data from piece count model")
    print("Mean squared error for model:\t {}\n number of positions tested:\t {}\ntotal time taken to evaluate positions:\t {}\navg time spent per move:\t {}\n% of time best move is scored highest:\t {}\n% of time best move is in top 3 choices:\t {}".format(scores[0], times[1], times[0], times[0]/num_values, pct_correct, pct_top_three))


    x_train, y = pickle.load(open("full_data_set.p", "rb"))
    # have to scale y data to -1, 1 range
    num_values = len(y)
    y_min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    y_true = y_min_max_scaler.fit_transform(y)

    # first test results from piece count model
    piece_count_model = load_model("chess_model.h5")
    scores = piece_count_model.evaluate(x_train, y_true)
    y_pred = piece_count_model.predict(x_train)
    times = time_move_evaluation(piece_count_model, x_train)
    # check 5000 slices
    pct_correct = percent_correct_move(y_true, y_pred, 5000)
    pct_top_three = top_three(y_true, y_pred, 5000)
    print("data from piece position model")
    print("Mean squared error for model:\t {}\n number of positions tested:\t {}\ntotal time taken to evaluate positions:\t {}\navg time spent per move:\t {}\n% of time best move is scored highest:\t {}\n% of time best move is in top 3 choices:\t {}".format(scores[0], times[1], times[0], times[0]/num_values, pct_correct, pct_top_three))
    return


# takes a keras network model and a set of input vectors of arbitrary length > 1
# returns a list containing the time taken to evaluate the inputs
# the number of input vectors, and the average time taken to evaluate each input vector
def time_move_evaluation(model, x_test):
    start = time.time()
    model.predict(x_test)
    end = time.time()
    total_time = end - start
    time_per_move = total_time/int(len(x_test))
    return [total_time, len(x_test), time_per_move]


# takes as input two np arrays of output values, true and predicted
# takes a series of random slices of those arrays and checks if the
# maximum values of each slice is at the same position for
# the true and predicted data
#
# also takes value n >= 1 for number of slices to check
# this corresponds to the percentage of times that
# the move with the maximum predicted score will correspond to the
# move with the maximum true score (true score = computer evaluated)
def percent_correct_move(y_true, y_pred, num_tests):
    # both inputs are np arrays, easier to index w/ lists
    y_true_list = list(y_true)
    y_pred_list = list(y_pred)

    number_correct = 0
    for x in range(num_tests):
        # tests 10 values
        slice_start = randint(0, len(y_true_list) - 50 )
        test_slice = y_true_list[slice_start:slice_start + 50]
        index = test_slice.index(max(test_slice))

        pred_test_slice = y_pred_list[slice_start:slice_start + 50]
        pred_index = pred_test_slice.index(max(pred_test_slice))
        if index == pred_index:
            number_correct += 1

    percent_correct = number_correct / num_tests
    return percent_correct

# returns the pct of time our model lists the best move in it's top 3 values
def top_three(y_true, y_pred, num_tests):
    # both inputs are np arrays, easier to index w/ lists
    y_true_list = list(y_true)
    y_pred_list = list(y_pred)

    number_correct = 0
    for x in range(num_tests):
        # tests 10 values
        slice_start = randint(0, len(y_true_list) - 50 )
        test_slice = y_true_list[slice_start:slice_start + 50]
        index = test_slice.index(max(test_slice))

        pred_test_slice = y_pred_list[slice_start:slice_start + 50]
        top_3 = heapq.nlargest(3, pred_test_slice)
        for x in top_3:
            pred_index = pred_test_slice.index(x)
            if pred_index == index:
                number_correct += 1
                break

    percent_correct = number_correct / num_tests
    return percent_correct

if __name__ == "__main__":
    main()
