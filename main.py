import numpy

from neural_network import NeuralNetwork


input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.2

n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

training_data_file = open("./data/mnist/mnist_train.csv", "r")
training_data_list = training_data_file.readlines()
training_data_file.close()

for record in training_data_list:
    all_values = record.split(",")
    scaled_input = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    targets = numpy.zeros(output_nodes) + 0.01
    target_int = int(all_values[0])
    targets[target_int] = 0.99
    n.train(scaled_input, targets)

print("Training complete")


test_data_file = open("./data/mnist/mnist_test.csv", "r")
test_data_list = test_data_file.readlines()
test_data_file.close()

scorecard = []

for record in test_data_list:
    all_values = record.split(",")
    scaled_input = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    outputs = n.query(scaled_input)
    label = numpy.argmax(outputs)
    scorecard.append(1) if label == int(all_values[0]) else scorecard.append(0)

print(f"Performance: {sum(scorecard) / len(scorecard) * 100}%")
