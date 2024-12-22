import numpy


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate
        self.w_ih = numpy.random.normal(
            0.0, pow(self.input_nodes, -0.5), (self.hidden_nodes, self.input_nodes)
        )
        self.w_ho = numpy.random.normal(
            0.0, pow(self.hidden_nodes, -0.5), (self.output_nodes, self.hidden_nodes)
        )
        self.sigmoid = lambda x: 1 / (1 + numpy.exp(-x))

    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # weights
        hidden_inputs = numpy.dot(self.w_ih, inputs)
        hidden_outputs = self.sigmoid(hidden_inputs)
        final_inputs = numpy.dot(self.w_ho, hidden_outputs)
        final_outputs = self.sigmoid(final_inputs)

        # errors
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.w_ho.T, output_errors)

        self.w_ho += self.compute_weight_delta(
            hidden_outputs, final_outputs, output_errors
        )
        self.w_ih += self.compute_weight_delta(inputs, hidden_outputs, hidden_errors)

    def compute_weight_delta(
        self,
        inputs,
        outputs,
        errors,
    ):
        return self.learning_rate * numpy.dot(
            (errors * outputs * (1 - outputs)), numpy.transpose(inputs)
        )

    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.w_ih, inputs)
        hidden_outputs = self.sigmoid(hidden_inputs)
        final_inputs = numpy.dot(self.w_ho, hidden_outputs)
        final_outputs = self.sigmoid(final_inputs)
        return final_outputs
