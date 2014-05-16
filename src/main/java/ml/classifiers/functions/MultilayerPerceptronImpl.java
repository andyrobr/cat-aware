package ml.classifiers.functions;

import weka.classifiers.functions.neural.NeuralNode;

public class MultilayerPerceptronImpl extends MultilayerPerceptron
{
    // Some member properties in MultilayerPerceptron have been changed from private to protected
    // to allow read-access from this class.

    public double[] getInputWeights(NeuralNode node)
    {
        int nodeInputs = node.getNumInputs();

        // NeuralNode stores more weights than it needs (by what I've seen, it stores 406 when only 400 are required).
        // What it does with the additional entries is fill it up with NaNs, so make sure that the 401th entry is a
        // NaN to verify that we're returning the correct number of weights.
        if (!Double.isNaN(node.weightValue(nodeInputs)))
            throw new IllegalStateException("Inconsistent number of weights.");

        double[] inputWeights = new double[nodeInputs];

        for (int i = 0; i < inputWeights.length; i++)
            inputWeights[i] = node.weightValue(i);

        return inputWeights;
    }

    public int getHiddenNodeCount()
    {
        // Neural nodes contains the nodes in the hidden layer as well as the output nodes.
        // Notice that m_neuralNodes actually has the output nodes as the first few nodes.
        return m_neuralNodes.length - m_outputs.length;
    }

    public NeuralNode[] getHiddenNodes()
    {
        // The hidden and output nodes are contained in m_neuralNodes. The output nodes are first in m_neuralNodes and
        // then the hidden nodes follow. We want to return the subarray of only hidden nodes, which is of length
        // |hidden nodes| - |output nodes|.
        int outputNodeCount = m_outputs.length;
        int hiddenNodeCount = getHiddenNodeCount();

        NeuralNode[] nodes = new NeuralNode[hiddenNodeCount];

        for (int nodeIndex = 0; nodeIndex < hiddenNodeCount; nodeIndex++) {
            nodes[nodeIndex] = (NeuralNode) m_neuralNodes[nodeIndex + outputNodeCount];
        }

        return nodes;
    }

    public NeuralNode[] getOutputNodes()
    {
        NeuralNode[] nodes = new NeuralNode[m_outputs.length];

        for (int nodeIndex = 0; nodeIndex < m_outputs.length; nodeIndex++) {
            nodes[nodeIndex] = (NeuralNode) m_neuralNodes[nodeIndex];
        }

        return nodes;
    }
}
