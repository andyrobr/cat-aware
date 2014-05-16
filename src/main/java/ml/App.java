package ml;

import ml.classifiers.functions.MultilayerPerceptronImpl;
import weka.classifiers.functions.neural.NeuralNode;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileReader;

public class App
{
    public static void main( String[] args ) throws Exception
    {
        // We want to start by creating a new training instance by supplying a
        // reader for the stock training ARFF file and setting the
        // number of attributes to use.
        Instances trainingData = new Instances(
                new BufferedReader(
                        new FileReader(
                                "test_data/unclassified.arff"
                        )
                )
        );

        trainingData.setClassIndex( trainingData.numAttributes() - 1 );

        // We want to test with the separate data so we open a separate
        // examples ARFF file to test against.
        Instances testingData = new Instances(
                new BufferedReader(
                        new FileReader(
                                "test_data/faces.arff"
                        )
                )
        );

        testingData.setClassIndex( testingData.numAttributes() - 1 );

        // The method toSummaryString prints a summary of a set of
        // training or testing instances.
        String summary = trainingData.toSummaryString();
        int sampleCount = trainingData.numInstances();
        int attributeCountPerSample = trainingData.numAttributes();

        System.out.println("Number of attributes in model = " + attributeCountPerSample);
        System.out.println("Number of samples = " + sampleCount);
        System.out.println("Summary: " + summary);
        System.out.println();

        // Go through each sample and pluck each attribute and value for every sample.
        for (int i = 0; i < sampleCount; i++)
        {
            Instance instance = testingData.instance(i);

            for (int j = 0; j < instance.numAttributes(); j++)
            {
                Attribute attribute = instance.attribute(j);
                System.out.printf("%s: %s\n", attribute.name(), instance.value(j));
            }
        }

        // Weka builds the neural network for us, all we need to specify is the number
        // of nodes in the hidden layer and the input data.
        MultilayerPerceptronImpl mp = new MultilayerPerceptronImpl();

        mp.setHiddenLayers("4");
        mp.buildClassifier(trainingData);

        // Go through each of the samples in the testing set and run them through
        // the model built by Weka.
        for (int i = 0; i < testingData.numInstances(); i++)
        {
            double pred = mp.classifyInstance(testingData.instance(i));

            System.out.print("Given value: " + testingData.classAttribute().
                    value((int) testingData.instance(i).classValue()));

            System.out.print(". Predicted value: " +
                    testingData.classAttribute().value((int) pred));

            // I'm still not sure what this represents, but I assume it has to do with
            // the probability for each node in the output layer given the instance.
            // (Recall we have three nodes in the output layer, one for each class).
            System.out.println(". Distribution: " +
                    ArrayUtils.joinDoubles(mp.distributionForInstance(testingData.instance(i))));
        }

        // Go through each of the nodes in the hidden layer...
        for (NeuralNode node : mp.getHiddenNodes())
        {
            System.out.println("ID: " + node.getId());

            System.out.println("Number of inputs: " + node.getNumInputs());
            System.out.println("Number of outputs: " + node.getNumOutputs());

            double[] weights = mp.getInputWeights(node);

            System.out.println("Number of weights: " + weights.length);
        }

        // ...and output layer.
        for (NeuralNode node : mp.getOutputNodes())
        {
            System.out.println("ID: " + node.getId());

            System.out.println("Number of inputs: " + node.getNumInputs());
            System.out.println("Number of outputs: " + node.getNumOutputs());

            double[] weights = mp.getInputWeights(node);

            System.out.println("Number of weights: " + weights.length);
        }
    }
}