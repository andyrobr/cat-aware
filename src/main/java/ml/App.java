package ml;

import ml.classifiers.functions.MultilayerPerceptronImpl;
import ml.test.ModelProcessor;
import ml.test.instance.ConsoleClassificationProcessor;
import ml.test.instance.InstanceModelProcessor;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileReader;

public class App
{
    private static final int HIDDEN_NODE_COUNT = 4;
    private static final double ACTIVATION_THRESHOLD = 0.5;

    public static ModelProcessor[] processors = {

            // Andy's model processor.
            new InstanceModelProcessor(ACTIVATION_THRESHOLD),

            // Insert any additional model processors.

            new ConsoleClassificationProcessor()
    };

    public static void main( String[] args ) throws Exception
    {
        // We want to start by creating a new training instance by supplying a
        // reader for the stock training ARFF file and setting the
        // number of attributes to use.
        Instances trainingData = new Instances(
                new BufferedReader(
                        new FileReader(
                                "test_data/faces.arff"
                        )
                )
        );

        trainingData.setClassIndex( trainingData.numAttributes() - 1 );

        // We want to test with the separate data so we open a separate
        // examples ARFF file to test against.
        Instances testingData = new Instances(
                new BufferedReader(
                        new FileReader(
                                "test_data/unclassified.arff"
                        )
                )
        );

        testingData.setClassIndex( testingData.numAttributes() - 1 );

        // Weka builds the neural network for us, all we need to specify is the number
        // of nodes in the hidden layer and the input data.
        MultilayerPerceptronImpl mp = new MultilayerPerceptronImpl();

        mp.setHiddenLayers("" + HIDDEN_NODE_COUNT);
        mp.buildClassifier(trainingData);

        int totalSamples = testingData.numInstances();

        // Go through each instance in the testing set and call each of the processors.
        for (int sampleIndex = 0; sampleIndex < totalSamples; sampleIndex++)
        {
            for (ModelProcessor processor : processors)
            {
                processor.process(mp, testingData.instance(sampleIndex), totalSamples, sampleIndex);
            }
        }
    }
}