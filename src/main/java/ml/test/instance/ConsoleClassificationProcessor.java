package ml.test.instance;

import ml.classifiers.functions.MultilayerPerceptronImpl;
import weka.core.Instance;

public class ConsoleClassificationProcessor extends AbstractModelProcessor
{
    private int correct = 0;
    private int total = 0;

    @Override
    public void reset(int total)
    {
        this.total = total;
    }

    @Override
    public void summary()
    {
        System.out.printf("Result: %d / %d (%.01f%%) correct", correct, total, (100.0 * correct) / total);
    }

    @Override
    public void process(MultilayerPerceptronImpl classifier, Instance instance) throws Exception
    {
        double pred = classifier.classifyInstance(instance);

        String actual = instance.classAttribute().value((int) instance.classValue()),
               result = instance.classAttribute().value((int) pred);

        System.out.printf("Given value: %s. Predicted value: %s\n", actual, result);

        if (actual.equals(result)) correct++;
    }
}
