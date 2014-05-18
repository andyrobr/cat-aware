package ml.test.instance;

import ml.classifiers.functions.MultilayerPerceptronImpl;
import ml.test.ModelProcessor;
import weka.core.Instance;

public abstract class AbstractModelProcessor implements ModelProcessor
{
    public abstract void reset(int total);
    public abstract void summary();

    public abstract void process(MultilayerPerceptronImpl classifier, Instance instance) throws Exception;

    public void process(MultilayerPerceptronImpl classifier, Instance instance, int total, int index) throws Exception
    {
        if (index == 0) reset(total);

        process(classifier, instance);

        if (index + 1 == total) summary();
    }
}
