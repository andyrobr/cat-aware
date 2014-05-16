package ml.test;

import ml.classifiers.functions.MultilayerPerceptronImpl;
import weka.core.Instance;

public interface ModelProcessor
{
    public void process(MultilayerPerceptronImpl classifier, Instance instance, int total, int index) throws Exception;
}