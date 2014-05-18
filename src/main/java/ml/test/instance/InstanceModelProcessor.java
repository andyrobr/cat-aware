package ml.test.instance;

import ml.classifiers.functions.MultilayerPerceptronImpl;
import ml.test.ModelProcessor;
import weka.core.Instance;

public class InstanceModelProcessor implements ModelProcessor
{
    private final double activationThreshold;

    /**
     *
     * @param activationThreshold Any node with a value greater than or equal to the activationThreshold would be
     *                            considered to have fired.
     */
    public InstanceModelProcessor(double activationThreshold)
    {
        this.activationThreshold = activationThreshold;
    }

    @Override
    public void process(MultilayerPerceptronImpl mp, Instance instance, int total, int index) throws Exception
    {
        double[] oldHiddenValues, newHiddenValues;

        oldHiddenValues = mp.getHiddenValues();

        mp.distributionForInstance(instance);

        newHiddenValues = mp.getHiddenValues();

        for (int i = 0; i < oldHiddenValues.length; i++)
        {
            if (Math.abs(newHiddenValues[i] - oldHiddenValues[i]) > 0.000001) {
                System.out.println("The values do not match.");
                break;
            }
        }
    }
}
