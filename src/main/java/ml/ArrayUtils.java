package ml;

public class ArrayUtils
{
    public static String joinDoubles(double[] doubles)
    {
        StringBuilder buffer = new StringBuilder();

        for (double d : doubles)
        {
            buffer.append(d).append(' ');
        }

        return buffer.toString();
    }
}
