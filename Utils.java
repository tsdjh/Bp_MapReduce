import org.jblas.DoubleMatrix;
import java.util.function.Function;
import java.lang.Math;

public class Utils {
    private static final int INPUT_DIMENSION = 784;
    private static final int HIDDENLAYER_NODECOUNT = 32;
    private static final int CATEGORY_COUNT = 10;
    private static final double W1_INITMAX = Math.sqrt(6 / (INPUT_DIMENSION + HIDDENLAYER_NODECOUNT));
    private static final double W2_INITMAX = Math.sqrt(6 / (CATEGORY_COUNT + HIDDENLAYER_NODECOUNT));
    public static DoubleMatrix w1 = DoubleMatrix.rand(HIDDENLAYER_NODECOUNT,INPUT_DIMENSION).mul(2 * W1_INITMAX).sub(W1_INITMAX);
    public static DoubleMatrix b1;
    public static DoubleMatrix w2 = DoubleMatrix.rand(CATEGORY_COUNT,HIDDENLAYER_NODECOUNT).mul(2 * W2_INITMAX).sub(W2_INITMAX);
    public static DoubleMatrix b2;
    private static DoubleMatrix a1;
    private static DoubleMatrix a2;
    public static DoubleMatrix dw1;
    public static DoubleMatrix dw2;
    public static DoubleMatrix db1;
    public static DoubleMatrix db2;
    public static double gd_step = 0.001;

    public static int getLabel(String key){
        return (int)Double.parseDouble(key);
    }

    public static DoubleMatrix getData(String value){
        String[] data = value.strip().split("\t");
        double[] x = new double[data.length];
        for (int i = 0;i < data.length;i++){
            x[i] = Double.parseDouble(data[i]);
        }
        return new DoubleMatrix(x);
    }

    private static DoubleMatrix map(DoubleMatrix origin, Function<Double,Double> mapper){
        double[] array = origin.toArray();
        for (int i = 0;i < array.length;i++){
            array[i] = mapper.apply(array[i]);
        }
        return new DoubleMatrix(array);
    }

    private static double sigmoid(double x){
        return 1 / (1 + Math.exp(-x));
    }

    public static void fp(DoubleMatrix x){
        DoubleMatrix z1 = w1.mmul(x).add(b1);
        a1 = map(z1,Math::tanh);
        DoubleMatrix z2 = w2.mmul(a1).add(b2);
        a2 = map(z2,Utils::sigmoid);
    }

    public static void bp(DoubleMatrix x,DoubleMatrix y){
        DoubleMatrix dz2 = a2.sub(y);
        dw2 = dz2.mul(a1);
        db2 = dz2;
        DoubleMatrix dz1 = w2.transpose().mmul(dz2).mul(a1.mul(a1.neg()).add(1));
        dw1 = dz1.mul(x);
        db1 = dz1;
    }

    public static DoubleMatrix reshapeLabel(int label){
        double[] y = new double[CATEGORY_COUNT];
        y[label] = 1.0;
        return new DoubleMatrix(y);
    }


}
