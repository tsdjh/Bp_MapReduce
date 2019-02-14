import java.io.IOException;
import org.apache.hadoop.mapreduce.Mapper;
import org.jblas.DoubleMatrix;
import org.apache.hadoop.io.Text;

public class BpMapper extends Mapper<Text,Text,DoubleMatrix,DoubleMatrix>{
    protected void map(Text key,Text value,Context context) throws IOException,InterruptedException{
        int label = Utils.getLabel(key.toString());
        DoubleMatrix x = Utils.getData(value.toString());
        DoubleMatrix y = Utils.reshapeLabel((int)label);
        Utils.fp(x);
        Utils.bp(x,y);
        context.write(Utils.w1,Utils.dw1.mul(-Utils.gd_step));
        context.write(Utils.w2,Utils.dw2.mul(-Utils.gd_step));
        context.write(Utils.b1,Utils.db1.mul(-Utils.gd_step));
        context.write(Utils.b2,Utils.db2.mul(-Utils.gd_step));
    }
}
