package PageRank;
import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
/**
 * 
 * @author atgarg iyadav
 *
 */

public class PageRankJob5Reducer extends MapReduceBase implements Reducer<FloatWritable,Text,Text,FloatWritable> 
{
	/**
	 * @method reduce to put sorted output and multiplying with -1 again since rank was multiplied in mapper.
	 */
	public void reduce(FloatWritable key, Iterator<Text> values, OutputCollector<Text, FloatWritable> output, Reporter reporter) throws IOException 
    {
		float rank = key.get();
		rank = rank*(-1);
		while(values.hasNext())
		{
			output.collect(values.next(), new FloatWritable(rank));
		}
    }  
}
