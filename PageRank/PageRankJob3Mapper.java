package PageRank;
import java.io.IOException;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
/**
 * 
 * @author atgarg iyadav
 *
 */
public class PageRankJob3Mapper extends MapReduceBase implements Mapper<LongWritable, Text, Text, Text> 
{
	/**
	 * Mapper Function overridden.
	 */
	public void map(LongWritable key,Text value,OutputCollector<Text,Text> output, Reporter reporter) throws IOException
	{
		//For each key output one.
		output.collect(new Text("1"),new Text(key.toString()));
		
	}
}
