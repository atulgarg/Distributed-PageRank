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

public class PageRankJob2Mapper extends MapReduceBase implements Mapper<LongWritable , Text, Text, Text> {
	/**
	 * Mapper Function overridden.
	 */
	public void map(LongWritable key, Text value, OutputCollector<Text, Text> output, Reporter reporter) throws IOException 
	{
		int pageIndex=value.find("\t");
	    String page1=Text.decode(value.getBytes(),0,pageIndex);
	    String page2=Text.decode(value.getBytes(),pageIndex+1,value.getLength()-pageIndex-1);
	 	output.collect(new Text(page1),new Text(page2.trim()));
	}

}