package PageRank;
import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;


public class PageRankJob3Reducer extends MapReduceBase implements Reducer<Text,Text,Text,Text>
{
	public void reduce(Text key, Iterator<Text> values, OutputCollector<Text, Text> output, Reporter reporter) throws IOException 
    {
		String keyValue = "N=";//"count";
		Integer count = 0;
		//evaluate total count of pages.
		while(values.hasNext())
		{
			values.next();
			count++;
		}
		output.collect(new Text(keyValue+count),new Text());//new Text(count+""));
    }

}
