package PageRank;
import java.io.IOException;
import java.util.Iterator;

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

public class PageRankJob2Reducer extends MapReduceBase implements Reducer<Text,Text,Text,Text>
{
	/**
	 * @method reduce overriden to collect all the records produced by mapper as outlinks.
	 * @param TEXT
	 * @param Iterator<Text>
	 * @param OutputCollector<Text,Text> output of mapper instance stored as key value pair as Text, Text.
	 * @param Reporter
	 */
	public void reduce(Text key, Iterator<Text> values, OutputCollector<Text, Text> output, Reporter reporter) throws IOException 
    {
		String pageValue;
		String pageLink="";
		while(values.hasNext())
		{
			pageValue=values.next().toString().trim();
			if(!pageValue.isEmpty())
				pageLink= pageLink + pageValue + "\t";
		}
		pageLink=pageLink.trim();
		output.collect(key,new Text(pageLink));
    }

  }