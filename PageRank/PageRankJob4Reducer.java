package PageRank;
import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;


public class PageRankJob4Reducer extends MapReduceBase implements Reducer<Text, Text, Text, Text> 
{
public final static String TOTAL_COUNT = "TOTAL_COUNT";
public final static float PRECISION_VALUE = 0.00000000001f;
private static Long N;

public void configure(JobConf job) {
N = Long.parseLong(job.get(TOTAL_COUNT));
}

private static final float DAMPING_FACTOR = 0.85f;

/**
* 
*/
public void reduce(Text page, Iterator<Text> values, OutputCollector<Text, Text> out, Reporter reporter) throws IOException 
{
	String pageValue;
	String splitPage[];
	String outgoingPage="";
	float totRank = 0;
	boolean notOne = false;
	boolean deadPage = false;
	while(values.hasNext())
	{
		pageValue=values.next().toString().trim();
		if(pageValue.equals("!"))
		{
			if(!values.hasNext() && !notOne)
			{
				deadPage = true;
				float newRank=((1-DAMPING_FACTOR)/N);
				out.collect(new Text(page),new Text(newRank+"\t"));
			}
			continue;
		}
		else
		{
			notOne = true;
			if(pageValue.startsWith("|"))
			{
				int pageValueIndex=pageValue.indexOf("|");
				outgoingPage=pageValue.substring(pageValueIndex+1);
				continue;
			}
			splitPage=pageValue.split("\t");
			float oldRank = Float.valueOf(splitPage[1]);

			//evaluate number of outlinks
			float outLink = Float.valueOf(splitPage[2]);

			//if outlinks are more than 0
			if(Math.abs(outLink-0.0f)>PRECISION_VALUE)
				totRank = totRank + (oldRank/outLink);
		}
	}
    
    if(!deadPage)
    {
    	float newRank = (DAMPING_FACTOR)*totRank+((1.0f-DAMPING_FACTOR)/N);
    		out.collect(page,new Text(newRank+"\t"+outgoingPage));
    }
	 
 }
}
