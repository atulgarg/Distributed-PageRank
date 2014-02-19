package PageRank;
import java.io.IOException;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;


public class PageRankJob4Mapper extends MapReduceBase implements Mapper<LongWritable, Text, Text, Text>
{
	public static final String ITERATION_COUNT ="ITERATION_COUNT";
	public static String TOTAL_COUNT = "TOTAL_COUNT";
	public static long N;
	private Integer iter;
	/**
	 * @method map over ridden from MapReduce Base.
	 */
	public void configure(JobConf job) 
	{
	    iter = Integer.parseInt(job.get(ITERATION_COUNT));
	    N = Long.parseLong(job.get(TOTAL_COUNT));;
	}
	/**
	 * 
	 */
	public void map(LongWritable key, Text value, OutputCollector<Text, Text> output, Reporter reporter) throws IOException {
		int pageIndex=value.find("\t");
	    int rankIndex=value.find("\t",pageIndex+1);
	    
	    String page=Text.decode(value.getBytes(),0,pageIndex);
	    String pagewithrank;
	    String pageLinks="";
	    if(iter.equals(1))
	    {
	    	float f = 1.0f/N;
	    	pagewithrank = page + "\t" + f + "\t";
	    	//check if more links are there in the page.
	    	if(value.getLength()-(pageIndex+1)>0)
	    	{
	    		pageLinks=Text.decode(value.getBytes(),pageIndex+1,value.getLength()-(pageIndex+1));
	    		pageLinks = pageLinks.trim();
	    	}
	    }
	    else
	    {
	    	pagewithrank=Text.decode(value.getBytes(),0,rankIndex+1);
	    	if(rankIndex !=-1)
	    		{
	    		pageLinks=Text.decode(value.getBytes(),rankIndex+1,value.getLength()-(rankIndex+1));
	    		pageLinks = pageLinks.trim();
	    		}
	    }
	    
	    output.collect(new Text(page), new Text("!"));
	    
	    String allPages[]=pageLinks.split("\t");
	    int totalLinks=allPages.length;
	
	    if(totalLinks == 1 && allPages[0].trim().isEmpty())
	    {
	    	;
	    }
	    else
	    {
	    	for(String pages :allPages)
	    	{
	    		Text outgoingLink=new Text(pagewithrank + totalLinks);
	    		output.collect(new Text(pages),outgoingLink);
	    	}
	    }
	    output.collect(new Text(page), new Text("|"+pageLinks));
	    
	}

}