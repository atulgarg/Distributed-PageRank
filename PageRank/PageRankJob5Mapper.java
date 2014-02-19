package PageRank;
import java.io.IOException;

import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;


public class PageRankJob5Mapper extends MapReduceBase implements Mapper<LongWritable, Text, FloatWritable,Text> 
{
	public static long N;
	public static float threshold = 0.0f;
	public void configure(JobConf job) 
	{
	    N = Long.parseLong(job.get(PageRankJob4Reducer.TOTAL_COUNT));;
	    threshold = 5.0f/N;
	}
	/**
	 * 
	 */
    public void map(LongWritable key, Text value, OutputCollector<FloatWritable,Text> output, Reporter reporter) throws IOException {
        String[] pagewithRank = new String[2];
        int PageIndex = value.find("\t");
        int RankIndex = value.find("\t", PageIndex + 1);
        
        // no tab after rank (when there are no links)
        int end;
        if (RankIndex == -1) {
            end = value.getLength() - (PageIndex + 1);
        } else {
            end = RankIndex - (PageIndex + 1);
        }
        
        pagewithRank[0] = Text.decode(value.getBytes(), 0, PageIndex);
        pagewithRank[1] = Text.decode(value.getBytes(), PageIndex + 1, end);
        
        float parseFloat = Float.parseFloat(pagewithRank[1]);
        
        Text page = new Text(pagewithRank[0]);
        FloatWritable finalRank = new FloatWritable((-1)*parseFloat);
        if(parseFloat>threshold)
        	output.collect(finalRank,page);
    }
    
    public static class DecreasingComparator extends WritableComparator {
        protected DecreasingComparator() {
            super(Text.class, true);
        }
        public int compare(WritableComparable w1, WritableComparable w2) {
            LongWritable key1 = (LongWritable) w1;
            LongWritable key2 = (LongWritable) w2;          
            return -1*key1.compareTo(key2);
        }
    }
}
