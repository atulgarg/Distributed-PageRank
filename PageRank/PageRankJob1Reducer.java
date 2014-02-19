package PageRank;
import java.io.IOException;
import java.util.ArrayList;
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

public class PageRankJob1Reducer extends MapReduceBase implements Reducer<Text,Text,Text,Text>
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
		//set initial rank as one for each of the page.
    	boolean isValid=false;
    	ArrayList<String> pageLists = new ArrayList<String>();
    	String pages=key.toString();
        while(values.hasNext())
        {
        	String pageValue=values.next().toString();
	    	 if(pageValue.equals("!"))	    		 
	    	 {
	    		 //to check for red links.
	    		 isValid=true;
	    		 output.collect(new Text(key.toString()), new Text(""));
	    		 continue; 
	    	 }
	    	 pageLists.add(pageValue);
	    	 
	   }
        
        if(isValid == true)
        {
        	int listSize = pageLists.size();
        		for(int i=0;i<listSize;i++)
        		{
        			output.collect(new Text(pageLists.get(i)), new Text(pages));
        		}
        	
        }
        
    }
}