package PageRank;

import java.io.IOException;
import java.nio.charset.CharacterCodingException;
import java.util.HashSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

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

public class PageRankJob1Mapper extends MapReduceBase implements Mapper<LongWritable, Text, Text, Text>{

	private final String TITLE = "TITLE";
	private final String TEXT = "TEXT";
	private static final Pattern pattern = Pattern.compile("\\[.+?\\]");
	private final String TITLE_START = "<title>";
	private final String TITLE_END = "</title>";
	private final String TEXT_START = "<text";
	private final String TEXT_MID = ">";
	private final String TEXT_END = "</text>";
	private final String PIPE = "|";
	private final String SPACE = "\\s";
	private final String UNDERSCORE = "_";
	
	/**
	 * @method map takes in input key value pair as Long and Text with content of the page in value and process it to find links to 
	 * other pages in the same page. Method internally checks if the links present are valid wiki links and if so adds it to output.
	 * Output will have a key value pair as page having the link as key and other page to which link is pointed as value.
	 * @param key
	 * @param value
	 * @param output
	 * @param reporter
	 * @throws IOException
	 */
	public void map(LongWritable key, Text value, OutputCollector<Text, Text> output, Reporter reporter) throws IOException 
	{
		//parse text value in title of page and text of page.
		String pageTitle = getTextValue(value,TITLE);
		String pageText = getTextValue(value, TEXT);
		//Once the text value is parsed check if the title is valid title. Proceed only if the title is valid.
		HashSet<String> linkSet = new HashSet<String>();
	    //replace all spaces in page title with '_' . This page title will be used as key for output for mapper.
		Text pageOutputKey = new Text(pageTitle.replaceAll("\\s","_"));
		output.collect(pageOutputKey,new Text("!"));
		//if(!pageTitle.contains(":"))
		{
		//Look for pattern in the text iteratively.
			Matcher m = pattern.matcher(pageText);
			while(m.find())
			{
				//find the string matching the pattern.
				String pageLinks = m.group();
				//only if the link specified is valid Wiki link.
					//once pattern matches filter the link of the string which may b of either form in wiki page.
					String otherPages = filterLink(pageLinks);
					if(otherPages.isEmpty() || otherPages == null)
						continue;
					
					//Add the record if valid to output with page and outgoing link.
					if(pageOutputKey.toString().equals(otherPages))
						continue;
					if(!linkSet.add(otherPages))
						continue;
					output.collect(new Text(otherPages),pageOutputKey);
				
			}
		}
			//if a page has no outlinks but page exists so needs to be added.
			if(linkSet.size() == 0)
				output.collect(new Text(""), pageOutputKey);
	}
	/**
	 * @method filterLink to filter page name of the link pointed to. Wiki Link may contain pages with links as [[link ]] or the 
	 * other way.
	 * @param wikiString
	 * @return
	 */
	String filterLink(String wikiString)
	{
		String filteredLink = "";
		if(wikiString.length()>0)
		{
			int startIndex = -1;
			if(wikiString.startsWith("[["))
				startIndex = 2;
			else
				startIndex = 1;
			//determine end of link. If the link is terminated by "]" or else by "|" find the min or else by hash "#". determine that
			// sequentially and reduce the end index to extract the substring of the string.
			int endIndex = wikiString.indexOf("]");
			//check for PIPE as end.
			endIndex = wikiString.indexOf(PIPE) > 0 ? wikiString.indexOf(PIPE) : endIndex;
			//check for hash as end.
			filteredLink = wikiString.substring(startIndex,endIndex);
 			filteredLink =  filteredLink.replaceAll(SPACE, UNDERSCORE);
 			filteredLink = filteredLink.trim();
		}
		return filteredLink;
	} 
	
	/**
	 * @method getTextValue to get text value based on tag passed as argument. If tag passed is TITLE then it gets the title
	 * of page else it gets the text between text tags. 
	 * @param value
	 * @return
	 * @throws CharacterCodingException 
	 */
	String getTextValue(Text value,String tag) throws CharacterCodingException
	{
		String textValue ="";
		int start;
		int end;
		if(TITLE.equals(tag))
		{
			//start value for tag "<title>"
			start = value.find(TITLE_START);
			//look for tag "<\title>" only after "<title>".
			end = value.find(TITLE_END,start);
			if(start == -1 || end ==-1)
				return new String("");
			textValue = Text.decode(value.getBytes(),start+TITLE_START.length(),end -start-TITLE_START.length());
		}else
		{
			//start value for tag "<text"
			start = value.find(TEXT_START);
			//search for closing tag ">" this after "<text" has been found since it will have properties.
			start = value.find(TEXT_MID,start);
			//look for "<\text>"
			end = value.find(TEXT_END,start);
			if(start == -1 || end == -1) 
	            return new String("");
			textValue = Text.decode(value.getBytes(),start+1,end-start-1);
		}
		return textValue;
	}
}
