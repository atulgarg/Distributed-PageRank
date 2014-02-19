package PageRank;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.text.DecimalFormat;
import java.text.NumberFormat;

import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.hadoop.mapred.TextOutputFormat;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobClient;

public class PageRank {
		private static NumberFormat nf = new DecimalFormat("0");
		 
		    public static void main(String[] args) throws Exception 
		    {
		    Configuration conf = new Configuration();
		    PageRank pageRanking = new PageRank();
	    	//Append Bucket Name as from the Argument.
	    	if(args.length != 1)
	    		return ;
	    	
	    	String bucketName = args[0];		//args[0] will specify bucket.
	        //Parse the XML and generate outlink Graph from XML -- JOB1 and JOB2
	    	String path = "s3://" + bucketName + "/";
	    	//pageRanking.runXmlParsing("wiki/in/", path+"tmp/job10");
	    	pageRanking.runXmlParsing("s3://spring-2014-ds/data/enwiki-latest-pages-articles.xml", path + "tmp/job10");
	        pageRanking.getOutLinks(path + "tmp/job10", path + "tmp/job11");
	        
	        //Job3 Input is result of Job2, counts total number of pages.
	        pageRanking.evaluateCount(path +"tmp/job11", path +"tmp/job2");
	        
	        //Evaluate Page Rank -- Iteration 1.
	        pageRanking.evaluatePageRank(path +"tmp/job11", path +"tmp/job31",1,path);
	        
	        //Sorting for First Iteration and moving results.
	        pageRanking.runRankOrdering(path +"tmp/job31",path +"tmp/job40",path);
	        
        	
	        //Output for each iteration goes in /tmp/job3+(iteration+1)
	        int iteration = 1;
	        for (; iteration <= 6; iteration++) 
	        {
	            //Job 3: Calculate new rank for each  of iteration.
	            pageRanking.evaluatePageRank(path +"tmp/job3" + nf.format(iteration), path +"tmp/job3"+nf.format(iteration + 1),iteration+1,path);
	        }
	        //Evaluate Page rank for each of iteration.
	        pageRanking.evaluatePageRank(path + "tmp/job37", path + "tmp/job38/",iteration+1,path);
	        //Sorting for 8th Iteration.
	        pageRanking.runRankOrdering(path + "tmp/job3" + nf.format(iteration+1), path + "tmp/job48/",path);
	        
	        //Move Output of Job2 to Results Directory.
	        Path srcPath = new Path(path+"tmp/job11/");
        	Path destinationPath = new Path(path + "results/PageRank.outlink.out");
        	FileSystem fs = srcPath.getFileSystem(conf);
        	FileUtil.copyMerge(fs, srcPath, fs, destinationPath, false, conf, null);
	        
	        //Move Output of Job4 Iteration 1
	        srcPath = new Path(path  + "tmp/job40/part-00000");
        	destinationPath = new Path(path + "results/PageRank.iter1.out");
        	fs.rename(srcPath, destinationPath);
	        
        	//Move Output of Job3 to Results Directory.
	        srcPath = new Path(path  + "tmp/job2/part-00000");
        	destinationPath = new Path(path + "results/PageRank.n.out");
        	fs.rename(srcPath, destinationPath);
	        
        	//Move Output of Job 4 Iteration 8
	        srcPath = new Path(path  + "tmp/job48/part-00000");
        	destinationPath = new Path(path + "results/PageRank.iter8.out");
        	fs.rename(srcPath, destinationPath);
	              
	      }
	    /**
	     * JOB 1:
	     * @method runXmlParsing to Parse Given XML after reading file from specified input path. Method writes the output
	     * of reducer to specified output path.
	     * @param inputPath	Path to read input files for mapper.
	     * @param outputPath	Path to output, output files for reducer.
	     * @throws IOException
	     */
	    public void runXmlParsing(String inputPath, String outputPath) throws IOException {
	        JobConf conf = new JobConf(PageRank.class);
	 
	        FileInputFormat.setInputPaths(conf, new String(inputPath));
	        // Mahout class to Parse XML + config
	        conf.setInputFormat(XmlInputFormat.class);
	        conf.set(XmlInputFormat.START_TAG_KEY, "<page>");
	        conf.set(XmlInputFormat.END_TAG_KEY, "</page>");
	        // Our class to parse links from content.
	        conf.setMapperClass(PageRankJob1Mapper.class);
	 
	        FileOutputFormat.setOutputPath(conf, new Path(outputPath));
	        conf.setOutputFormat(TextOutputFormat.class);
	        conf.setOutputKeyClass(Text.class);
	        conf.setOutputValueClass(Text.class);
	        conf.setReducerClass(PageRankJob1Reducer.class);
	 
	        JobClient.runJob(conf);
	    }
	    /**
	     * JOB 2:
	     * @method evaluateCount to evaluate total count of pages in given xml. This is executed as second job
	     * for evaluating Page Rank.
	     * @param inputPath
	     * @param outputPath
	     * @throws IOException
	     */
	    public void getOutLinks(String inputPath, String outputPath) throws IOException {
	    	JobConf conf = new JobConf(PageRank.class);
	        FileInputFormat.setInputPaths(conf, new String(inputPath));
	        conf.setInputFormat(TextInputFormat.class);
	    	conf.setMapperClass(PageRankJob2Mapper.class);
	    	
	    	FileOutputFormat.setOutputPath(conf,new Path(outputPath));
	    	conf.setOutputFormat(TextOutputFormat.class);
	    	conf.setOutputKeyClass(Text.class);
	    	conf.setOutputValueClass(Text.class);
	    	conf.setReducerClass(PageRankJob2Reducer.class);
	    	
	    	JobClient.runJob(conf);
	    }
	    /**
	     * JOB 3:
	     * @param inputPath
	     * @param outputPath
	     * @throws IOException
	     */
	    public void evaluateCount(String inputPath, String outputPath) throws IOException
	    {
	    	JobConf conf = new JobConf(PageRank.class);
	    	FileInputFormat.setInputPaths(conf, new String(inputPath));
	    	conf.setInputFormat(TextInputFormat.class);
	    	conf.setMapperClass(PageRankJob3Mapper.class);
	    	
	    	FileOutputFormat.setOutputPath(conf,new Path(outputPath));
	    	conf.setOutputFormat(TextOutputFormat.class);
	    	conf.setOutputKeyClass(Text.class);
	    	conf.setOutputValueClass(Text.class);
	    	conf.setReducerClass(PageRankJob3Reducer.class);
	    	conf.setNumReduceTasks(1);
	    	JobClient.runJob(conf);
	    }
	    /**
	     * @method evaluatePageRank to determine Page Rank. Method sets Mapper and reducer for Page rank evaluation
	     * for Job3. Method initially reads input for total number of pages from Job2 from a predefined file which
	     * is set in mapper as input and Mapper reads output link graph from results of Job1.
	     * @param inputPath
	     * @param outputPath
	     * @throws IOException
	     */
	    public void evaluatePageRank(String inputPath,String outputPath,int iter,String path1) throws IOException
	    {
	    	//Determine count by reading output from File output of Job2.
	    	int count = 1;
			Configuration config = new Configuration();
			Path path = new Path(path1 + "tmp/job2/part-00000");
			FileSystem fs = path.getFileSystem(config);
			BufferedReader br=new BufferedReader(new InputStreamReader(fs.open(path)));
			String line = br.readLine();
			
			if(line!=null && !line.isEmpty())
				count = Integer.parseInt(line.substring(2).trim());
			br.close();
			fs.close();
			
			//Set all Parameters for Mapper and Reducer.
	    	JobConf conf = new JobConf(PageRank.class);
	    	FileInputFormat.setInputPaths(conf, new String(inputPath));
	    	conf.setInputFormat(TextInputFormat.class);
	    	conf.set(PageRankJob4Reducer.TOTAL_COUNT, count+"");
	    	conf.set(PageRankJob4Mapper.ITERATION_COUNT,iter+"");
	    	conf.setMapperClass(PageRankJob4Mapper.class);
	    	
	    	FileOutputFormat.setOutputPath(conf,new Path(outputPath));
	    	conf.setOutputFormat(TextOutputFormat.class);
	    	conf.setOutputKeyClass(Text.class);
	    	conf.setOutputValueClass(Text.class);
	    	conf.setReducerClass(PageRankJob4Reducer.class);
	    	JobClient.runJob(conf);
	    }
	    /**
	     * JOB - 4
	     * @method runRankOrdering to sort output of Job3 based on Page rank of each Page.
	     * @param inputPath
	     * @param outputPath
	     * @throws IOException
	     */
	    public void runRankOrdering(String inputPath, String outputPath,String path1) throws IOException 
	    {
	        JobConf conf = new JobConf(PageRank.class);
	        int count = 0;
	        Configuration config = new Configuration();
			Path path = new Path(path1 + "tmp/job2/part-00000");
			FileSystem fs = path.getFileSystem(config);
			BufferedReader br=new BufferedReader(new InputStreamReader(fs.open(path)));
			String line = br.readLine();
			if(line!=null && !line.isEmpty())
				count = Integer.parseInt(line.substring(2).trim());
			br.close();
			fs.close();
			
	        conf.setOutputKeyClass(FloatWritable.class);
	        conf.setOutputValueClass(Text.class);
	        conf.set(PageRankJob4Reducer.TOTAL_COUNT, count+"");
	        conf.setInputFormat(TextInputFormat.class);
	        conf.setOutputFormat(TextOutputFormat.class);
	        
	        conf.setReducerClass(PageRankJob5Reducer.class);
	        FileInputFormat.setInputPaths(conf, new Path(inputPath));
	        FileOutputFormat.setOutputPath(conf, new Path(outputPath));
	        
	        conf.setMapperClass(PageRankJob5Mapper.class);
	        conf.setNumReduceTasks(1);
	        JobClient.runJob(conf);
	    }
	    
}