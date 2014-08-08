#include<iostream>
#include<fstream>
#include<iomanip>
#include<string>
#include<cmath>
#include<omp.h>
#include<vector>
#include<set>
#include<stdlib.h>
#include<cstring>
using namespace std;
const double DAMPING_FACTOR = 0.85f;

void print(vector<vector<pair<int, double> > > matrix)
{
    for( int i=0;i<matrix.size();i++)
    {
        cout<<i<<" "<<matrix[i].size()<<endl;
    }
}

/**
 * @method getFileSize to return number of distinct source destination in file. Method uses set to support unique element insert in file.
 * @param filename char * to read from file.
 * @returns int number of distinct nodes for matrix.
 *
 */
int getFileSize(char* filename)
{
    ifstream infile(filename, std::ifstream::in);
    std::set<int> myset;
    int dummy1,dummy2;
    if(!infile.good())
    {
        cout<<"Invalid File"<<endl;
        exit(0);
    }
    while(infile>>dummy1>>dummy2)
    {
        myset.insert(dummy1);
        myset.insert(dummy2);
    }
    infile.close();
   cout<<"File Size :: "<<setprecision(20)<<myset.size()<<endl;
   return myset.size();
}

/**
 * @method normaliseMatrix to normalise matrix read from file based on number of values for inlink.
 * @param matrix vector of vector for matrix stored read from file.
 * @param norm_col_factor normalization value for each column to tell number of inlinks for particular node.
 *
 */
void normaliseMatrix(vector<vector<pair<int, double> > >& matr, int *norm_col_factor)
{
    for(int i =0;i<matr.size();i++)
    {
        for(int j=0;j<matr[i].size();j++)
        {
            matr[i][j].second = 1.0/norm_col_factor[matr[i][j].first];
        }
    }
}

/**
 * @method readMatrix to read matrix from file specified. and initialise matrix passed by reference.
 * Sparse Matrix implementaion to save on time and space.
 * @param filename char* to file name.
 * @param matr vector of vector for storing source destination matrix.
 */
void readMatrix(char* filename, vector<vector<pair<int, double> > >& matr)
{
    int* norm_col_factor = NULL;                 //used to track sum for each column for normalisation.
    int rows,cols;
    rows = cols = matr.size();

    norm_col_factor = new int[cols];         //memset. 
    
    //memset(norm_col_factor, 0, cols*sizeof(double));
    
    for(int i=0;i<cols;i++)
    {
        norm_col_factor[i] = 0;
    }

    //Read data from file.
    ifstream infile(filename, std::ifstream::in);
    int count = 0;
    int source, destination;
    while(infile>>source>>destination)
    {
        
        norm_col_factor[source]++;
        norm_col_factor[destination]++;

        matr[source].push_back(make_pair(destination, 1.0));
        matr[destination].push_back(make_pair(source, 1.0));
        count++;
    }
    normaliseMatrix(matr, norm_col_factor);
    infile.close();

}

/**
 * @method evaluateIteration to evaluate iterations for Pagerank for matrix passed as reference.
 * @param matrix vector of vector to store adjacency matrix.
 * @returns double* with size equal number of sources. This stores page rank for each of the pages.
 */
double* evaluateIteration(vector<vector<pair<int, double> > >& matrix)
{
    double* pagerank = NULL;
    double* pagerankcopy = NULL;

    pagerank = new double[matrix.size()];
    pagerankcopy = new double[matrix.size()];
    const double constant_initialiser = 1.0/matrix.size();

    //initialisation.

#pragma omp parallel for
    for(int i=0;i<matrix.size();i++)
        pagerank[i] = constant_initialiser;                                // TBD  1.0/rows;

    //Iteration
    bool notConverged = true;
    static int count = 0;

    const double add_factor = (1.0-DAMPING_FACTOR)/matrix.size();
    //To be checked for convergence
    while(notConverged)
    {
        notConverged = false;
#pragma omp parallel for
        for(int i=0;i<matrix.size();i++)
        {
            double sum = 0.0;
#pragma omp parallel for reduction(+:sum)
            for(int j=0;j<matrix[i].size();j++)
                sum = sum + (pagerank[matrix[i][j].first] * matrix[i][j].second);
            pagerankcopy[i] = (sum * DAMPING_FACTOR) + add_factor;

            if(abs(pagerankcopy[i]-pagerank[i]) > 0.000000000000001 && count<150)
                notConverged = true;
        }
        
        //swap values for next iteration.
        double* temp = pagerank;
        pagerank = pagerankcopy;
        pagerankcopy = temp;

        count++;
    }
    cout<<"Total Iterations:: "<<count<<endl;
    return pagerank;
}

/*
 * @method writeToFile to write output results to a file.
 * @param outfile name of output file to write results.
 * @param pagerank pointer to array of results.
 * @param rows size of array.
 */
void writeToFile(char* outfile, double* pagerank, int rows)
{
    ofstream file(outfile);
    for(int i=0;i<rows;i++)
    {
        file<<i<<","<<pagerank[i]<<endl; 
    }
    file.close();
}
int main(int argc,char *argv[])
{ 
    if(argc != 2)
    {
        cout<<"Incorrect Usage\n"<<"Usage:: ./pagerank inputfile"<<endl;
        return 0;
    }
    //set number of threads to maximum possible threads.
    omp_set_num_threads(4);

    double start_time = omp_get_wtime();
    int matrix_size = getFileSize(argv[1]);
    
    //matrix to read from file.
    vector<vector<pair<int, double> > >  matrix(matrix_size);
    
    //read matrix from file.
    readMatrix(argv[1], matrix);
    
    //evaluate Page rank
    double* pagerank = evaluateIteration(matrix);
    double end_time = omp_get_wtime();

    cout<<"Time:: "<<setprecision(20)<<end_time-start_time<<endl;
    
    //write output to file.
    writeToFile("Output_Task1.csv",pagerank, matrix_size);
    
    return 0;
}
