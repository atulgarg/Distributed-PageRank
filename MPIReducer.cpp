#include<mpi.h>
#include<iostream>
#include<vector>
#include<fstream>
#include<algorithm>
#include<stdlib.h>
#include<stdio.h>
#include<sstream>
#include<cstring>
#include<cmath>
using namespace std;
#define KEY 0
#define VALUE 1

/**
 * @method getFileSize to get size of the file to be read. Method reads the file as specified by parameter by filename and returns the number
 * of key value pairs in the file.
 * @param char* specifying file name.
 * @returns int count of key value pairs
 */
int getFileSize(char *filename)
{
    ifstream infile(filename, std::ifstream::in);
    string st;
    infile>>st;
    int count =0;
    int a,b;
    char c;
    while(infile>>a>>c>>b)
    {
        count++;
    }
    infile.close();
    return count;

}
/**
 * @method readtextfile to read file specified by filename in a integer list. Method initialises vkmap to keep a mapping of keys read
 * and new keys assigned using iterator. This helps in non continous random set of key value pairs. 
 * @param filename char* specifying file to read.
 * @param list_size size of file to read.
 * @param max_key total number of distinct keys, passed by reference to initialise here.
 * @param vkmap map containing mapping of new key, old key.
 *
 */
int* readtextfile(char* filename, int& list_size,int& max_key, map<int, int>& vkmap)
{
    list_size = getFileSize(filename);
    int* list = new int[list_size*2];
    //Read data from file.
    ifstream infile(filename, std::ifstream::in);

    //read first line from file.
    string st;
    infile>>st;
    
    int iter = 0;
    int count = 0;
    while(count < list_size)
    {
        int key, value;
        char comma;
        infile>>key>>comma>>value;
        //check if the key already exists if yes used the earlier index else add a new key value.
        if(vkmap.find(key)!=vkmap.end())
        {
            key = vkmap.find(key)->second;
        }
        else
        {
            vkmap[key] = iter;
            key = iter;  
            iter++;
        }

        list[2*count+KEY] = key;
        list[2*count+VALUE] = value;
        count++;
    }
    
    max_key = iter;
    cout<<"Total Keys:  "<<max_key<<endl;
    infile.close();
    return list;
}
int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int* list;
    int list_size = 0;
    int max_keys = 0;
    map<int, int> vkmap;
    
    if(myrank == 0)
    {
        list_size = 0;
        list = readtextfile("100000_key-value_pairs.csv", list_size, max_keys, vkmap); 
    }

    //broad cast number of key value pairs in file.
    MPI_Bcast(&list_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    //broad cast range of keys.
    MPI_Bcast(&max_keys, 1, MPI_INT, 0, MPI_COMM_WORLD); 

    //dynamically calculate size of elements process will recieve based on rank.
    double lsize = list_size;
    int size = ceil(lsize/world_size);
    
    if(myrank*size + size > list_size)
        size = list_size-(myrank*size);
   
    int *send_count;
    int *displ;

    //initialise parameters for number of keys to send to each process and offsets for Scatterv
    if(myrank ==0)
    {
        send_count = new int[world_size];
        displ = new int[world_size];
        int lastcount = 0;
        for(int i=0;i<world_size;i++)
        {
            send_count[i] = size*2;
            displ[i] = lastcount;
            lastcount = lastcount + size*2;
        }
        send_count[world_size-1] = (list_size - size*(world_size-1))*2;
    }
    
    int _list[size*2];
   
    //Distribute the total key value pair equally block wise among all the process.
    MPI_Scatterv(list, send_count, displ, MPI_INT, _list,size*2, MPI_INT, 0, MPI_COMM_WORLD); 

    
    int send_buf[max_keys];
    //initialise the send buffer to 0.
    memset(send_buf, 0, sizeof(send_buf));
    
    //First Local Reduction. Hash Based Summation of values. Hash is index of array itself.
    for(int i=0 ;i<size ; i++)
    {
        send_buf[_list[2*i]] = send_buf[_list[2*i]] + _list[2*i+1];
    }
   
    //To store local reduction of each of processors.
    int recv_data[world_size][max_keys]; 
   
    //initialised the recieve buffer to 0
    memset(recv_data, 0, sizeof(recv_data[0][0]*world_size*max_keys));
   
    //All process gather results of first local reduction.
    MPI_Allgather(send_buf, max_keys, MPI_INT, recv_data, max_keys, MPI_INT, MPI_COMM_WORLD);
        
    //second local reduction.
    int l = ceil((float)max_keys/(float)world_size);
    int k = 0;

    int key_start = myrank*l;
    int key_end = (key_start+l>max_keys) ? (max_keys) : (myrank+1)*l;

    //each process iterates over its assigned set of keys for second local reduction. 
    for(int iter = key_start;iter<key_end;iter++,k++)
    {
        send_buf[k] = 0;
        for(int i=0;i<world_size;i++)
        {
            send_buf[k]+=recv_data[i][iter];
        }
    }
     
    int* buf  = NULL;
    //Root process again initialises amount of keys to be recieved from each process and strides. This will
    //help in, where total keys are not multiple of processors.
    if(myrank == 0)
    {
        buf = new int[max_keys];
        int lastdispl = 0;
        for(int i=0;i<world_size;i++)
        {
            send_count[i] = (key_end-key_start);
            displ[i] = lastdispl;
            lastdispl = lastdispl + send_count[i];
        }
        send_count[world_size-1] = (max_keys -((key_end-key_start)*(world_size-1)));
    }
    
    //Barrier so as each of the process has completed its second local reduction before root process can
    //do the computation of final result.
    MPI_Barrier(MPI_COMM_WORLD);    
    
    //Gather results in Process 0  
    MPI_Gatherv(send_buf, (key_end-key_start), MPI_INT, buf, send_count,displ, MPI_INT, 0, MPI_COMM_WORLD);
    
    if(myrank == 0)
    {
        //write to file
        ofstream myfile("Output_Task2");
        //Gets the original key from map and writes to output file.
        for(map<int,int>::iterator iter = vkmap.begin();iter!=vkmap.end();iter++)
        {
            myfile<<iter->first<<","<<buf[iter->second]<<endl;
        }
        myfile.close();
        delete buf;
        delete list;
        delete send_count;
        delete displ;
    }
   
    MPI_Finalize();
    return 0;
}
