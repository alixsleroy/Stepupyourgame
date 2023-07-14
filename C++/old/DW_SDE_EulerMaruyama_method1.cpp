#include <stdio.h>
#include <random>
#include <cmath>
#include <string>
#include <list>
#include <iostream>
#include <chrono>
#include <thread>
#include <fstream>
#include <iterator>
#include <iomanip>
using namespace std;

// Xtensor vector
#include <iostream>
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xrandom.hpp"

// Random number generator
#include "/home/s2133976/anaconda3/include/pcg_random.hpp"
// Random number generator - boost functions 
#include "/home/s2133976/anaconda3/include/pcg_random.hpp"
#include <boost/random/random_device.hpp> //boost function
#include <boost/random/normal_distribution.hpp> //include normal distribution
#include <boost/random/mersenne_twister.hpp>

/* 
Generating random number is the most expensive part of this bits of code, it takes between 70 to 95% of the time required to generate a sample. 
The standard random number generator is accurate and slow, having tried different other options and could not really get anything faster (pcg, boost):

// ******** Try PCG - fast random number generator
// Seed with a real random value, if available
// pcg_extras::seed_seq_from<std::random_device> seed_source;


// ******** Try Boost
#include <boost/random/random_device.hpp> //boost function
#include <boost/random/normal_distribution.hpp> //include normal distribution
#include <boost/random/mersenne_twister.hpp>
random_device rd1; (for some reason could not make boost random device run)
boost::random::normal_distribution<> normal(0,1.0);
boost::random::mt19937 gen(rd1());


I have established three ways of generating samples and evaluate how fast they are. 

* Method 1: 
-----------
The random number are generated inside the loop that we parralelise. The random number generator must be thread safe. Therefore the structure is typically: 
for i in M //first loop
    // mt19937 generator(rd1());
    // normal_distribution<double> normal(mean, stddev);
    for t in T:  // second loop 
        b1=normal(generator); 
        do my calculations

Speed without parallel and M=10^5 samples over N=10^3 steps = 3.2 sec
Speed with parallel and M=10^5 samples over N=10^3 steps = 1.3sec

A sample is saved in vec_sol1.txt 

* Method 2: 
-----------
The random seed is generated outside of the loop, to avoid having to use the random device in the loop and try and save time. I generate the seed beforehand using xt::array: 
xt::xarray<double> rd_vec= xt::random::randint({M,1}, 0, 1000000000, gen);
And then use each element of this array as a random seed that I allocate during the loop. 

for(int r =0; r<M;r++){
    mt19937 gen{rd_vec()};
    normal_distribution<double> normal(mean, stddev);
    for(int u = 0; u <= N; u++){ // run the loop until time N
            mat_normal(r,u)=normal(gen); 
        }
}

Speed without parallel and M=10^5 samples over N=10^3 steps = 3.11 + 0.5 sec for the random seed 
Speed with parallel and M=10^5 samples over N=10^3 steps = 1.354 + 0.6 for the random seed 

A sample is saved in vec_sol2.txt 

* Method 3: 
-----------
Generating the random numbers beforehands and store them in a xt::array matrix. However limitations are that one does not know how many random we will need for the random number in advance and secondly xt::array can not 
store more than 10^6 columns of values. 
either use: 
xt::xarray<double> mat_normal= xt::random::randn({M,N}, mean, stddev, gen);

or

xt::xarray<double> mat_normal = xt::zeros<double>({M, N});
for(int r =0; r<M;r++){
    mt19937 gen{rd1()};
    normal_distribution<double> normal(mean, stddev);
    for(int u = 0; u <= N; u++){ // run the loop until time N
            mat_normal(r,u)=normal(gen); 
        }
}

Speed without parallel and M=10^5 samples over N=10^3 steps = 0.8 sec + 2.2 with xtensor and 3.2 with loop. 
Speed with parallel and M=10^5 samples over N=10^3 steps = 0.136 sec + 2.2 with xtensor and 3.2 with loop. 

A sample is saved in vec_sol3.txt 

* Numba 
--------
When using numba, it seems that the performance achieved is clearly above what this code can do: 
Speed with parallel and M=10^5 samples over N=10^3 steps = 0.5 sec

Therefore, we choose to use numba for the rest of those simulations. 

* Comparing methods results
---------------------------
In order to ensure that the code generates solutions, I wrote the code CcodeVisualisation.ipynb. In this notebook, I plot the potential of the SDE of interest (double well sde), 
generate solutions with a simple python code, generate solutions with a numba code, and generate solutions with C code and compare the resutls. 


* Task file to compile
-----------------------
In order to run the C code and compile it fast, I made use of Pragma op to parralelise the loops. Set up in this way pragma op will parralelise the first loops
and leave the second as it is. In order for it to work, the first loop needs to have a larger number of iterations than the second. 

#pragma omp parallel
#pragma omp for
for loop1 
    for loop2 
        do whatever

In addition, one needs to ask in the task.json: 
        "args": [
                "-fdiagnostics-color=always",
                "-g",
                "${file}",
                "-o",
                "${fileDirname}/${fileBasenameNoExtension}",
                "-Ofast",
                "-fopenmp"

Overall, the task json looks like: 

{
    "tasks": [
        {
            "type": "cppbuild",
            "label": "C/C++: g++ build active file",
            "command": "/usr/bin/g++",
            "args": [
                "-fdiagnostics-color=always",
                "-g",
                "${file}",
                "-o",
                "${fileDirname}/${fileBasenameNoExtension}",
                "-Ofast",
                "-fopenmp"
            ],
            "options": {
                "cwd": "${fileDirname}"
            },
            "problemMatcher": [
                "$gcc"
            ],
            "group": {
                "kind": "build",
                "isDefault": true
            },
            "detail": "Task generated by Debugger."
        }
    ],
    "version": "2.0.0"
}


The task file has several options to be faster such as: O2, O3, Ofast... more info can be found here: https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html
To run the C++ code, one requires a launch.json file and a task.json file stored in a vscode. 
To run a C++ file on visual studio, one can generally follow the steps here: https://code.visualstudio.com/docs/languages/cpp
*/

/**************************************************************************************************/
/******************************** GRADIENT OF POTENTIAL OF INTEREST *******************************/ 
/**************************************************************************************************/

double gradV(double x)
/*
Compute the value of the gradient of V at x
-----------
Input
-----------
x: double

Return
-----------
gV : double
    Value of the gradient of V at x
*/
{

    double gV =  x*x*x-x-3.0*x*x+3.0;
    return gV;
}


/**************************************************************************************************/
/******************************** NUMERICAL METHODS APPLIED TO SDE ********************************/ 
/**************************************************************************************************/
double e_m(double y0,double b1, double dt)
/*
Compute the value of yn+1 after one increment with the numerical scheme euler maruyama
-----------
Input
-----------
y0: double
    value of the SDE at time tn 
b1: double
    value of the brownian increments at time tn
dt: double
    size of the time steps discretisation 

Return
-----------
y1 : double
    Value of yn+1 after one increment
*/
{
    double y1 = y0-gradV(y0)*dt+b1;

    return y1;
}


/**************************************************************************************************/
/************************RANDOM NUMBER MATRIX ****************************/ 
/**************************************************************************************************/

// int** random_mat(int N,int M, double mean, double stddev){
//     static double mat[N][M];
//     random_device rd;
//     mt19937 generator{rd()};
//     normal_distribution<double> normal(mean, stddev);    
//     for (int i = 0; i < N; i++)
//     {
//         for (int j = 0; j < M; j++)
//         {
//            mat[i][j]= normal(generator); 
//            cout<<mat[i][j]<<endl;

//         }
//     }
//   return 0;
// }

/**************************************************************************************************/
/************************ MAIN FUNCTION TO RUN NUMERICAL SCHEME ON SDE ****************************/ 
/**************************************************************************************************/

int main(){

// ******************* Set up parameters of the simulations *******************
int M=100000; //00000;
double dt=0.005;
int N=1000; //000;
double y0=1.0;

// ******************* Set up random device ************************************
auto start_rd = chrono::high_resolution_clock::now();

// Declare variables 
double mean = 0.0;
double stddev  = 0.1;
double b1;

// ******** Try PCG - fast random number generator
// Seed with a real random value, if available
// pcg_extras::seed_seq_from<std::random_device> seed_source;


// ******** Try Boost
random_device rd1;
// boost::random::normal_distribution<> normal(0,1.0);
boost::random::mt19937 gen(rd1());
// double b = normal(gen);   


//*********** Method 1 ************** // 
// nothing to generate outside the loop 

//*********** Method 2 ************** // 
// xt::xarray<double> rd_vec= xt::random::randint({M,1}, 0, 1000000000, gen);
// double rd_vec[M];
// std::uniform_int_distribution<> distrib(-9999999, 9999999);
// for(int i =0; i<M;i++){
//     mt19937 gen{rd1()};
//     rd_vec[i] = distrib(gen);
// }

//*********** Method 3 ************** // 
// randomness outside the loop 
// xt::xarray<double> mat_normal= xt::random::randn({M,N}, mean, stddev, gen);

// xt::xarray<double> mat_normal = xt::zeros<double>({M, N});
// for(int r =0; r<M;r++){
//     mt19937 gen{rd1()};
//     normal_distribution<double> normal(mean, stddev);
//     for(int u = 0; u <= N; u++){ // run the loop until time N
//             mat_normal(r,u)=normal(gen); 
//         }
// }

auto end_rd = chrono::high_resolution_clock::now(); // end the chrono
chrono::duration<double, std::milli> double_rd = end_rd - start_rd;
cout << "The elapsed time for rd generation is " << double_rd.count() << " milliseconds" << endl;


// ******************* Initialise vector of solution ***************************
vector<double> vec(M,y0);
// double array1[M] = { y0 }; // all elements 0
// ******************* Set up the timer ****************************************
auto start = chrono::high_resolution_clock::now();

// ******************* Set up parallelised loops *******************************
int i;
// only outer loop is parallelised, which is what we want
#pragma omp parallel
#pragma omp for
for(i =0; i<M;i++)
{   //**************** Random Number Generation ******************* // 
    // ************* Method 1 *************
    mt19937 generator(rd1());
    normal_distribution<double> normal(mean, stddev);

    // ************* Method 2 *************
    // mt19937 generator(rd_vec(i,0));
    // normal_distribution<double> normal(mean, stddev);

    // ************* Method 3 *************

    for(int t = 0; t <= N; t++){ // run the loop until time N
            // ************* Method 1 and 2 *************
            b1=normal(generator); 
            // ************* Method 3 *************
            // b1 = mat_normal(i,99);
            vec[i]=e_m(vec[i],b1,dt);

            // *********** save random number
            // vec[i]=b1;
        }
} 


// ******************* Print elapsed time ****************************************
auto end = chrono::high_resolution_clock::now(); // end the chrono
chrono::duration<double, std::milli> double_ms = end - start;
cout << "The elapsed time is " << double_ms.count() << " milliseconds" << endl;


auto start_txt = chrono::high_resolution_clock::now();

// ******************* Save vector results into txt file *************************

// Copy the value of the array into the vector to save after in a txt file
// vector<double> vec_save(&array1[0], &array1[M]);
// vec_save.insert(vec_save.end(), vec_save.begin(), vec_save.end());


// copy the value in a txt file
fstream file;
file << fixed << setprecision(16) << endl;
file.open("Ctxtfiles/vec_sol1.txt",ios_base::out);
ostream_iterator<double> out_itr(file, "\n");
copy(vec.begin(), vec.end(), out_itr);
file.close();

auto end_txt = chrono::high_resolution_clock::now(); // end the chrono
chrono::duration<double, std::milli> double_txt = end_txt - start_txt;
cout << "The elapsed time is " << double_txt.count() << " milliseconds" << endl;


return 0;
}

