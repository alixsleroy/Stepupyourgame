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
******************************* METHOD 3 *******************************
------------------------------------------------------------------------
------------------------------------------------------------------------

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
int M=1000000; //00000;
double dt=0.005;
int N=10; //000;
double y0=1.0;

// ******************* Set up random device ************************************
auto start_rd = chrono::high_resolution_clock::now();

// Declare variables 
double mean = 0.0;
double stddev  = 0.1;
double b1;
// ******** Try Boost
random_device rd1;
boost::random::mt19937 gen(rd1());




//*********** Method 3 ************** // 
// randomness outside the loop 
xt::xarray<double> mat_normal= xt::random::randn({M,N}, mean, stddev, gen);

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
{   
    for(int t = 0; t <= N; t++){ // run the loop until time N
            // ************* Method 3 *************
            b1=mat_normal(i,t);
            vec[i]=e_m(vec[i],b1,dt);
        }
} 


// ******************* Print elapsed time ****************************************
auto end = chrono::high_resolution_clock::now(); // end the chrono
chrono::duration<double, std::milli> double_ms = end - start;
cout << "The elapsed time is " << double_ms.count() << " milliseconds" << endl;
auto start_txt = chrono::high_resolution_clock::now();

// ******************* Save vector results into txt file *************************
// copy the value in a txt file
fstream file;
file << fixed << setprecision(16) << endl;
file.open("Ctxtfiles/vec_sol3.txt",ios_base::out);
ostream_iterator<double> out_itr(file, "\n");
copy(vec.begin(), vec.end(), out_itr);
file.close();

auto end_txt = chrono::high_resolution_clock::now(); // end the chrono
chrono::duration<double, std::milli> double_txt = end_txt - start_txt;
cout << "The elapsed time is " << double_txt.count() << " milliseconds" << endl;

return 0;
}

