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
/************************ MAIN FUNCTION TO RUN NUMERICAL SCHEME ON SDE ****************************/ 
/**************************************************************************************************/

int main(){

// ******************* Set up parameters of the simulations *******************
int M=500000;
double dt=0.005;
int N=1000;
double y0=1.0;
double vec_i;

// ******************* Set up random device ************************************
random_device rd;
double mean = 0.0;
double stddev  = 0.1;

// ******************* Initialise vector of solution ***************************
vector<double> vec(M,y0);
//double vec[M];
//for(int i =0; i<M;i++) vec[i] = y0;

// ******************* Set up the timer ****************************************
auto start = chrono::high_resolution_clock::now();

// ******************* Set up parallelised loops *******************************

int i;
// only outer loop is parallelised, which is what we want
#pragma omp parallel
#pragma omp for
for(i =0; i<M;i++)
{
    // need to initialise the random number in the thread to avoid setting the seed only once for all threads
    mt19937 generator{rd()};
    normal_distribution<double> normal(mean, stddev);
    // run the loop until time N
    for(int count = 0; count <= N; count++){ // run the loop until time N
            double b1; 
            b1= normal(generator);    
            vec[i]=e_m(vec[i],b1,dt);
        }

}

// ******************* Print elapsed time ****************************************
auto end = chrono::high_resolution_clock::now(); // end the chrono
chrono::duration<double, std::milli> double_ms = end - start;
cout << "The elapsed time is " << double_ms.count() << " milliseconds" << endl;


// auto start_txt = chrono::high_resolution_clock::now();

// ******************* Save vector results into txt file *************************
// fstream file;
// file << fixed << setprecision(16) << endl;
// file.open("vector_file_5para.txt",ios_base::out);
// ostream_iterator<double> out_itr(file, "\n");
// copy(vec.begin(), vec.end(), out_itr);
// file.close();

// auto end_txt = chrono::high_resolution_clock::now(); // end the chrono
// chrono::duration<double, std::milli> double_txt = end_txt - start_txt;
// cout << "The elapsed time is " << double_txt.count() << " milliseconds" << endl;


return 0;
}

