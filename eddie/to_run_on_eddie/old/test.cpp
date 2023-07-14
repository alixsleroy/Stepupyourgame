
// -g "/home/s2133976/OneDrive/ExtendedProject/Code/Weak SDE approximation/Python/accuracy/accuracy_openmp_c/main_accuracy.cpp" -o "/home/s2133976/OneDrive/ExtendedProject/Code/Weak SDE approximation/Python/accuracy/accuracy_openmp_c/main_accuracy" -Ofast -fopenmp
//
//  main.c
//  adaptive
//
//  Created by Alix on 18/05/2022.
//  This is the working code to compute samples from underdamped using splitting scheme 
//  Baoab. This code implements Euler-Maruyama for the transformed SDE. 
//  
#include <math.h>
#include <cstring>
#include <stdio.h>
#include <random>
#include <cmath>
#include <string>
#include <iostream>
#include <fstream>
#include <iterator>
#include <iomanip>
using namespace std;


int main(){
    cout<<"hello world";
    string parameters="PROUT";
    fstream file;
    file << fixed << setprecision(16) << endl;
    string information="parameters_used.txt";
    file.open(information,ios_base::out);
    file << parameters;
    file.close();

}