#include <iostream>
#include <math.h>
#include <cstring>
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
#include <boost/random/random_device.hpp> //boost function
#include <boost/random/normal_distribution.hpp> //include normal distribution
#include <boost/random/mersenne_twister.hpp>


int main(){
	int numsam=10;
	#pragma omp parallel
    #pragma omp for
    for(int ns = 0; ns <= numsam; ns++){ // run the loop for ns samples
			std::cout<< "Hello Alix, what's the problem with this? It works fine!\n";
	}
	return 0;

}
