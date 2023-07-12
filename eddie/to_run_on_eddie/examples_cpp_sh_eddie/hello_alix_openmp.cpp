#include <iostream>
# include <math.h>
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
#include <chrono>
#include <omp.h>
using namespace std::chrono;
using namespace std;



int main(){
	int rank;
    #pragma omp parallel private(rank)
    #pragma omp for

	for(int ns = 0; ns <= 10; ns++){ // run the loop for ns samples
		rank=omp_get_thread_num();

		// copy the value in a txt file
		fstream file;
		string filename="ranks/rank"+to_string(rank);
		file.open(filename,ios_base::out);
    	file <<"Hello Alix";
    	file <<to_string(rank);
		file.close();
	}

	return 0;

}
