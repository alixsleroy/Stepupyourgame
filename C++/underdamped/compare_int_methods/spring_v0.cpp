// Spring potential 
// v0 implies BAOAB applied in a dumb manner 
// no fixed point integrations 


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
#include <boost/multi_array.hpp>
#include <chrono>

using namespace std::chrono;
 
using namespace std;

<<<<<<< HEAD
#define m               0.001
#define M               1.5
#define gamma           1.            // friction coefficient
#define tau             .1            // 'temperature'
#define T               50         // Time to integrate to
#define numsam          10000       // total number of trajectories
#define printskip       10
#define PATH   "/home/s2133976/OneDrive/ExtendedProject/Code/Stepupyourgame/Stepupyourgame/data/C/underdamped/comp_int_methods/v0"
=======
#define m               0.5
#define M               1.5
#define gamma           1.            // friction coefficient
#define tau             .1            // 'temperature'
#define T               500          // Time to integrate to
#define numsam          100       // total number of trajectories
#define printskip       10
>>>>>>> ef0a9a43faefdc2754d349208bb595f5e89feb4a



/////////////////////////////////
// Square potential definition //
/////////////////////////////////
<<<<<<< HEAD
vector<double> dtlist ={0.1  , 0.14 , 0.195, 0.273, 0.38 , 0.531, 0.741, 1.034, 1.443, 2.014};
=======
//vector<double> dtlist = {exp(-4.5),exp(-4.21),exp(-3.93),exp(-3.64),exp(-3.36),exp(-3.07),exp(-2.79),exp(-2.5),exp(-2.21),exp(-1.93),exp(-1.64),exp(-1.36),exp(-1.07), exp(-0.79), exp(-0.5)};
vector<double> dtlist = {exp(-3.93),exp(-3.36),exp(-2.79),exp(-2.21),exp(-1.64),exp(-1.07),exp(-0.5)};
>>>>>>> ef0a9a43faefdc2754d349208bb595f5e89feb4a

// /////////////////////////////////
// // Spring potential definition //
// /////////////////////////////////
// Spring potential 
//parameters of the potential 
#define a               2.75
#define b               0.1
#define x0              0.5
#define c               0.1

long double Up(double x)
{
   long double xx02= (x-x0)*(x-x0);
   long double wx =b/(b/a+xx02);
    return (wx*wx+c)*x;
}

double getg(double x)
{
    double wx,f,xi,g;
    wx =(b/a+pow(x-x0,2))/b;
    f = wx*wx;
    xi = f+m;
    g = 1/(1/M+1/sqrt(xi));
    return(g);

}

double getgprime(double x)
{
    double wx,f,fp,xi,gprime;
    wx =(b/a+pow(x-x0,2))/b;
    f = wx*wx;
    fp = 4*(x-x0)*((b/a)+pow(x-x0,2))/(b*b);
    xi=sqrt(f+m*m);
    gprime= M*M*fp/(2*xi*(xi+M)*(xi+M));
    return(gprime);
}

///////////////////// ADAPTIVE STEP SIZE

vector<double> one_step_tr(double dt, double numruns, int i)
{
    // ******** Try Boost
    random_device rd1;
    boost::random::mt19937 gen(rd1());
    double q,p,f,g,gp,gdt,C,g0,g1;
    int ns,nt,nsp;

    // Savethe values 
    vector<double> vec_q((numsam/printskip),0);
    vector<double> vec_p((numsam/printskip),0);
    vector<double> vec_g((numsam/printskip),0);

    // Compute the moments, so its done
    vector<double> moments(8,0);

    // Initialise snapshot
    nsp=0;
    #pragma omp parallel private(q,p,f,C,nt,gdt,g,g0,g1) shared(ns,vec_q,vec_p,vec_g,moments,nsp)
    #pragma omp for
    for(ns = 0; ns<numsam; ns++){
        // Normal generator 
        mt19937 generator(rd1());
        normal_distribution<double> normal(0, 1);
        q = 0.;
        p = 0.;
        g = getg(q);
        gdt = dt*g;
        gp=getgprime(q);
        f = -Up(q);   // force
        for(nt = 0; nt<numruns; nt++)
        {
            //
            // BAOAB integrator
            //

            //**********
            //* STEP B *
            //**********
            p += 0.5*gdt*f;
            p += 0.5*dt*tau*gp;

            //**********
            //* STEP A *
            //**********
            // fixed point iteration
            // g0=getg(q+dt/2*p*g);
            // g1=getg(q+dt/2*p*g0);
            // g0=getg(q+dt/2*p*g1);
            // g1=getg(q+dt/2*p*g0);
            gdt=g*dt;

            q += 0.5*gdt*p;

            //**********
            //* STEP O *
            //**********
            C = exp(-gdt*gamma);
            p = C*p + sqrt((1.-C*C)*tau)*normal(generator);

            //**********
            //* STEP A *
            //**********
            // fixed point iteration
            // g0=getg(q+dt/2*p*g);
            // g1=getg(q+dt/2*p*g0);
            // g0=getg(q+dt/2*p*g1);
            // g1=getg(q+dt/2*p*g0);
            gdt=g*dt;
            q += 0.5*gdt*p;

            //**********
            //* STEP B *
            //**********
            f = -Up(q);
            g = getg(q);
            gdt = dt*g;
            p += 0.5*gdt*f;
            p += 0.5*dt*tau*gp;

            // g = getg(q);
            // gdt = dt*g;
            gp=getgprime(q);

        }

    
    // compute the moments for p and for q 

    // compute the moments for q 
    moments[0]+=q;
    moments[1]+=q*q;
    moments[2]+=q*q*q;
    moments[3]+=q*q*q*q;

    // compute the moments for p 
    moments[4]+=p;
    moments[5]+=p*p;
    moments[6]+=p*p*p;
    moments[7]+=p*p*p*p;

    // Save every printskip values    
    if(ns%printskip==0){
        vec_q[nsp]=q;
        vec_p[nsp]=p;
        vec_g[nsp]=g;
        nsp+=1;
        }
    
    }

    // rescale the moments 
    moments[0]=moments[0]/numsam;
    moments[1]=moments[1]/numsam;
    moments[2]=moments[2]/numsam;
    moments[3]=moments[3]/numsam;
    moments[4]=moments[4]/numsam;
    moments[5]=moments[5]/numsam;
    moments[6]=moments[6]/numsam;
    moments[7]=moments[7]/numsam;


    // save the some of the values generated. 
<<<<<<< HEAD
    string path=PATH;
=======
    string path="/home/s2133976/OneDrive/ExtendedProject/Code/Stepupyourgame/Stepupyourgame/data/C/underdamped/comp_int_methods/v0";
>>>>>>> ef0a9a43faefdc2754d349208bb595f5e89feb4a
    fstream file;
    file << fixed << setprecision(16) << endl;
    string list_para="i="+to_string(i); 
    string file_name=path+"/vec_tr_q"+list_para+".txt";
    file.open(file_name,ios_base::out);
    ostream_iterator<double> out_itr(file, "\n");
    copy(vec_q.begin(), vec_q.end(), out_itr);
    file.close();

    file_name=path+"/vec_tr_p"+list_para+".txt";
    file.open(file_name,ios_base::out);
    copy(vec_p.begin(), vec_p.end(), out_itr);
    file.close();

    file_name=path+"/vec_tr_g"+list_para+".txt";
    file.open(file_name,ios_base::out);
    copy(vec_g.begin(), vec_g.end(), out_itr);
    file.close();

    // return the saved moments 
    return moments;
    }



/////////////////////////////////
// Non adaptive one step function //
/////////////////////////////////

vector<double> one_step(double dt, double numruns, int i)
{
    //tools for sampling random increments
    random_device rd1;
    boost::random::mt19937 gen(rd1());

    // set variables
    double q,p,f,g,gp,gdt,C;
    int ns,nt,nsp;
    // Save the values 
    vector<double> vec_q(numsam/printskip,0);
    vector<double> vec_p(numsam/printskip,0);

    // Compute the moments, so its done
    vector<double> moments(8,0);
    nsp=0;

    #pragma omp parallel private(q,p,f,C,nt,gdt) shared(nsp,ns,vec_q,vec_p,moments)
    #pragma omp for
    for(ns = 0; ns<numsam; ns++){
        // Normal generator 
        mt19937 generator(rd1());
        normal_distribution<double> normal(0, 1);
        q = 0.;
        p = 0.;
        f = -Up(q);  
        for(nt = 0; nt<numruns; nt++)
        {
            //
            // BAOAB integrator
            //

            //**********
            //* STEP B *
            //**********
            p += 0.5*dt*f;

            //**********
            //* STEP A *
            //**********
            q += 0.5*dt*p;

            //**********
            //* STEP O *
            //**********
            C = exp(-dt*gamma);
            p = C*p + sqrt((1.-C*C)*tau)*normal(generator);

            //**********
            //* STEP A *
            //**********
            q += 0.5*dt*p;

            //**********
            //* STEP B *
            //**********
            f = -Up(q);
            p += 0.5*dt*f;
        }

    // compute the moments for p and for q 
    // compute the moments for q 
    moments[0]+=(q);
    moments[1]+=q*q;
    moments[2]+=(q)*q*q;
    moments[3]+=q*q*q*q;

    // compute the moments for p 
    moments[4]+=p;
    moments[5]+=p*p;
    moments[6]+=p*p*p;
    moments[7]+=p*p*p*p;

    // Save every printskip values    
    if(ns%printskip==0){
        vec_q[nsp]=q;
        vec_p[nsp]=p;
        nsp+=1;
        }
    }

// rescale the moments 
moments[0]=moments[0]/numsam;
moments[1]=moments[1]/numsam;
moments[2]=moments[2]/numsam;
moments[3]=moments[3]/numsam;
moments[4]=moments[4]/numsam;
moments[5]=moments[5]/numsam;
moments[6]=moments[6]/numsam;
moments[7]=moments[7]/numsam;

// save the some of the values generated. 
<<<<<<< HEAD
string path=PATH;
=======
string path="/home/s2133976/OneDrive/ExtendedProject/Code/Stepupyourgame/Stepupyourgame/data/C/underdamped/comp_int_methods/v0";
>>>>>>> ef0a9a43faefdc2754d349208bb595f5e89feb4a
fstream file;
file << fixed << setprecision(16) << endl;
string list_para="i="+to_string(i); 
string file_name=path+"/vec_noada_q"+list_para+".txt";
file.open(file_name,ios_base::out);
ostream_iterator<double> out_itr(file, "\n");
copy(vec_q.begin(), vec_q.end(), out_itr);
file.close();

file_name=path+"/vec_noada_p"+list_para+".txt";
file.open(file_name,ios_base::out);
copy(vec_p.begin(), vec_p.end(), out_itr);
file.close();

return moments;
}




int main(void) {    

    // Compute how much time it takes
    auto start = high_resolution_clock::now();
    using namespace std;
    vector<double> moments_1(dtlist.size(),0);
    vector<double> moments_2(dtlist.size(),0);
    vector<double> moments_3(dtlist.size(),0);
    vector<double> moments_4(dtlist.size(),0);

    vector<double> moments_tr_1(dtlist.size(),0);
    vector<double> moments_tr_2(dtlist.size(),0);
    vector<double> moments_tr_3(dtlist.size(),0);
    vector<double> moments_tr_4(dtlist.size(),0);


    for(int i = 0; i < dtlist.size(); i++){ // run the loop for ns samples

        double dti = dtlist[i];
        double ni = T/dti;

        // no adaptivity 
        vector<double> moments_di=one_step(dti,ni,i);
        moments_1[i]=moments_di[0];
        moments_2[i]=moments_di[1];
        moments_3[i]=moments_di[2];
        moments_4[i]=moments_di[3];


        // transformed 
        moments_di=one_step_tr(dti,ni,i);
        moments_tr_1[i]=moments_di[0];
        moments_tr_2[i]=moments_di[1];
        moments_tr_3[i]=moments_di[2];
        moments_tr_4[i]=moments_di[3];
 
<<<<<<< HEAD

    // * SAVE THE COMPUTED MOMENTS IN A FILE
    /////////////////////////////////////////
    string path=PATH;
=======
    }

       // * SAVE THE COMPUTED MOMENTS IN A FILE
    /////////////////////////////////////////
    string path="/home/s2133976/OneDrive/ExtendedProject/Code/Stepupyourgame/Stepupyourgame/data/C/underdamped/comp_int_methods/v0";
>>>>>>> ef0a9a43faefdc2754d349208bb595f5e89feb4a

    // NON ADAPTIVE
    fstream file;
    file << fixed << setprecision(16) << endl;
    string file_name=path+"/noada_moment1.txt";
    file.open(file_name,ios_base::out);
    ostream_iterator<double> out_itr(file, "\n");
    copy(moments_1.begin(), moments_1.end(), out_itr);
    file.close();

    file_name=path+"/noada_moment2.txt";
    file.open(file_name,ios_base::out);
    copy(moments_2.begin(), moments_2.end(), out_itr);
    file.close();

    file_name=path+"/noada_moment3.txt";
    file.open(file_name,ios_base::out);
    copy(moments_3.begin(), moments_3.end(), out_itr);
    file.close();

    file_name=path+"/noada_moment4.txt";
    file.open(file_name,ios_base::out);
    copy(moments_4.begin(), moments_4.end(), out_itr);
    file.close();

    // TRANSFORMED
    file_name=path+"/tr_moment1.txt";
    file.open(file_name,ios_base::out);
    copy(moments_tr_1.begin(), moments_tr_1.end(), out_itr);
    file.close();

    file_name=path+"/tr_moment2.txt";
    file.open(file_name,ios_base::out);
    copy(moments_tr_2.begin(), moments_tr_2.end(), out_itr);
    file.close();

    file_name=path+"/tr_moment3.txt";
    file.open(file_name,ios_base::out);
    copy(moments_tr_3.begin(), moments_tr_3.end(), out_itr);
    file.close();

    file_name=path+"/tr_moment4.txt";
    file.open(file_name,ios_base::out);
    copy(moments_tr_4.begin(), moments_tr_4.end(), out_itr);
    file.close();
<<<<<<< HEAD
    }
=======

>>>>>>> ef0a9a43faefdc2754d349208bb595f5e89feb4a
    // * SAVE THE TIME AND PARAMETERS OF THE SIMULATION IN A INFO FILE
    ///////////////////////////////////////////////////////////////////
    // find time by subtracting stop and start timepoints 
    auto stop = high_resolution_clock::now();
    auto duration_m = duration_cast<minutes>(stop - start);
    auto duration_s = duration_cast<seconds>(stop - start);
    auto duration_ms = duration_cast<microseconds>(stop - start);
    // save the parameters in a file info
<<<<<<< HEAD
    string path=PATH;
    fstream file;
=======
>>>>>>> ef0a9a43faefdc2754d349208bb595f5e89feb4a
    string parameters="M="+to_string(M)+"-m="+to_string(m)+"-Ns="+to_string(numsam)+"-time_sim_min="+to_string(duration_m.count())+"-time_sim_sec="+to_string(duration_s.count())+"-time_sim_ms="+to_string(duration_ms.count());
    string information=path+"/parameters_used.txt";
    file.open(information,ios_base::out);
    file << parameters;
    file <<"\n";
    file <<"list of dt";
<<<<<<< HEAD
    ostream_iterator<double> out_itr(file, "\n");
=======
>>>>>>> ef0a9a43faefdc2754d349208bb595f5e89feb4a
    copy(dtlist.begin(), dtlist.end(), out_itr);
    file.close();

return 0;
}















// /////////////////////////////////
// // Double well 


// #define a 0.4
// #define c 1.6


// long double Up(double x)
// {
//     double xc =x-c;
//     double xa=x+a;
//     double v=2*xc*xc*xc*(xa*xc+2*xa*xa-0.0002);
//     return v;
// }

// ///////////////////////////////////////////////////////////////
// // with the definition 1/(1/M+1/f+m) with f=1/((x-a)^2(x+c)^3)
// ///////////////////////////////////////////////////////////////
// double getg(double x)
// {
//     double xc,xa,f,f2,xi,den,g;
//     xc=x-c;
//     xa=x+a;
//     f=0.5*abs(pow(xc,3)*pow(xa,2));
//     f2=f*f;
//     xi=sqrt(1+m1*f2);
//     den=M1*xi+f;
//     g=xi/den;
//     return(g);
// }
<<<<<<< HEAD
=======

>>>>>>> ef0a9a43faefdc2754d349208bb595f5e89feb4a
// double getgprime(double x)
// {
//     double xc,xa,f,f2,xi,fp,gp;
//     xc=x-c;
//     xa=x+a;
//     f=0.5*abs(pow(xc,3)*pow(xa,2));
//     f2=f*f;
//     fp=0.5*pow(xa,3)*pow(xc,5)*(3*a-2*c+5*x)/abs(f);
//     xi=sqrt(1+m1*f2);
//     gp=-xi*xi*fp/(pow(xi,3)*pow(M1*xi+f,2));
//     return(gp);
//     }
<<<<<<< HEAD
=======







>>>>>>> ef0a9a43faefdc2754d349208bb595f5e89feb4a
// /////////////////////////////////
// // Spring potential definition //
// /////////////////////////////////
// // Spring potential 
// //parameters of the potential 
// #define a               20.0
// #define b               0.1
// #define x0              0.1
// #define c               0.1
<<<<<<< HEAD
=======

>>>>>>> ef0a9a43faefdc2754d349208bb595f5e89feb4a
// long double Up(double x)
// {
//    long double xx02= (x-x0)*(x-x0);
//    long double wx =b/(b/a+xx02);
//     return (wx*wx+c)*x;
// }
<<<<<<< HEAD
=======

>>>>>>> ef0a9a43faefdc2754d349208bb595f5e89feb4a
// double getg(double x)
// {
//     double wx,f,xi,g;
//     wx =(b/a+pow(x-x0,2))/b;
//     f = wx*wx;
//     xi = f+m;
//     g = 1/(1/M+1/sqrt(xi));
//     return(g);
<<<<<<< HEAD
// }
=======

// }

>>>>>>> ef0a9a43faefdc2754d349208bb595f5e89feb4a
// double getgprime(double x)
// {
//     double wx,f,fp,xi,gprime;
//     wx =(b/a+pow(x-x0,2))/b;
//     f = wx*wx;
//     fp = 4*(x-x0)*((b/a)+pow(x-x0,2))/(b*b);
//     xi=sqrt(f+m*m);
//     gprime= M*M*fp/(2*xi*(xi+M)*(xi+M));
//     return(gprime);
// }

/////////////////////////////////
// Square definition //
/////////////////////////////////


// long double Up(double x)
// {
//    return x;
// }

// double getg(double x)
// {
//     return 1;
// }

// double getgprime(double x)
// {
//     return 0;
// }



/////////////////////////////////
// Spring potential definition //
/////////////////////////////////
// Spring potential 
//parameters of the potential 
// #define a               1.0
// #define b               1.0
// #define x0              0.5
// #define c               0.1
// vector<double> dtlist = {exp(-4.5),exp(-4.21),exp(-3.93),exp(-3.64),exp(-3.36),exp(-3.07),exp(-2.79),exp(-2.5),exp(-2.21),exp(-1.93),exp(-1.64),exp(-1.36),exp(-1.07), exp(-0.79), exp(-0.5)};

// long double Up(double x)
// {
//    long double xx02= (x-x0)*(x-x0);
//    long double wx =b/(b/a+xx02);
//     return (wx*wx+c)*x;
// }

// double getg(double x)
// {
//     double wx,f,xi,g;
//     wx =(b/a+pow(x-x0,2))/b;
//     f = wx*wx;
//     xi = f+m;
//     g = 1/(1/M+1/sqrt(xi));
//     return(g);

// }

// double getgprime(double x)
// {
//     double wx,f,fp,xi,gprime;
//     wx =(b/a+pow(x-x0,2))/b;
//     f = wx*wx;
//     fp = 4*(x-x0)*((b/a)+pow(x-x0,2))/(b*b);
//     xi=sqrt(f+m*m);
//     gprime= M*M*fp/(2*xi*(xi+M)*(xi+M));
//     return(gprime);
// }
