
// Spring potential 
// v2 implies BAOAB with \nabla g computed in step O
// AND fixed point integration for step A



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


#define m1              0.001
#define M1              1/1.5
#define gamma           1.            // friction coefficient
#define tau             .1            // 'temperature'
#define T               100          // Time to integrate to
#define numsam          1000       // total number of trajectories
#define printskip       1
#define PATH   "/home/s2133976/OneDrive/ExtendedProject/Code/Stepupyourgame/Stepupyourgame/data/C/underdamped_2d_ani_mod"

/////////////////////////////////
// Square potential definition //
/////////////////////////////////
#define k1 0.1
#define k2 10.
vector<double> dtlist ={0.01}; //, 0.14 , 0.195, 0.273, 0.38,0.5,0.8,1. }; //, 0.531, 0.741, 1.034, 1.443, 2.014};


double getp1(double x, double y){
    return pow(y-x*x+4,2);
}
double getp2(double x, double y){
    return pow(y+x*x-4,2);
}
double getp1_x(double x, double y){
    return -4*x*(y-x*x+4);
}
double getp1_y(double x, double y){
    return 2*(y-x*x+4);
}
double getp2_x(double x, double y){
    return 4*x*(y+x*x-4);
}
double getp2_y(double x, double y){
    return 2*(y+x*x-4);
}

double U(double x, double y){
    double p1=getp1(x,y);
    double p2=getp2(x,y);
    double res=(1+k1*p1*p2)/(1+p1)+(k2*p1*p2)/(1+k2*p2);
    return (res);
}

double Upx(double x, double y){
    double p1=getp1(x,y);
    double p2=getp1(x,y);
    double p1_p=getp1_x(x,y);
    double p2_p=getp2_x(x,y);

    double q1_p= (p1_p*(k1*p2-1)+k1*p1*(1+p1)*p2_p)/pow(1+p2,2);
    double q2_p = k2*(k2*p1_p*p2*p2+p1_p*p2+p1*p2_p)/pow(1+k2*p2,2);
    return q1_p+q2_p;
}

double Upy(double x, double y){
    double p1=getp1(x,y);
    double p2=getp1(x,y);
    double p1_p=getp1_y(x,y);
    double p2_p=getp2_y(x,y);
    double q1_p= (p1_p*(k1*p2-1)+k1*p1*(1+p1)*p2_p)/pow(1+p2,2);
    double q2_p = k2*(k2*p1_p*p2*p2+p1_p*p2+p1*p2_p)/pow(1+k2*p2,2);
    return q1_p+q2_p;
}

double getg(double x, double y)
{
    return(1);
}


double getgprime_x(double x,double y)
{

    return(0);
    }

double getgprime_y(double x,double y)
{
    return(0);
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
    double qx,qy,px,py,fx,fy,g,gpx,gpy,gdt,C;
    int ns,nt,nsp;

    // Save the values 
    // -X coordinates
    vector<double> vec_qx(numsam/printskip,0);
    vector<double> vec_px(numsam/printskip,0);
    // -Y coordinates
    vector<double> vec_qy(numsam/printskip,0);
    vector<double> vec_py(numsam/printskip,0);

    // Compute the moments, so its done
    vector<double> moments(5,0);
    nsp=0;

    #pragma omp parallel private(qx,qy,px,py,fx,fy,gpx,gpy,C,nt,gdt) shared(nsp,ns,vec_qx,vec_qy,vec_px,vec_py,moments)
    #pragma omp for
    for(ns = 0; ns<numsam; ns++){
        // Normal generator 
        mt19937 generator(rd1());
        normal_distribution<double> normal(0, 1);
        // X coordinates
        qx = 1.;
        px = 1.;

        // Y coordinates
        qy = 1.;
        py = 1.;

        // Values of dU/dx and dU/dy
        fx = -Upx(qx,qy);  
        fy = -Upy(qx,qy);  

        for(nt = 0; nt<numruns; nt++)
        {
            //
            // BAOAB integrator
            //

            //**********
            //* STEP B *
            //**********
            // -X coordinates
            px += 0.5*dt*fx;
            // -Y coordinates
            py += 0.5*dt*fy;

            //**********
            //* STEP A *
            //**********
            // -X coordinates
            qx += 0.5*dt*px;
            // -Y coordinates
            qy += 0.5*dt*py;


            //**********
            //* STEP O *
            //**********
            C = exp(-dt*gamma);
            // -X coordinates
            px = C*px + sqrt((1.-C*C)*tau)*normal(generator);
            // -Y coordinates
            py = C*py + sqrt((1.-C*C)*tau)*normal(generator);

            //**********
            //* STEP A *
            //**********
            // -X coordinates
            qx += 0.5*dt*px;
            // -Y coordinates
            qy += 0.5*dt*py;

            //**********
            //* STEP B *
            //**********
            // -X coordinates
            fx = -Upx(qx,qy);
            px += 0.5*dt*fx;
            // -Y coordinates
            fy = -Upy(qx,qy);
            py += 0.5*dt*fy;
        }

    // compute the moments
    moments[0]+=qy;
    moments[1]+=qx;
    moments[2]+=U(qx,qy);
    moments[3]+=qx*qx;
    moments[4]+=qy*qy;


    // Save every printskip values    
    if(ns%printskip==0){
        // -X coordinates
        vec_qx[nsp]=qx;
        vec_px[nsp]=px;
        // -Y coordinates
        vec_qy[nsp]=qy;
        vec_py[nsp]=py;
        nsp+=1;
        }
    }

// rescale the moments 
moments[0]=moments[0]/numsam;
moments[1]=moments[1]/numsam;
moments[2]=moments[2]/numsam;
moments[3]=moments[3]/numsam;
moments[4]=moments[4]/numsam;



fstream file;
string file_name;
string path=PATH;
file << fixed << setprecision(16) << endl;
file_name=path+"/vec_noada_x.txt";
file.open(file_name,ios_base::out);
ostream_iterator<double> out_itr(file, "\n");
file<<"q\n";
ostream_iterator<double> out_itr2(file, "\n");
std::copy(vec_qx.begin(), vec_qx.end(), out_itr2);
file<<"p\n";
ostream_iterator<double> out_itr4(file, "\n");
std::copy(vec_px.begin(), vec_px.end(), out_itr4);
file.close();

file << fixed << setprecision(16) << endl;
file_name=path+"/vec_noada_y.txt";
file.open(file_name,ios_base::out);
file<<"q\n";
std::copy(vec_qy.begin(), vec_qy.end(), out_itr2);
file<<"p\n";
std::copy(vec_py.begin(), vec_py.end(), out_itr4);
file.close();

return moments;
}



vector<double> one_step_tr(double dt, double numruns, int i)
{
    // ******** Try Boost
    random_device rd1;
    boost::random::mt19937 gen(rd1());
    double qx,px,qy,py,fx,fy,gpx,gpy,g,gdt,C,g0,g1,g_av;
    int ns,nt,nsp;

    // Savethe values 
    // X- coordinates
    vector<double> vec_qx((numsam/printskip),0);
    vector<double> vec_px((numsam/printskip),0);
    // Y-coordinates
    vector<double> vec_qy((numsam/printskip),0);
    vector<double> vec_py((numsam/printskip),0);

    vector<double> vec_g((numsam/printskip),0);


    // Compute the moments, so its done
    vector<double> moments(8,0);

    // Initialise snapshot
    nsp=0;
    #pragma omp parallel private(qx,px,qy,py,fx,fy,C,nt,gdt,g,g0,g1,g_av,gpx,gpy) shared(ns,vec_qx,vec_qy,vec_px,vec_py,vec_g,moments,nsp)
    #pragma omp for
    for(ns = 0; ns<numsam; ns++){
        // Normal generator 
        mt19937 generator(rd1());
        normal_distribution<double> normal(0, 1);
        // X- coordinates 
        qx =1.;
        px = 1.;

        // Y- coordinates 
        qy = 1.;
        py = 1.;

        // 
        gpx=getgprime_x(qx,qy);
        gpy=getgprime_y(qx,qy);

        fx = -Upx(qx,qy);   // force
        fy = -Upy(qx,qy);   // force

        g_av=0.;
        g = getg(qx,qy);
        gdt = dt*g;
        for(nt = 0; nt<numruns; nt++)
        {
            //
            // BAOAB integrator
            //

            //**********
            //* STEP B *
            //**********
            // X- coordinates 
            px += 0.5*gdt*fx;
            // Y- coordinates 
            py += 0.5*gdt*fy;


            //**********
            //* STEP A *
            //**********
            // fixed point iteration
            g0=getg(qx+dt/4*px*g,qy+dt/4*py*g);
            g1=getg(qx+dt/4*px*g0,qy+dt/4*py*g0);
            g0=getg(qx+dt/4*px*g1,qy+dt/4*py*g1);
            g1=getg(qx+dt/4*px*g0,qy+dt/4*py*g0);
            gdt=g1*dt;
            
            // X- coordinates 
            qx += 0.5*gdt*px;
            // Y- coordinates 
            qy += 0.5*gdt*py;

            //**********
            //* STEP O *
            //**********
            g = getg(qx,qy);
            gdt = dt*g;
            C = exp(-gdt*gamma);
            gpx=getgprime_x(qx,qy);
            gpy=getgprime_y(qx,qy);
            // X- coordinates 
            px = C*px+(1.-C)*tau*gpx/(gamma*g) + sqrt((1.-C*C)*tau)*normal(generator);
             // Y- coordinates 
            py = C*py+(1.-C)*tau*gpy/(gamma*g) + sqrt((1.-C*C)*tau)*normal(generator);

            //**********
            //* STEP A *
            //**********
            // fixed point iteration
            g0=getg(qx+dt/4*px*g,qy+dt/4*py*g);
            g1=getg(qx+dt/4*px*g0,qy+dt/4*py*g0);
            g0=getg(qx+dt/4*px*g1,qy+dt/4*py*g1);
            g1=getg(qx+dt/4*px*g0,qy+dt/4*py*g0);
            gdt=g1*dt;
            
            // X- coordinates 
            qx += 0.5*gdt*px;
            // Y- coordinates 
            qy += 0.5*gdt*py;

            //**********
            //* STEP B *
            //**********
            // X- coordinates 
            fx = -Upx(qx,qy);   // force
            fy = -Upy(qx,qy);   // force           
            g = getg(qx,qy);
            gdt = dt*g;

            // X- coordinates 
            px += 0.5*gdt*fx;
            // Y- coordinates 
            py += 0.5*gdt*fy;


            //* Save values of g
            g_av+=g;
            // cout<<"\ng\n";
            // cout<<g;


        }

    
    // compute the moments for p and for q 

    // compute the moments
    moments[0]+=qy;
    moments[1]+=qx;
    moments[2]+=U(qx,qy);
    moments[3]+=qx*qx;
    moments[4]+=qy*qy;
    moments[5]+=g_av/numruns; // save the values taken by g 

    // Save every printskip values    
    if(ns%printskip==0){
        // X- coordinates
        vec_qx[nsp]=qx;
        vec_px[nsp]=px;

        // Y- coordinates
        vec_qy[nsp]=qy;
        vec_py[nsp]=py;

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



    fstream file;
    string file_name;
    string path=PATH;
    file << fixed << setprecision(16) << endl;
    file_name=path+"/vec_tr_x.txt";
    file.open(file_name,ios_base::out);
    ostream_iterator<double> out_itr(file, "\n");
    file<<"q\n";
    ostream_iterator<double> out_itr2(file, "\n");
    std::copy(vec_qx.begin(), vec_qx.end(), out_itr2);
    file<<"p\n";
    ostream_iterator<double> out_itr4(file, "\n");
    std::copy(vec_px.begin(), vec_px.end(), out_itr4);
    file.close();

    file << fixed << setprecision(16) << endl;
    file_name=path+"/vec_tr_y.txt";
    file.open(file_name,ios_base::out);
    file<<"q\n";
    std::copy(vec_qy.begin(), vec_qy.end(), out_itr2);
    file<<"p\n";
    std::copy(vec_py.begin(), vec_py.end(), out_itr4);
    file.close();


    // return the saved moments 
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
    vector<double> moments_5(dtlist.size(),0);

    vector<double> moments_tr_1(dtlist.size(),0);
    vector<double> moments_tr_2(dtlist.size(),0);
    vector<double> moments_tr_3(dtlist.size(),0);
    vector<double> moments_tr_4(dtlist.size(),0);
    vector<double> moments_tr_5(dtlist.size(),0);    
    vector<double> moments_tr_6(dtlist.size(),0);

    for(int i = 0; i < dtlist.size(); i++){ // run the loop for ns samples

        double dti = dtlist[i];
        double ni = T/dti;

      // transformed 
        vector<double> moments_di=one_step_tr(dti,ni,i);
        moments_tr_1[i]=moments_di[0];
        moments_tr_2[i]=moments_di[1];
        moments_tr_3[i]=moments_di[2];
        moments_tr_4[i]=moments_di[3];
        moments_tr_5[i]=moments_di[4];
        moments_tr_6[i]=moments_di[5];

        double gdti=dti*moments_di[3];

        // no adaptivity 
        moments_di=one_step(gdti,ni,i);
        moments_1[i]=moments_di[0];
        moments_2[i]=moments_di[1];
        moments_3[i]=moments_di[2];
        moments_4[i]=moments_di[3];
        moments_5[i]=moments_di[4];


  
 

    // * SAVE THE COMPUTED MOMENTS IN A FILE
    /////////////////////////////////////////
    string path=PATH;

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

    file_name=path+"/noada_moment5.txt";
    file.open(file_name,ios_base::out);
    copy(moments_5.begin(), moments_5.end(), out_itr);
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

    file_name=path+"/tr_moment5.txt";
    file.open(file_name,ios_base::out);
    copy(moments_tr_5.begin(), moments_tr_5.end(), out_itr);
    file.close();

    file_name=path+"/tr_moment6.txt";
    file.open(file_name,ios_base::out);
    copy(moments_tr_6.begin(), moments_tr_6.end(), out_itr);
    file.close();



    }

    // * SAVE THE TIME AND PARAMETERS OF THE SIMULATION IN A INFO FILE
    ///////////////////////////////////////////////////////////////////
    // find time by subtracting stop and start timepoints 
    string path=PATH;
    fstream file;
    ostream_iterator<double> out_itr(file, "\n");
    auto stop = high_resolution_clock::now();
    auto duration_m = duration_cast<minutes>(stop - start);
    auto duration_s = duration_cast<seconds>(stop - start);
    auto duration_ms = duration_cast<microseconds>(stop - start);
    // save the parameters in a file info
    string parameters="M1="+to_string(M1)+"-m1="+to_string(m1)+"-Ns="+to_string(numsam)+"-time_sim_min="+to_string(duration_m.count())+"-time_sim_sec="+to_string(duration_s.count())+"-time_sim_ms="+to_string(duration_ms.count());
    string information=path+"/parameters_used.txt";
    file.open(information,ios_base::out);
    file << parameters;
    file <<"\n";
    file <<"list of dt";
    copy(dtlist.begin(), dtlist.end(), out_itr);
    file.close();
    
return 0;
}
