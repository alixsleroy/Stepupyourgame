

// anisotropic with two dimensions, with stepsize larger where potential steeper 

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

// Xtensor vector
// #include <iostream>
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xrandom.hpp"



using namespace std::chrono;

using namespace std;
#define m               0.6
#define M               1.
#define m1              m*m         // minimum step scale factor
#define M1              1./M             // maximum step        // maximum step scale factor
#define numsam          5000       // number of sample
vector<double> dtlist = {0.05,0.025,0.01,0.0075,0.005};
#define tau             1.5 

#define gamma           .1         // friction coefficient
#define printskip       100
#define Nt              100000
#define T               500
#define PATH   "/home/s2133976/OneDrive/ExtendedProject/Code/Stepupyourgame/Stepupyourgame/data/C/underdamped/hist_narrowpass"

#define d       2.2
#define R       8.3
#define p       0.001
#define K       .1
#define r       2.
#define ax      0.20
#define xw      0.15 // parameter of phi_1, determines how much of a double the potentila is
#define yw      0.1


///////////////////////////////////////
/// Bobsled Potential around x=4     //
///////////////////////////////////////
double s(double x){
    return 1./(1.+pow(x/d,6.));
    }
double phi1(double x,double y){
    return  p*pow((x-R)*(x+R),2.) + xw*x*x + yw*y*y;
    }
double phi2(double y){
    return 2.+K*y*y;
}
double phi1_x(double x){
    return x*(-4.*p*R*R+4.*p*x*x+2*xw);
}
double phi1_y(double y){
    return 0.2*y;
}
double phi2_y(double y){
    return 2.*K*y;
}
double s_x(double x){
    double x5,d6,res;
    x5=pow(x,5.);
    d6=pow(d,6.);
    res=-6.*x5*d6/(pow(x*x5+d6,2.));
    return res;
}
double Upx(double x, double y){
    double upx,sx_x;
    sx_x=s_x(x);
    upx=phi1_x(x)*(1.-s(x))+sx_x*(phi2(x)-phi1(x,y));
    return upx;
}

double Upy(double x, double y){
    double upy;
    double sx=s(x);
    upy = phi1_y(y)*(1.-sx)+sx*phi2_y(y);
    return upy;
    }



double getg(double x,double y){
    double f=1/(ax*pow(x,(2*r))); //+ay*y*y);  // a*pow((x+3)*(x-3),(2*r));  //(pow((x-d)*(x+d),2*r)*p/R+y*y*0.05*K);   
    double f2=f*f;
    double xi=sqrt(1.+m1*f2);
    double den=M1*xi+sqrt(f2);
    double g=xi/den;
    return(g);
    }

double getgprime_x(double x,double y){
    double f=1/(ax*pow(x,(2*r))); //+ay*y*y);  // a*pow((x+3)*(x-3),(2*r));  //(pow((x-d)*(x+d),2*r)*p/R+y*y*0.05*K);   
    double fp=-2*r*pow(x,-2*r-1)/ax;    ///-2*ax*r*pow(x,2*r-1)/pow(ax*pow(x,2*r)+ay*y*y,2);    //4*a*r*x*pow(x*x-9,2*r-1);                               //4*p*r*x*pow(x*x-d*d,2*r-1);   
    double f2=f*f;
    double xi=sqrt(1.+m1*f2);
    double sqf=sqrt(f2);
    double gp=-f*fp/(sqf*xi*pow(M1*xi+sqf,2.));
    return(gp);
}

double getgprime_y(double x,double y){
    double f=1/(ax*pow(x,(2*r))); //+ay*y*y);  // a*pow((x+3)*(x-3),(2*r));  //(pow((x-d)*(x+d),2*r)*p/R+y*y*0.05*K);   
    double fp=0; //-2*y*ay/pow(ax*pow(x,2*r)+ay*y*y,2); 
    double f2=f*f;
    double xi=sqrt(1.+m1*f2);
    double sqf=sqrt(f2);
    double gp=-f*fp/(sqf*xi*pow(M1*xi+sqf,2.));
    return(gp);
}

/////////////////////////////////////
// Non adaptive one step function //
////////////////////////////////////

vector<double> one_step(double ds, double numruns, int i)
{
    // ******** Try Boost
    random_device rd1;
    boost::random::mt19937 gen(rd1());
    double qx,px,gpx,C,fx,dwx,j,fmpt;
    double qy,py,gpy,fy,dwy;
    // Savethe values 
    vector<double> vec_q((numsam),0);
    vector<double> vec_p((numsam),0);

    // Compute the moments, so its done
    vector<double> moments(8,0);

    int ns,nt,nsp;
    nsp=0;

    #pragma omp parallel private(qx,qy,px,py,fx,fy,C,nt,dwx,dwy) shared(ns,vec_q,vec_p,nsp)
    #pragma omp for
    for(ns = 0; ns<numsam; ns++){
        // Normal generator 
        mt19937 generator(rd1());
        normal_distribution<double> normal(0, 1);

        // X coordinates
        qx = 1.;
        // Derive the sign of the x coordinate 
        px = 0.1;

        // Y coordinates
        qy = 0.;
        py = 0.1;

        // Values of dU/dx and dU/dy
        fx = -Upx(qx,qy);  
        fy = -Upy(qx,qy);  

        j=0;
        for(nt = 0; nt<numruns; nt++)
        {
            //
            // BAOAB integrator
            //

            //**********
            //* STEP B *
            //**********
            // -X coordinates
            px += 0.5*ds*fx;
            // -Y coordinates
            py += 0.5*ds*fy;

            //**********
            //* STEP A *
            //**********
            // -X coordinates
            qx += 0.5*ds*px;
            // -Y coordinates
            qy += 0.5*ds*py;


            //**********
            //* STEP O *
            //**********
            C = exp(-ds*gamma);
            // -X coordinates
            px = C*px + sqrt((1.-C*C)*tau)*normal(generator);
            // -Y coordinates
            py = C*py + sqrt((1.-C*C)*tau)*normal(generator);

            //**********
            //* STEP A *
            //**********
            // -X coordinates
            qx += 0.5*ds*px;
            // -Y coordinates
            qy += 0.5*ds*py;

            //**********
            //* STEP B *
            //**********
            // -X coordinates
            fx = -Upx(qx,qy);
            px += 0.5*ds*fx;
            // -Y coordinates
            fy = -Upy(qx,qy);
            py += 0.5*ds*fy;

            //*****************************
            //* Save values of qx to average
            //******************************
            if(nt%printskip==0 && nt>10000){
                moments[0]+=qx;
                moments[1]+=qx*qx;
                moments[2]+=qx*qx*qx;
                moments[3]+=qx*qx*qx*qx;
                nsp+=1;
            }
        }   


    //*****************************
    //* Save values to plot
    //******************************
    vec_q[ns]=qx;
    vec_p[ns]=px;

}
// rescale the moments 
moments[0]=moments[0]/nsp;
moments[1]=moments[1]/nsp;
moments[2]=moments[2]/nsp;
moments[3]=moments[3]/nsp;




fstream file;
string file_name;
string path=PATH;
file_name=path+"/vec_noada_qxi="+to_string(i)+".txt";
file.open(file_name,ios_base::out);
ostream_iterator<double> out_itr(file, "\n");
copy(vec_q.begin(), vec_q.end(), out_itr);
file.close();
file_name=path+"/vec_noada_pxi="+to_string(i)+".txt";
file.open(file_name,ios_base::out);
copy(vec_p.begin(), vec_p.end(), out_itr);
file.close();

return moments;
}

////////////////////////////////////////////////
// Adaptive one step function with corr in O //
///////////////////////////////////////////////
vector<double> one_step_trO(double dt, double numruns, int i)
{
    // ******** Try Boost
    random_device rd1;
    boost::random::mt19937 gen(rd1());
    double qx,px,fx,g,gpx,gdt,C,g0,g1,j,g_av,g_av_sample;
    double qy,py,fy,gpy;
    int ns,nt,nsp;
    g_av_sample=0;
    nsp=0;
    // Savethe values 
    vector<double> vec_q((numsam),0);
    vector<double> vec_p((numsam),0);
    vector<double> vec_g((numsam),0);

    // Compute the moments, so its done
    vector<double> moments(8,0);

    #pragma omp parallel private(qx,qy,px,py,fx,fy,gpx,gpy,C,nt,gdt,g) shared(ns,vec_q,vec_p,vec_g,g_av,moments,nsp)
    #pragma omp for

    for(ns = 0; ns<numsam; ns++){
        // Normal generator 
        mt19937 generator(rd1());
        normal_distribution<double> normal(0, 1);

        // X- coordinates 
        qx =1.;
        px = .1;
  
    
        // Y- coordinates 
        qy = 0.;
        py = .1;

        // 
        gpx=getgprime_x(qx,qy);
        gpy=getgprime_y(qx,qy);

        fx = -Upx(qx,qy);   // force
        fy = -Upy(qx,qy);   // force

        // g_av=0.;
        g = getg(qx,qy);
        gdt = dt*g;

        j=0;
        g_av=0;
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


            // * Save values of g
            g_av+=g;

            //*****************************
            //* Save values of qx to average
            //******************************
            if(nt%printskip==0 && nt>10000){
                moments[0]+=qx;
                moments[1]+=qx*qx;
                moments[2]+=qx*qx*qx;
                moments[3]+=qx*qx*qx*qx;
                nsp+=1;
            }
        }
    //*****************************
    //* Save values of g to average
    //******************************
    g_av=g_av/(numruns);
    g_av_sample+=g_av;

    //*****************************
    //* Save values to average
    //******************************
    vec_q[ns]=qx;
    vec_p[ns]=px;
    vec_g[ns]=g;
    }

// rescale the moments 
moments[0]=moments[0]/nsp;
moments[1]=moments[1]/nsp;
moments[2]=moments[2]/nsp;
moments[3]=moments[3]/nsp;



fstream file;
string file_name;
string path=PATH;
file_name=path+"/vec_trO_qxi="+to_string(i)+".txt";
file.open(file_name,ios_base::out);
ostream_iterator<double> out_itr(file, "\n");
copy(vec_q.begin(), vec_q.end(), out_itr);
file.close();

file_name=path+"/vec_trO_pxi="+to_string(i)+".txt";
file.open(file_name,ios_base::out);
copy(vec_p.begin(), vec_p.end(), out_itr);
file.close();

file_name=path+"/vec_trO_gi="+to_string(i)+".txt";
file.open(file_name,ios_base::out);
copy(vec_g.begin(), vec_g.end(), out_itr);
file.close();

return moments;
}



////////////////////////////////////////////////
// Adaptive one step function with corr in B //
///////////////////////////////////////////////
vector<double> one_step_trB(double dt, double numruns, int i)
{
    // ******** Try Boost
    random_device rd1;
    boost::random::mt19937 gen(rd1());
    double qx,px,fx,g,gpx,gdt,C,g0,g1,j,g_av,g_av_sample;
    double qy,py,fy,gpy;
    int ns,nt,nsp;
    g_av_sample=0;
    nsp=0;

    // Savethe values 
    vector<double> vec_q((numsam),0);
    vector<double> vec_p((numsam),0);
    vector<double> vec_g((numsam),0);

    // Compute the moments, so its done
    vector<double> moments(8,0);

    #pragma omp parallel private(qx,qy,px,py,fx,fy,gpx,gpy,C,nt,gdt,g) shared(ns,vec_q,vec_p,vec_g,g_av,moments)
    #pragma omp for

    for(ns = 0; ns<numsam; ns++){
        // Normal generator 
        mt19937 generator(rd1());
        normal_distribution<double> normal(0, 1);

        // X- coordinates 
        qx =1.;
        px = .1;
  
    
        // Y- coordinates 
        qy = 0.;
        py = .1;

        // 
        gpx=getgprime_x(qx,qy);
        gpy=getgprime_y(qx,qy);

        fx = -Upx(qx,qy);   // force
        fy = -Upy(qx,qy);   // force

        // g_av=0.;
        g = getg(qx,qy);
        gdt = dt*g;

        j=0;
        g_av=0;
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
            px += 0.5*dt*tau*gpx;

            // Y- coordinates 
            py += 0.5*gdt*fy;
            py += 0.5*dt*tau*gpy;


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
            // X- coordinates 
            px = C*px+ sqrt((1.-C*C)*tau)*normal(generator);
             // Y- coordinates 
            py = C*py+ sqrt((1.-C*C)*tau)*normal(generator);

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
            px += 0.5*dt*tau*gpx;

            // Y- coordinates 
            py += 0.5*gdt*fy;
            py += 0.5*dt*tau*gpy;


            // * Save values of g
            g_av+=g;

            //*****************************
            //* Save values of qx to average
            //******************************
            if(nt%printskip==0 && nt>10000){
                moments[0]+=qx;
                moments[1]+=qx*qx;
                moments[2]+=qx*qx*qx;
                moments[3]+=qx*qx*qx*qx;
            }

        }
    //*****************************
    //* Save values of g to average
    //******************************
    g_av=g_av/(numruns);
    g_av_sample+=g_av;

    //*****************************
    //* Save values of vector
    //******************************
    vec_q[ns]=qx;
    vec_p[ns]=px;
    vec_g[ns]=g;

    }

// rescale the moments 
moments[0]=moments[0]/nsp;
moments[1]=moments[1]/nsp;
moments[2]=moments[2]/nsp;
moments[3]=moments[3]/nsp;



fstream file;
string file_name;
string path=PATH;
file_name=path+"/vec_trB_qxi="+to_string(i)+".txt";
file.open(file_name,ios_base::out);
ostream_iterator<double> out_itr(file, "\n");
copy(vec_q.begin(), vec_q.end(), out_itr);
file.close();

file_name=path+"/vec_trB_pxi="+to_string(i)+".txt";
file.open(file_name,ios_base::out);
copy(vec_p.begin(), vec_p.end(), out_itr);
file.close();

file_name=path+"/vec_trB_gi="+to_string(i)+".txt";
file.open(file_name,ios_base::out);
copy(vec_g.begin(), vec_g.end(), out_itr);
file.close();

return moments;
}


int main(void) {    


    // Start the clock to have the number of simulation 
    auto start = high_resolution_clock::now();
    using namespace std;

    vector<double> moments_1(dtlist.size(),0);
    vector<double> moments_2(dtlist.size(),0);
    vector<double> moments_3(dtlist.size(),0);
    vector<double> moments_4(dtlist.size(),0);

    vector<double> moments_trB_1(dtlist.size(),0);
    vector<double> moments_trB_2(dtlist.size(),0);
    vector<double> moments_trB_3(dtlist.size(),0);
    vector<double> moments_trB_4(dtlist.size(),0);

    vector<double> moments_trO_1(dtlist.size(),0);
    vector<double> moments_trO_2(dtlist.size(),0);
    vector<double> moments_trO_3(dtlist.size(),0);
    vector<double> moments_trO_4(dtlist.size(),0);

    for(int i = 0; i < dtlist.size(); i++){ 
        double dti = dtlist[i];
        double ni = Nt;

           // no adaptivity 
        vector<double> moments_di=one_step(dti,ni,i);
        moments_1[i]=moments_di[0];
        moments_2[i]=moments_di[1];
        moments_3[i]=moments_di[2];
        moments_4[i]=moments_di[3];


        // transformed with corr in step B 
        moments_di=one_step_trB(dti,ni,i);
        moments_trB_1[i]=moments_di[0];
        moments_trB_2[i]=moments_di[1];
        moments_trB_3[i]=moments_di[2];
        moments_trB_4[i]=moments_di[3];
 
        // transformed with corr in step O 
        moments_di=one_step_trO(dti,ni,i);
        moments_trO_1[i]=moments_di[0];
        moments_trO_2[i]=moments_di[1];
        moments_trO_3[i]=moments_di[2];
        moments_trO_4[i]=moments_di[3];

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

    // TRANSFORMED with corr in B 
    file_name=path+"/tr_moment1B.txt";
    file.open(file_name,ios_base::out);
    copy(moments_trB_1.begin(), moments_trB_1.end(), out_itr);
    file.close();

    file_name=path+"/tr_moment2B.txt";
    file.open(file_name,ios_base::out);
    copy(moments_trB_2.begin(), moments_trB_2.end(), out_itr);
    file.close();

    file_name=path+"/tr_moment3B.txt";
    file.open(file_name,ios_base::out);
    copy(moments_trB_3.begin(), moments_trB_3.end(), out_itr);
    file.close();

    file_name=path+"/tr_moment4B.txt";
    file.open(file_name,ios_base::out);
    copy(moments_trB_4.begin(), moments_trB_4.end(), out_itr);
    file.close();

        // TRANSFORMED with corr in O 
    file_name=path+"/tr_moment1O.txt";
    file.open(file_name,ios_base::out);
    copy(moments_trO_1.begin(), moments_trO_1.end(), out_itr);
    file.close();

    file_name=path+"/tr_moment2O.txt";
    file.open(file_name,ios_base::out);
    copy(moments_trO_2.begin(), moments_trO_2.end(), out_itr);
    file.close();

    file_name=path+"/tr_moment3O.txt";
    file.open(file_name,ios_base::out);
    copy(moments_trO_3.begin(), moments_trO_3.end(), out_itr);
    file.close();

    file_name=path+"/tr_moment4O.txt";
    file.open(file_name,ios_base::out);
    copy(moments_trO_4.begin(), moments_trO_4.end(), out_itr);
    file.close();

    // * SAVE THE TIME AND PARAMETERS OF THE SIMULATION IN A INFO FILE
    ///////////////////////////////////////////////////////////////////
    // find time by subtracting stop and start timepoints 
    auto stop = high_resolution_clock::now();
    auto duration_m = duration_cast<minutes>(stop - start);
    auto duration_s = duration_cast<seconds>(stop - start);
    auto duration_ms = duration_cast<microseconds>(stop - start);
    // save the parameters in a file info
    // string parameters="M="+to_string(M)+"-m="+to_string(m)+"-gamma="+to_string(gamma)+"-tau="+to_string(tau)+"-a="+to_string(a)+"-b="+to_string(b)+"-x0="+to_string(x0)+"-c="+to_string(c)+"-Ns="+to_string(numsam)+"-time_sim_min="+to_string(duration_m.count())+"-time_sim_sec="+to_string(duration_s.count())+"-time_sim_ms="+to_string(duration_ms.count());
    string parameters="M1="+to_string(M1)+"-m1="+to_string(m1)+"-gamma="+to_string(gamma)+"-tau="+to_string(tau)+"-time_sim_min="+to_string(duration_m.count())+"-time_sim_sec="+to_string(duration_s.count())+"-time_sim_ms="+to_string(duration_ms.count());
    string information=path+"/parameters_used.txt";
    file.open(information,ios_base::out);
    file << parameters;
    file <<"\n";
    file <<"list of dt";
    copy(dtlist.begin(), dtlist.end(), out_itr);
    file.close();
    
    }

return 0;
}