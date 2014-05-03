// ###
// ###
// ### Symmetric Banded Matrix Reduction to Tridiagonal Form via Householder Transformations
// ### 
// ###
// ### Grant Bartel, grant.bartel@tum.de
// ### Christoph Riesinger, riesinge@in.tum.de
// ###
// ### 
// ### Technical University of Munich
// ###
// ###

#ifndef AUX_H
#define AUX_H

#include <cuda_runtime.h>
#include <ctime>
#include <string>
#include <sstream>

// parameter processing
template<typename T>
bool getParam(std::string param, T &var, int argc, char **argv)
{
    const char *c_param = param.c_str();
    for(int i=argc-1; i>=1; i--)
    {
        if (argv[i][0]!='-') continue;
        if (strcmp(argv[i]+1, c_param)==0)
        {
            if (!(i+1<argc)) continue;
            std::stringstream ss;
            ss << argv[i+1];
            ss >> var;
            return (bool)ss;
        }
    }
    return false;
}

// measuring time
class Timer
{
    public:
	Timer() : tStart(0), running(false), sec(0.f)
	{
	}
	void start()
	{
		tStart = clock();
		running = true;
	}
	void end()
	{
		if (!running) { sec = 0; return; }
        cudaDeviceSynchronize();
		clock_t tEnd = clock();
		sec = (float)(tEnd - tStart) / CLOCKS_PER_SEC;
		running = false;
	}
	float get()
	{
		if (running) end();
		return sec;
	}
    private:
	clock_t tStart;
	bool running;
	float sec;
};

// cuda error checking
#define CUDA_CHECK cuda_check(__FILE__,__LINE__)
void cuda_check(std::string file, int line);

#endif  // AUX_H
