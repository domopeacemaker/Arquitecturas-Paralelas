#include <cstdio>
#include <iostream>
#include <chrono>

using namespace std;

void cpi(const int n, double& pi)
{
    double x, sum = 0;

    for (int i=1; i<=n; i++) 
    {
	x = (i-0.5) / n;
        sum += 4.0 / (1.0 + x*x);
    }

    pi = sum / n;
}

int main()
{
    int n;
    cout << "Enter number of polynomial terms: ";
    cin >> n;

    auto t1 = std::chrono::high_resolution_clock::now();

    double pi;
    cpi(n, pi);

    auto t2 = std::chrono::high_resolution_clock::now();

    double t_elapsed = std::chrono::duration<double, std::milli>(t2 - t1).count();
    cout << "Result of PI: " << pi << endl;
    cout << "Serial time: " << t_elapsed << "msec. \n" << endl;
}
