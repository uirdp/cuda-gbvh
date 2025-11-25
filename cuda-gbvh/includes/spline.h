#pragma once

#include <cmath>
#include <cassert>
#include <vector>

using std::vector;
template <typename T>
class Spline {
    vector<T> points;
    vector<float> times;
    vector<T> tangents;

public:

    static void make_spline(const vector<T>& points, const vector<float>& times, Spline<T>& result){
        int n = points.size();
        if(n < 2) return;

        result.points = points;
        result.times = times;
        result.tangents.resize(n);

        result.tangents[0] = (points[1] - points[0]) / (times[1] - times[0]);
#ifdef FRITSCH_BUTLAND 
	/* Fritsch-Butland's monotone-preserving */
	for( int i = 1; i < n-1; i++ ) {
	    float  h0 = times[i] - times[i-1];
	    float  h1 = times[i+1] - times[i];
	    T  m0 = (points[i] - points[i-1]) / h0;
	    T  m1 = (points[i+1] - points[i]) / h1;
	    float a = (h0 + h1 * 2) / ((h0 + h1) * 3);
	    if( m0 * m1 <= 0 ) {
		result.tangents[i] = 0;
	    } else {
		result.tangents[i] = m0 / (a * m1 + (1-a) * m0) * m1; 
	    }
	}
#else
        /* Cutmull-Rom */
        for( int i = 1; i < n-1; i++ ) {
            result.tangents[i] =
            (points[i+1] - points[i-1]) / (times[i+1] - times[i-1]);
        }
#endif
	    result.tangents[n-1] = (points[n-1] - points[n-2]) / (times[n-1] - times[n-2]);
    
    }

public:
    Spline() {}
    Spline(const vector<T>& points, const vector<float>& times){
        make_spline(points, times, *this);
    }

    bool is_valid()  const { return times.size() >= 2; }

    T get_value(float time) {
        assert(is_valid());
        int i = 1;
        while(i < times.size() - 1 && time > times[i]) i++;
        float h = times[i] - times[i-1]; 
        float t = (time - times[i-1]) / h;
        float t2 = t*t;
        float t3 = t2*t;
        float tt = 2 * t3 - 3 * t2;
        return( (tt + 1) * points[i-1] +
                (t3 - 2 * t2 + t) * h * tangents[i-1] +
                -tt * points[i] +
                (t3 - t2) * h * tangents[i] );
    }
};