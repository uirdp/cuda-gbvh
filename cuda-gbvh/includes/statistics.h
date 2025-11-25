#pragma once
#include <string>
#include <stdio.h>
#define MAX_NUM_THREADS 32

enum EnumSC {
    SC_RAYS,
    SC_LEAF_NODES,
    SC_DIV_NODES,
    SC_GRID_NODES,
    SC_INTERSECTS,
    SC_TRAVERSE,
    SC_AABB,
    SC_STORED_PHOTONS,
    SC_STORED_PHOTONS_C,
    SC_PHOTON_QUERIES,
    SC_PHOTON_TRAVERSE,
    SC_AABB_INS,
    SC_AABB_SA,
    SC_ACTIONS,
    SC_ALL_NODES,
    SC_DIRTY_NODES,
    _COUNTERS_END_
};

#define COUNTER_NAME_STRING(id) \
    (id == SC_RAYS ? "Number of traced rays" : \
     id == SC_LEAF_NODES ? "Number of leaf nodes (frame 0)" : \
     id == SC_DIV_NODES ? "Number of branch nodes (frame 0)" : \
     id == SC_GRID_NODES ? "Number of grid nodes (frame 0)" : \
     id == SC_TRAVERSE ? "Number of tree-node visits (excl. leaves)" : \
     id == SC_INTERSECTS ? "Number of ray-object intersection tests" : \
     id == SC_AABB ? "Number of bounding-box tests" : \
     id == SC_STORED_PHOTONS ? "Number of stored photons (global)" : \
     id == SC_STORED_PHOTONS_C ? "Number of stored photons (caustics)" : \
     id == SC_PHOTON_QUERIES ? "Number of photon-map queries" : \
     id == SC_PHOTON_TRAVERSE ? "Number of node visits in the photon-map kd-tree" : \
     id == SC_AABB_INS ? "Number of AABB growing operations" : \
     id == SC_AABB_SA ? "Number of AABB surface area computations" : \
     id == SC_ACTIONS ? "Number of object insert/delete operations" : \
     id == SC_ALL_NODES ? "Cumulative number of grid/leaf nodes (frame 1+)" : \
     id == SC_DIRTY_NODES ? "Cumulative number of dirty nodes (frame 1+)" : \
     "Unknown counter")

     enum EnumST{
        ST_TREE_CONSTRUCT,
        ST_RAY_TRACE,
        ST_GRID_CONSTRUCT,
        ST_BV_CONSTRUCT,
        _TIMERS_END_
     };

     #define SC_ALL (EnumSC)(-1)
     #define ST_ALL (EnumST)(-1)

     #define TIMER_NAME_STRING(id) \
    (id == ST_TREE_CONSTRUCT ? "Time for tree construction" : \
     id == ST_RAY_TRACE ? "Time for tracing rays" : \
     id == ST_GRID_CONSTRUCT ? "Time for (re)building grids" : \
     id == ST_BV_CONSTRUCT ? "Time for (re)building BVH tree" : \
     "Unknown timer")
//     id == ST_SORT ? "Time for sorting in tree construction" : \

typedef long long  counter_type;

class Statistics {
    counter_type counter[MAX_NUM_THREADS][_COUNTERS_END_];
    double start_time[_TIMERS_END_];
    double cumulative_time[_TIMERS_END_];

public:
    Statistics() {
	clear();
    }

    void clear_count(/*enum_sc*/ int id) { 
	for( int tid = 0; tid < MAX_NUM_THREADS; tid++ ) counter[tid][id] = 0;
    }
    void clear() {
	for( int i = 0; i < _COUNTERS_END_; i++ )  clear_count(i);
	for( int i = 0; i < _TIMERS_END_; i++ )  cumulative_time[i] = 0;
    }

    void reinitForTracing() {
        clear_count(SC_RAYS);
        clear_count(SC_TRAVERSE);
        clear_count(SC_AABB);
        clear_count(SC_INTERSECTS);
        clear_count(SC_PHOTON_QUERIES);
        clear_count(SC_PHOTON_TRAVERSE);
        clear_timer(ST_RAY_TRACE);
        clear_timer(ST_TREE_CONSTRUCT);
    }

    void add_count(EnumSC id, counter_type n) {
#ifndef NDEBUG
#ifdef _OPENMP
	counter[omp_get_thread_num()][id] += n;
#else
	counter[0][id] += n;
#endif
#endif
    }

    void increment_count(EnumSC id) { add_count(id, 1); }

    counter_type get_count(/*enum_sc*/ int id) const {
        counter_type sum = 0;
        for( int tid = 0; tid < MAX_NUM_THREADS; tid++ )
            sum += counter[tid][id];
        return sum;
    }
    
    const char *get_counter_name(EnumSC id) const { 
        return COUNTER_NAME_STRING(id); 
    }

    /* タイマはシングルスレッド専用 */
    void start_timer(EnumST id) {
#ifndef NO_TIMER
        double RT_GetCPUTime();
        start_time[id] = RT_GetCPUTime();
#endif
    }

    void stop_timer(EnumST id) {
#ifndef NO_TIMER
        double RT_GetCPUTime();
        cumulative_time[id] += RT_GetCPUTime() - start_time[id];
#endif
    }

    float get_time(EnumST id) const { return cumulative_time[id]; }
        const char *get_timer_name(EnumSC id) const { 
        return TIMER_NAME_STRING(id); 
    }

    void clear_timer(EnumST id) {
	    cumulative_time[id] = 0;
    }

    static std::string format_int(counter_type p) {  // format an integer with commas
        char buf[8];
        if( p < 0 ) return std::string("-") + format_int(-p);
        else if( p < 1000 ) {
            sprintf(buf, "%d", (int)p);
            return buf;
        } else {
            sprintf(buf, ",%03d", (int)(p % 1000));
            return format_int(p / 1000) + buf;
        }
    }

    void reportCounters(EnumSC id = SC_ALL) {
#ifndef NDEBUG
	for( int i = (id >= 0 ? id : 0);
	     i < (id >= 0 ? id+1 : _COUNTERS_END_); i++ ) {
	        printf("%s: %s", COUNTER_NAME_STRING(i), 
		    std::to_string(get_count(i)).c_str());
//		   formatInt(getCount(i)).c_str());
            if( i == SC_INTERSECTS || i == SC_TRAVERSE || i == SC_AABB ) {
            printf(" (%.2f per ray)",
                (float)get_count(i) / get_count(SC_RAYS));
            } else if( i == SC_DIRTY_NODES ) {
            printf(" (%f%%)", 
                get_count(i) * 100.0 / get_count(SC_ALL_NODES));
            }
            putchar('\n');
	    }
#endif
    }
    
    void reportTimers(EnumST id = ST_ALL, int scale = 1) {
#ifndef NO_TIMER
        for( int i = (id >= 0 ? id : 0); 
            i < (id >= 0 ? id+1 : _TIMERS_END_); i++ ) {
            printf("%s: %.3lf ms\n", TIMER_NAME_STRING(i), cumulative_time[i] / scale * 1000);
        }
#endif
    }

};

extern Statistics statistics;