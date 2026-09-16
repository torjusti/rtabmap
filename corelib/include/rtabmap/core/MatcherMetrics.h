#ifndef RTABMAP_MATCHER_METRICS_H
#define RTABMAP_MATCHER_METRICS_H
#include <rtabmap/core/rtabmap_core_export.h>
#include <map>
#include <string>
#include <cstdint>
namespace rtabmap {
// Process-wide cumulative counters, keyed by matcher path and model. Take
// differences of snapshots to measure a GUI operation. Seconds count actual
// Python call wall time, including transfers, not summed waiting workers.
struct RTABMAP_CORE_EXPORT MatcherMetrics
{
    uint64_t calls=0, pairs=0, failures=0, batchCalls=0, batchPairs=0;
    double pythonSeconds=0, initializationSeconds=0;
    static std::map<std::string,MatcherMetrics> snapshot();
    static void record(const std::string & key,size_t pairs,double seconds,bool failed);
    static void report(const std::string & key);
    static void initialized(const std::string & key,double seconds);
    MatcherMetrics operator-(const MatcherMetrics & before) const;
};
}
#endif
