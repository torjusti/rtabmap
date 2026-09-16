#include <rtabmap/core/MatcherMetrics.h>
#include <rtabmap/utilite/ULogger.h>
#include <mutex>
#include <chrono>
namespace rtabmap {
namespace {
std::mutex metricsMutex;
std::map<std::string,MatcherMetrics> counters;
std::map<std::string,std::chrono::steady_clock::time_point> lastLog;
}
std::map<std::string,MatcherMetrics> MatcherMetrics::snapshot()
{std::lock_guard<std::mutex> lock(metricsMutex); return counters;}
void MatcherMetrics::record(const std::string & key,size_t pairs,double seconds,bool failed)
{
    bool due=false;
    {
        std::lock_guard<std::mutex> lock(metricsMutex);
        auto & s=counters[key]; ++s.calls; s.pythonSeconds+=seconds;
        if(failed) ++s.failures; else s.pairs+=pairs;
        if(pairs>1) {++s.batchCalls; if(!failed) s.batchPairs+=pairs;}
        const auto now=std::chrono::steady_clock::now();
        due=std::chrono::duration<double>(now-lastLog[key]).count()>=5;
        if(due) lastLog[key]=now;
    }
    if(due) report(key);
}
void MatcherMetrics::report(const std::string & key)
{
    MatcherMetrics s;
    {
        std::lock_guard<std::mutex> lock(metricsMutex);
        auto it=counters.find(key); if(it==counters.end()) return; s=it->second;
    }
    UINFO("Matcher timing [%s]: %llu returned pairs, %llu Python calls (%llu batch calls, %llu batched pairs), %.3fs Python including transfers, %.3fs initialization, %.3fms/returned pair, %llu failed calls. Cumulative process totals.",
        key.c_str(),(unsigned long long)s.pairs,(unsigned long long)s.calls,(unsigned long long)s.batchCalls,
        (unsigned long long)s.batchPairs,s.pythonSeconds,s.initializationSeconds,
        s.pairs?1000*s.pythonSeconds/s.pairs:0,(unsigned long long)s.failures);
}

void MatcherMetrics::initialized(const std::string & key,double seconds)
{std::lock_guard<std::mutex> lock(metricsMutex); counters[key].initializationSeconds+=seconds;}
MatcherMetrics MatcherMetrics::operator-(const MatcherMetrics & b) const
{
    MatcherMetrics d; d.calls=calls-b.calls; d.pairs=pairs-b.pairs; d.failures=failures-b.failures;
    d.batchCalls=batchCalls-b.batchCalls; d.batchPairs=batchPairs-b.batchPairs;
    d.pythonSeconds=pythonSeconds-b.pythonSeconds; d.initializationSeconds=initializationSeconds-b.initializationSeconds;
    return d;
}
}
