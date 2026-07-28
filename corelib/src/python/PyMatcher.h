/**
 * Python interface for python matchers like:
 *  - SuperGlue: https://github.com/magicleap/SuperGluePretrainedNetwork
 *  - OANET https://github.com/zjhthu/OANet
 */

#ifndef PYMATCHER_H
#define PYMATCHER_H

#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>
#include "rtabmap/core/PythonInterface.h"
#include "rtabmap/core/rtabmap_core_export.h"
#include <vector>
#include <list>
#include <mutex>
#include <condition_variable>
#include <Python.h>

namespace rtabmap
{

class RTABMAP_CORE_EXPORT PyMatcher
{
public:
  PyMatcher(const std::string & pythonMatcherPath,
		  float matchThreshold = 0.2f,
		  int iterations = 20,
		  bool cuda = true,
		  const std::string & model = "indoor");
  virtual ~PyMatcher();

  const std::string & path() const {return path_;}
  float matchThreshold() const {return matchThreshold_;}
  int iterations() const {return iterations_;}
  bool cuda() const {return cuda_;}
  const std::string & model() const {return model_;}

  // Thread-safe. Concurrent calls are coalesced into a single batched
  // inference on the GPU (dynamic batching) when the python script
  // provides a match_batch() function, otherwise they are serialized.
  std::vector<cv::DMatch> match(
		  const cv::Mat & descriptorsQuery,
		  const cv::Mat & descriptorsTrain,
		  const std::vector<cv::KeyPoint> & keypointsQuery,
		  const std::vector<cv::KeyPoint> & keypointsTrain,
		  const cv::Size & imageSize);

private:
  struct MatchRequest
  {
	  const cv::Mat * descriptorsQuery;
	  const cv::Mat * descriptorsTrain;
	  const std::vector<cv::KeyPoint> * keypointsQuery;
	  const std::vector<cv::KeyPoint> * keypointsTrain;
	  cv::Size imageSize;
	  std::vector<cv::DMatch> * matches;
	  bool done;
  };

  // All three assume the caller is the batch leader (only one at a time)
  // and acquire the GIL themselves.
  bool ensureInitialized(int descriptorDim);
  void executeBatch(const std::vector<MatchRequest*> & batch);
  void executeSingle(MatchRequest & request);

  PyObject * pModule_;
  PyObject * pFunc_;
  PyObject * pFuncBatch_;
  std::string path_;
  float matchThreshold_;
  int iterations_;
  bool cuda_;
  std::string model_;

  std::mutex batchMutex_;
  std::condition_variable batchCond_;
  std::list<MatchRequest*> batchQueue_;
  bool batchLeaderActive_;
};

}

#endif
