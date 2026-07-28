/**
 * Python interface for SuperGlue: https://github.com/magicleap/SuperGluePretrainedNetwork
 */

#include <python/PyMatcher.h>
#include <rtabmap/utilite/ULogger.h>
#include <rtabmap/utilite/UDirectory.h>
#include <rtabmap/utilite/UFile.h>
#include <rtabmap/utilite/UStl.h>
#include <rtabmap/utilite/UConversion.h>
#include <rtabmap/utilite/UTimer.h>

#include <pybind11/embed.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include <numpy/arrayobject.h>

#include <map>
#include <tuple>

namespace rtabmap
{

PyMatcher::PyMatcher(
		const std::string & pythonMatcherPath,
		float matchThreshold,
		int iterations,
		bool cuda,
		const std::string & model) :
				pModule_(0),
				pFunc_(0),
				pFuncBatch_(0),
				matchThreshold_(matchThreshold),
				iterations_(iterations),
				cuda_(cuda),
				batchLeaderActive_(false)
{
	PythonInterface::instance("PyMatcher");
	auto expandTilde = [](const std::string & p) -> std::string {
		if(!p.empty() && p[0] == '~' && (p.size() == 1 || p[1] == '/' || p[1] == '\\'))
			return UDirectory::homeDir() + p.substr(1);
		return p;
	};
	path_  = expandTilde(pythonMatcherPath);
	model_ = expandTilde(model);
	UINFO("path = %s", path_.c_str());
	UINFO("model = %s", model_.c_str());

	if(!UFile::exists(path_) || UFile::getExtension(path_).compare("py") != 0)
	{
		UERROR("Cannot initialize Python matcher, the path is not valid: \"%s\"", path_.c_str());
		return;
	}

	pybind11::gil_scoped_acquire acquire;

	std::string matcherPythonDir = UDirectory::getDir(path_);
	if(!matcherPythonDir.empty())
	{
		// For windows:
		matcherPythonDir = uReplaceChar(matcherPythonDir, '\\', '/');
		PyRun_SimpleString("import sys");
		PyRun_SimpleString(uFormat("sys.path.append(\"%s\")", matcherPythonDir.c_str()).c_str());
	}

	_import_array();

	// Invalidate importlib's directory-listing caches so a script created
	// after sys.path was first scanned in this process is still found.
	// Without this, the second Py* instance pointing at a freshly-written
	// script in an already-known directory fails with ModuleNotFoundError.
	PyRun_SimpleString("import importlib; importlib.invalidate_caches()");

	std::string scriptName = uSplit(UFile::getName(path_), '.').front();
	PyObject * pName = PyUnicode_FromString(scriptName.c_str());
	UDEBUG("PyImport_Import");
	pModule_ = PyImport_Import(pName);
	Py_DECREF(pName);

	if(!pModule_)
	{
		UERROR("Module \"%s\" could not be imported! (File=\"%s\")", scriptName.c_str(), path_.c_str());
		UERROR("%s", getPythonTraceback().c_str());
	}
}

PyMatcher::~PyMatcher()
{
	pybind11::gil_scoped_acquire acquire;
	if(pFunc_)
	{
		Py_DECREF(pFunc_);
	}
	if(pFuncBatch_)
	{
		Py_DECREF(pFuncBatch_);
	}
	if(pModule_)
	{
		Py_DECREF(pModule_);
	}
}

std::vector<cv::DMatch> PyMatcher::match(
		  const cv::Mat & descriptorsQuery,
		  const cv::Mat & descriptorsTrain,
		  const std::vector<cv::KeyPoint> & keypointsQuery,
		  const std::vector<cv::KeyPoint> & keypointsTrain,
		  const cv::Size & imageSize)
{
	std::vector<cv::DMatch> matches;

	if(!pModule_)
	{
		UERROR("Python matcher module not loaded!");
		return matches;
	}

	if(!(!descriptorsQuery.empty() &&
		 descriptorsQuery.cols == descriptorsTrain.cols &&
		 descriptorsQuery.type() == CV_32F &&
		 descriptorsTrain.type() == CV_32F &&
		 descriptorsQuery.rows == (int)keypointsQuery.size() &&
		 descriptorsTrain.rows == (int)keypointsTrain.size() &&
		 imageSize.width>0 && imageSize.height>0))
	{
		UERROR("Invalid inputs! Supported python matchers require float descriptors.");
		return matches;
	}

	MatchRequest request;
	request.descriptorsQuery = &descriptorsQuery;
	request.descriptorsTrain = &descriptorsTrain;
	request.keypointsQuery = &keypointsQuery;
	request.keypointsTrain = &keypointsTrain;
	request.imageSize = imageSize;
	request.matches = &matches;
	request.done = false;

	// Dynamic batching: concurrent callers queue their requests, one
	// caller at a time becomes the "leader" and executes all the queued
	// requests in a single batched inference, then wakes the others up.
	std::unique_lock<std::mutex> lock(batchMutex_);
	batchQueue_.push_back(&request);
	while(!request.done)
	{
		if(!batchLeaderActive_)
		{
			batchLeaderActive_ = true;
			// SuperGlue's attention maps scale with batchSize * N0max * N1max
			// (quadratic in keypoint count), so bound the batch by that
			// product instead of a fixed size to keep VRAM usage in check:
			// ~24M "cells" is ~6 pairs of 2048 keypoints or ~24 pairs of
			// 1024 keypoints, peaking at a few GB of GPU memory.
			const size_t budgetCells = 24*1024*1024;
			size_t maxQuery = 0;
			size_t maxTrain = 0;
			std::vector<MatchRequest*> batch;
			while(!batchQueue_.empty())
			{
				MatchRequest * next = batchQueue_.front();
				size_t nextMaxQuery = std::max(maxQuery, next->keypointsQuery->size());
				size_t nextMaxTrain = std::max(maxTrain, next->keypointsTrain->size());
				if(!batch.empty() && (batch.size()+1)*nextMaxQuery*nextMaxTrain > budgetCells)
				{
					break;
				}
				maxQuery = nextMaxQuery;
				maxTrain = nextMaxTrain;
				batch.push_back(next);
				batchQueue_.pop_front();
			}
			lock.unlock();
			executeBatch(batch);
			lock.lock();
			for(size_t i=0; i<batch.size(); ++i)
			{
				batch[i]->done = true;
			}
			batchLeaderActive_ = false;
			batchCond_.notify_all();
		}
		else
		{
			batchCond_.wait(lock);
		}
	}
	return matches;
}

bool PyMatcher::ensureInitialized(int descriptorDim)
{
	if(pFunc_)
	{
		return true;
	}

	UTimer timer;
	UDEBUG("matchThreshold=%f, iterations=%d, cuda=%d", matchThreshold_, iterations_, cuda_?1:0);

	PyObject * pFunc = PyObject_GetAttrString(pModule_, "init");
	if(pFunc)
	{
		if(PyCallable_Check(pFunc))
		{
			PyObject * result = PyObject_CallFunction(pFunc, "ifiis", descriptorDim, matchThreshold_, iterations_, cuda_?1:0, model_.c_str());

			if(result == NULL)
			{
				UERROR("Call to \"init(...)\" in \"%s\" failed!", path_.c_str());
				UERROR("%s", getPythonTraceback().c_str());
				Py_DECREF(pFunc);
				return false;
			}
			Py_DECREF(result);

			pFunc_ = PyObject_GetAttrString(pModule_, "match");
			if(pFunc_ && PyCallable_Check(pFunc_))
			{
				// we are ready!
			}
			else
			{
				UERROR("Cannot find method \"match(...)\" in %s", path_.c_str());
				UERROR("%s", getPythonTraceback().c_str());
				if(pFunc_)
				{
					Py_DECREF(pFunc_);
					pFunc_ = 0;
				}
				Py_DECREF(pFunc);
				return false;
			}

			// Optional batched interface, used to match multiple pairs in a
			// single inference when match() is called concurrently.
			pFuncBatch_ = PyObject_GetAttrString(pModule_, "match_batch");
			if(pFuncBatch_ && !PyCallable_Check(pFuncBatch_))
			{
				Py_DECREF(pFuncBatch_);
				pFuncBatch_ = 0;
			}
			else if(!pFuncBatch_)
			{
				PyErr_Clear();
			}
			if(!pFuncBatch_)
			{
				UWARN("No callable \"match_batch(...)\" in %s: concurrent "
					  "match() calls will be serialized instead of batched.", path_.c_str());
			}
		}
		else
		{
			UERROR("Cannot call method \"init(...)\" in %s", path_.c_str());
			UERROR("%s", getPythonTraceback().c_str());
			Py_DECREF(pFunc);
			return false;
		}
		Py_DECREF(pFunc);
	}
	else
	{
		UERROR("Cannot find method \"init(...)\"");
		UERROR("%s", getPythonTraceback().c_str());
		return false;
	}
	UDEBUG("init time = %fs", timer.ticks());
	return true;
}

void PyMatcher::executeBatch(const std::vector<MatchRequest*> & batch)
{
	UASSERT(!batch.empty());

	pybind11::gil_scoped_acquire acquire;

	// The "init" python function expects the descriptor dimension: use the
	// one of the first request (it is constant for a given feature type).
	if(!ensureInitialized(batch[0]->descriptorsQuery->cols))
	{
		return;
	}

	if(batch.size() == 1 || !pFuncBatch_)
	{
		for(size_t i=0; i<batch.size(); ++i)
		{
			executeSingle(*batch[i]);
		}
		return;
	}

	// Only batch requests that can share a forward pass with no zero-padding:
	// same image size (keypoint normalization depends on it) AND identical
	// keypoint counts. Padding shorter pairs up to the batch maximum is not
	// equivalent to matching them alone, because the fake tokens still take
	// part in SuperGlue's (unmasked) attention and in the optimal-transport
	// marginals, which are sized from the padded counts. Grouping by exact
	// (width,height,N0,N1) keeps each batched result bit-for-bit identical to
	// the serial one; pairs with no same-shape sibling fall through to
	// executeSingle().
	std::map<std::tuple<int, int, int, int>, std::vector<MatchRequest*> > groups;
	for(size_t i=0; i<batch.size(); ++i)
	{
		groups[std::make_tuple(
				batch[i]->imageSize.width, batch[i]->imageSize.height,
				(int)batch[i]->keypointsQuery->size(), (int)batch[i]->keypointsTrain->size())].push_back(batch[i]);
	}

	for(std::map<std::tuple<int, int, int, int>, std::vector<MatchRequest*> >::iterator iter=groups.begin(); iter!=groups.end(); ++iter)
	{
		const std::vector<MatchRequest*> & group = iter->second;
		if(group.size() == 1)
		{
			executeSingle(*group[0]);
			continue;
		}

		UTimer timer;
		npy_intp b = group.size();
		npy_intp dim = group[0]->descriptorsQuery->cols;
		npy_intp maxQuery = 0;
		npy_intp maxTrain = 0;
		for(size_t i=0; i<group.size(); ++i)
		{
			maxQuery = std::max(maxQuery, (npy_intp)group[i]->keypointsQuery->size());
			maxTrain = std::max(maxTrain, (npy_intp)group[i]->keypointsTrain->size());
		}

		npy_intp dimsKpQuery[3] = {b, maxQuery, 2};
		npy_intp dimsKpTrain[3] = {b, maxTrain, 2};
		npy_intp dimsScoresQuery[2] = {b, maxQuery};
		npy_intp dimsScoresTrain[2] = {b, maxTrain};
		npy_intp dimsDescQuery[3] = {b, maxQuery, dim};
		npy_intp dimsDescTrain[3] = {b, maxTrain, dim};
		npy_intp dimsCounts[2] = {b, 2};

		PyObject* pKpQuery = PyArray_ZEROS(3, dimsKpQuery, NPY_FLOAT, 0);
		PyObject* pKpTrain = PyArray_ZEROS(3, dimsKpTrain, NPY_FLOAT, 0);
		PyObject* pScoresQuery = PyArray_ZEROS(2, dimsScoresQuery, NPY_FLOAT, 0);
		PyObject* pScoresTrain = PyArray_ZEROS(2, dimsScoresTrain, NPY_FLOAT, 0);
		PyObject* pDescQuery = PyArray_ZEROS(3, dimsDescQuery, NPY_FLOAT, 0);
		PyObject* pDescTrain = PyArray_ZEROS(3, dimsDescTrain, NPY_FLOAT, 0);
		PyObject* pCounts = PyArray_ZEROS(2, dimsCounts, NPY_INT32, 0);
		UASSERT(pKpQuery && pKpTrain && pScoresQuery && pScoresTrain && pDescQuery && pDescTrain && pCounts);

		float * kpQueryData = (float*)PyArray_DATA((PyArrayObject*)pKpQuery);
		float * kpTrainData = (float*)PyArray_DATA((PyArrayObject*)pKpTrain);
		float * scoresQueryData = (float*)PyArray_DATA((PyArrayObject*)pScoresQuery);
		float * scoresTrainData = (float*)PyArray_DATA((PyArrayObject*)pScoresTrain);
		float * descQueryData = (float*)PyArray_DATA((PyArrayObject*)pDescQuery);
		float * descTrainData = (float*)PyArray_DATA((PyArrayObject*)pDescTrain);
		int * countsData = (int*)PyArray_DATA((PyArrayObject*)pCounts);

		for(size_t i=0; i<group.size(); ++i)
		{
			const MatchRequest & req = *group[i];
			size_t nQuery = req.keypointsQuery->size();
			size_t nTrain = req.keypointsTrain->size();
			countsData[i*2] = (int)nQuery;
			countsData[i*2+1] = (int)nTrain;
			for(size_t j=0; j<nQuery; ++j)
			{
				kpQueryData[(i*maxQuery + j)*2] = req.keypointsQuery->at(j).pt.x;
				kpQueryData[(i*maxQuery + j)*2+1] = req.keypointsQuery->at(j).pt.y;
				scoresQueryData[i*maxQuery + j] = req.keypointsQuery->at(j).response;
			}
			for(size_t j=0; j<nTrain; ++j)
			{
				kpTrainData[(i*maxTrain + j)*2] = req.keypointsTrain->at(j).pt.x;
				kpTrainData[(i*maxTrain + j)*2+1] = req.keypointsTrain->at(j).pt.y;
				scoresTrainData[i*maxTrain + j] = req.keypointsTrain->at(j).response;
			}
			for(int r=0; r<req.descriptorsQuery->rows; ++r)
			{
				memcpy(descQueryData + (i*maxQuery + r)*dim, req.descriptorsQuery->ptr<float>(r), dim*sizeof(float));
			}
			for(int r=0; r<req.descriptorsTrain->rows; ++r)
			{
				memcpy(descTrainData + (i*maxTrain + r)*dim, req.descriptorsTrain->ptr<float>(r), dim*sizeof(float));
			}
		}

		PyObject * pImageWidth = PyLong_FromLong(std::get<0>(iter->first));
		PyObject * pImageHeight = PyLong_FromLong(std::get<1>(iter->first));

		UDEBUG("Preparing batch data time = %fs (batch=%d)", timer.ticks(), (int)b);

		PyObject * pReturn = PyObject_CallFunctionObjArgs(pFuncBatch_, pKpQuery, pKpTrain, pScoresQuery, pScoresTrain, pDescQuery, pDescTrain, pCounts, pImageWidth, pImageHeight, NULL);
		if(pReturn == NULL)
		{
			UWARN("Failed to call match_batch() function (batch=%d)! Falling "
				  "back to matching the pairs one by one.", (int)b);
			UWARN("%s", getPythonTraceback().c_str());
			for(size_t i=0; i<group.size(); ++i)
			{
				executeSingle(*group[i]);
			}
		}
		else
		{
			UDEBUG("Python batched matching time = %fs (batch=%d)", timer.ticks(), (int)b);

			PyArrayObject * np_ret = reinterpret_cast<PyArrayObject*>(pReturn);
			int rows = PyArray_SHAPE(np_ret)[0];
			int cols = PyArray_SHAPE(np_ret)[1];
			int type = PyArray_TYPE(np_ret);
			UASSERT_MSG(cols == 3, uFormat("match_batch() should return a Mx3 array, received Mx%d", cols).c_str());
			UASSERT_MSG(type == NPY_INT64 || type == NPY_UINT64 || type == NPY_INT32 || type == NPY_UINT32,
					uFormat("match_batch() should return an integer array, received type=%d", type).c_str());
			for(int r=0; r<rows; ++r)
			{
				long long bi, from, to;
				if(type == NPY_INT64 || type == NPY_UINT64)
				{
					long long * data = reinterpret_cast<long long*>(PyArray_DATA(np_ret));
					bi = data[r*3];
					from = data[r*3+1];
					to = data[r*3+2];
				}
				else
				{
					int * data = reinterpret_cast<int*>(PyArray_DATA(np_ret));
					bi = data[r*3];
					from = data[r*3+1];
					to = data[r*3+2];
				}
				UASSERT(bi >= 0 && bi < (long long)group.size());
				group[bi]->matches->push_back(cv::DMatch(from, to, 0));
			}
			Py_DECREF(pReturn);
		}

		Py_DECREF(pKpQuery);
		Py_DECREF(pKpTrain);
		Py_DECREF(pScoresQuery);
		Py_DECREF(pScoresTrain);
		Py_DECREF(pDescQuery);
		Py_DECREF(pDescTrain);
		Py_DECREF(pCounts);
		Py_DECREF(pImageWidth);
		Py_DECREF(pImageHeight);
	}
}

void PyMatcher::executeSingle(MatchRequest & request)
{
	UTimer timer;
	const cv::Mat & descriptorsQuery = *request.descriptorsQuery;
	const cv::Mat & descriptorsTrain = *request.descriptorsTrain;
	const std::vector<cv::KeyPoint> & keypointsQuery = *request.keypointsQuery;
	const std::vector<cv::KeyPoint> & keypointsTrain = *request.keypointsTrain;
	std::vector<cv::DMatch> & matches = *request.matches;

	std::vector<float> descriptorsQueryV(descriptorsQuery.rows * descriptorsQuery.cols);
	memcpy(descriptorsQueryV.data(), descriptorsQuery.data, descriptorsQuery.total()*sizeof(float));
	npy_intp dimsFrom[2] = {descriptorsQuery.rows, descriptorsQuery.cols};
	PyObject* pDescriptorsQuery = PyArray_SimpleNewFromData(2, dimsFrom, NPY_FLOAT, (void*)descriptorsQueryV.data());
	UASSERT(pDescriptorsQuery);

	npy_intp dimsTo[2] = {descriptorsTrain.rows, descriptorsTrain.cols};
	std::vector<float> descriptorsTrainV(descriptorsTrain.rows * descriptorsTrain.cols);
	memcpy(descriptorsTrainV.data(), descriptorsTrain.data, descriptorsTrain.total()*sizeof(float));
	PyObject* pDescriptorsTrain = PyArray_SimpleNewFromData(2, dimsTo, NPY_FLOAT, (void*)descriptorsTrainV.data());
	UASSERT(pDescriptorsTrain);

	std::vector<float> keypointsQueryV(keypointsQuery.size()*2);
	std::vector<float> scoresQuery(keypointsQuery.size());
	for(size_t i=0; i<keypointsQuery.size(); ++i)
	{
		keypointsQueryV[i*2] = keypointsQuery[i].pt.x;
		keypointsQueryV[i*2+1] = keypointsQuery[i].pt.y;
		scoresQuery[i] = keypointsQuery[i].response;
	}

	std::vector<float> keypointsTrainV(keypointsTrain.size()*2);
	std::vector<float> scoresTrain(keypointsTrain.size());
	for(size_t i=0; i<keypointsTrain.size(); ++i)
	{
		keypointsTrainV[i*2] = keypointsTrain[i].pt.x;
		keypointsTrainV[i*2+1] = keypointsTrain[i].pt.y;
		scoresTrain[i] = keypointsTrain[i].response;
	}

	npy_intp dimsKpQuery[2] = {(int)keypointsQuery.size(), 2};
	PyObject* pKeypointsQuery = PyArray_SimpleNewFromData(2, dimsKpQuery, NPY_FLOAT, (void*)keypointsQueryV.data());
	UASSERT(pKeypointsQuery);

	npy_intp dimsKpTrain[2] = {(int)keypointsTrain.size(), 2};
	PyObject* pkeypointsTrain = PyArray_SimpleNewFromData(2, dimsKpTrain, NPY_FLOAT, (void*)keypointsTrainV.data());
	UASSERT(pkeypointsTrain);

	npy_intp dimsScoresQuery[1] = {(int)keypointsQuery.size()};
	PyObject* pScoresQuery = PyArray_SimpleNewFromData(1, dimsScoresQuery, NPY_FLOAT, (void*)scoresQuery.data());
	UASSERT(pScoresQuery);

	npy_intp dimsScoresTrain[1] = {(int)keypointsTrain.size()};
	PyObject* pScoresTrain = PyArray_SimpleNewFromData(1, dimsScoresTrain, NPY_FLOAT, (void*)scoresTrain.data());
	UASSERT(pScoresTrain);

	PyObject * pImageWidth = PyLong_FromLong(request.imageSize.width);
	PyObject * pImageHeight = PyLong_FromLong(request.imageSize.height);

	UDEBUG("Preparing data time = %fs", timer.ticks());

	PyObject *pReturn = PyObject_CallFunctionObjArgs(pFunc_, pKeypointsQuery, pkeypointsTrain, pScoresQuery, pScoresTrain, pDescriptorsQuery, pDescriptorsTrain, pImageWidth, pImageHeight, NULL);
	if(pReturn == NULL)
	{
		UERROR("Failed to call match() function!");
		UERROR("%s", getPythonTraceback().c_str());
	}
	else
	{
		UDEBUG("Python matching time = %fs", timer.ticks());

		PyArrayObject *np_ret = reinterpret_cast<PyArrayObject*>(pReturn);

		// Convert back to C++ array and print.
		int len1 = PyArray_SHAPE(np_ret)[0];
		int len2 = PyArray_SHAPE(np_ret)[1];
		int type = PyArray_TYPE(np_ret);
		UDEBUG("Matches array %dx%d (type=%d)", len1, len2, type);
		UASSERT_MSG(type == NPY_INT32 || type == NPY_UINT32 || type == NPY_INT64 || type == NPY_UINT64, uFormat("Returned matches should type INT32=%d UINT32=%d, INT64=%d or UINT64=%d, received type=%d", NPY_INT, NPY_UINT32, NPY_INT64, NPY_UINT64, type).c_str());
		if(type == NPY_UINT64)
		{
			long long* c_out = reinterpret_cast<long long*>(PyArray_DATA(np_ret));
			for (int i = 0; i < len1*len2; i+=2)
			{
				matches.push_back(cv::DMatch(c_out[i], c_out[i+1], 0));
			}
		}
		if(type == NPY_INT64)
		{
			unsigned long long* c_out = reinterpret_cast<unsigned long long*>(PyArray_DATA(np_ret));
			for (int i = 0; i < len1*len2; i+=2)
			{
				matches.push_back(cv::DMatch(c_out[i], c_out[i+1], 0));
			}
		}
		else if(type == NPY_UINT32)
		{
			unsigned int* c_out = reinterpret_cast<unsigned int*>(PyArray_DATA(np_ret));
			for (int i = 0; i < len1*len2; i+=2)
			{
				matches.push_back(cv::DMatch(c_out[i], c_out[i+1], 0));
			}
		}
		else // NPY_INT
		{
			int* c_out = reinterpret_cast<int*>(PyArray_DATA(np_ret));
			for (int i = 0; i < len1*len2; i+=2)
			{
				matches.push_back(cv::DMatch(c_out[i], c_out[i+1], 0));
			}
		}
		Py_DECREF(pReturn);
	}

	Py_DECREF(pDescriptorsQuery);
	Py_DECREF(pDescriptorsTrain);
	Py_DECREF(pKeypointsQuery);
	Py_DECREF(pkeypointsTrain);
	Py_DECREF(pScoresQuery);
	Py_DECREF(pScoresTrain);
	Py_DECREF(pImageWidth);
	Py_DECREF(pImageHeight);

	UDEBUG("Fill matches (%d/%d) and cleanup time = %fs", matches.size(), std::min(descriptorsQuery.rows, descriptorsTrain.rows), timer.ticks());
}

}
