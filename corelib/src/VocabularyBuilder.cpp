/*
Copyright (c) 2010-2016, Mathieu Labbe - IntRoLab - Universite de Sherbrooke
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Universite de Sherbrooke nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "rtabmap/core/VocabularyBuilder.h"

#include "rtabmap/core/DBDriver.h"
#include "rtabmap/core/FlannIndex.h"
#include "rtabmap/core/VisualWord.h"
#include "rtabmap/core/VWDictionary.h"
#include "rtabmap/utilite/ULogger.h"
#include "rtabmap/utilite/UTimer.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace rtabmap {

cv::Mat VocabularyBuilder::sampleDescriptors(
		const std::list<std::string> & databasePaths,
		int maxDescriptors,
		unsigned int seed)
{
	if(databasePaths.empty() || maxDescriptors <= 0)
	{
		UERROR("No database to sample from, or invalid sample size (%d).", maxDescriptors);
		return cv::Mat();
	}

	// Draw the nodes to read first, so that the sample is spread over all the
	// databases and not taken from the beginning of the first one.
	std::vector<std::vector<int> > nodeIds(databasePaths.size());
	std::vector<std::pair<size_t, int> > allNodes; // database index, node id
	{
		size_t dbIndex = 0;
		for(std::list<std::string>::const_iterator iter=databasePaths.begin(); iter!=databasePaths.end(); ++iter, ++dbIndex)
		{
			DBDriver * driver = DBDriver::create();
			if(!driver->openConnection(*iter, false))
			{
				UERROR("Could not open \"%s\" to sample descriptors from.", iter->c_str());
				delete driver;
				return cv::Mat();
			}
			std::set<int> ids;
			driver->getAllNodeIds(ids, false, false, false);
			driver->closeConnection(false);
			delete driver;

			for(std::set<int>::const_iterator jter=ids.begin(); jter!=ids.end(); ++jter)
			{
				allNodes.push_back(std::make_pair(dbIndex, *jter));
			}
		}
	}
	if(allNodes.empty())
	{
		UERROR("The databases to sample descriptors from don't have any node.");
		return cv::Mat();
	}

	std::mt19937 rng(seed);
	std::shuffle(allNodes.begin(), allNodes.end(), rng);
	for(size_t i=0; i<allNodes.size(); ++i)
	{
		nodeIds[allNodes[i].first].push_back(allNodes[i].second);
	}

	cv::Mat sample;
	int rows = 0;
	UTimer timer;
	size_t dbIndex = 0;
	for(std::list<std::string>::const_iterator iter=databasePaths.begin();
		iter!=databasePaths.end() && (sample.empty() || rows < sample.rows);
		++iter, ++dbIndex)
	{
		DBDriver * driver = DBDriver::create();
		if(!driver->openConnection(*iter, false))
		{
			UERROR("Could not open \"%s\" to sample descriptors from.", iter->c_str());
			delete driver;
			return cv::Mat();
		}

		const std::vector<int> & ids = nodeIds[dbIndex];
		for(size_t i=0; i<ids.size() && (sample.empty() || rows < sample.rows); ++i)
		{
			std::multimap<int, int> words;
			std::vector<cv::KeyPoint> keypoints;
			std::vector<cv::Point3f> points;
			cv::Mat descriptors;
			driver->getLocalFeatures(ids[i], words, keypoints, points, descriptors);
			if(descriptors.empty())
			{
				continue;
			}
			if(descriptors.type() != CV_32FC1)
			{
				UERROR("Only float descriptors can be clustered into a vocabulary, "
					   "but \"%s\" has descriptors of type %d.", iter->c_str(), descriptors.type());
				driver->closeConnection(false);
				delete driver;
				return cv::Mat();
			}
			if(sample.empty())
			{
				sample = cv::Mat(maxDescriptors, descriptors.cols, CV_32FC1);
				UWARN("Sampling up to %d descriptors of dimension %d (%.1f GB) from %d nodes...",
						maxDescriptors, descriptors.cols,
						double(maxDescriptors) * descriptors.cols * sizeof(float) / 1e9,
						(int)allNodes.size());
			}
			else if(descriptors.cols != sample.cols)
			{
				UERROR("Descriptors of \"%s\" have dimension %d, where the previous ones had %d: "
					   "the databases were not made with the same feature type.",
						iter->c_str(), descriptors.cols, sample.cols);
				driver->closeConnection(false);
				delete driver;
				return cv::Mat();
			}

			const int taken = std::min(descriptors.rows, sample.rows - rows);
			descriptors.rowRange(0, taken).copyTo(sample.rowRange(rows, rows + taken));
			rows += taken;
		}

		driver->closeConnection(false);
		delete driver;
	}

	if(rows == 0)
	{
		UERROR("No descriptor could be read: the databases were likely created without "
			   "keeping their raw descriptors (see Mem/RawDescriptorsKept).");
		return cv::Mat();
	}
	UWARN("Sampled %d descriptors in %.1fs.", rows, timer.ticks());
	// Copied out when short, so that the unused part of the allocation, which
	// a view would keep referenced, is freed.
	return rows == sample.rows ? sample : sample.rowRange(0, rows).clone();
}

cv::Mat VocabularyBuilder::cluster(
		const cv::Mat & descriptors,
		int size,
		int iterations,
		int checks,
		unsigned int seed)
{
	if(descriptors.empty() || descriptors.type() != CV_32FC1)
	{
		UERROR("Clustering needs float descriptors.");
		return cv::Mat();
	}
	if(size <= 0 || size >= descriptors.rows)
	{
		UERROR("Cannot cluster %d descriptors into %d words: the vocabulary has to be smaller "
			   "than the sample it is built from.", descriptors.rows, size);
		return cv::Mat();
	}
	if(iterations <= 0)
	{
		UERROR("At least one clustering iteration is needed, %d asked.", iterations);
		return cv::Mat();
	}

	// Seed the centroids with descriptors drawn at random (Forgy initialization).
	std::mt19937 rng(seed);
	std::vector<int> order(descriptors.rows);
	std::iota(order.begin(), order.end(), 0);
	std::shuffle(order.begin(), order.end(), rng);
	cv::Mat centroids(size, descriptors.cols, CV_32FC1);
	for(int i=0; i<size; ++i)
	{
		descriptors.row(order[i]).copyTo(centroids.row(i));
	}

	// Accumulate the sums in double: a centroid can end up with millions of
	// descriptors, which float would round away.
	std::vector<double> sums((size_t)size * descriptors.cols);
	std::vector<int> counts(size);
	UTimer timer;
	for(int iteration=0; iteration<iterations; ++iteration)
	{
		std::fill(sums.begin(), sums.end(), 0.0);
		std::fill(counts.begin(), counts.end(), 0);
		double inertia = 0.0;
		{
			// The index refers to the centroids without copying them, so it is
			// released before they are updated below.
			FlannIndex index;
			index.buildIndex(FlannIndex::FLANN_INDEX_KDTREE, centroids, false, 1.0f);

			const int batchSize = 1 << 20;
			for(int start=0; start<descriptors.rows; start+=batchSize)
			{
				const cv::Mat batch = descriptors.rowRange(start, std::min(start + batchSize, descriptors.rows));
				cv::Mat indices, dists;
				index.knnSearch(batch, indices, dists, 1, checks, 0.0f, true, 0); // 0 = all cores
				for(int i=0; i<batch.rows; ++i)
				{
					const int centroid = indices.at<int>(i, 0);
					if(centroid < 0)
					{
						continue;
					}
					const float * descriptor = batch.ptr<float>(i);
					double * sum = &sums[(size_t)centroid * descriptors.cols];
					for(int j=0; j<descriptors.cols; ++j)
					{
						sum[j] += descriptor[j];
					}
					++counts[centroid];
					inertia += dists.at<float>(i, 0);
				}
			}
		}

		int empty = 0;
		for(int i=0; i<size; ++i)
		{
			if(counts[i] == 0)
			{
				++empty;
				continue;
			}
			float * centroid = centroids.ptr<float>(i);
			const double * sum = &sums[(size_t)i * descriptors.cols];
			for(int j=0; j<descriptors.cols; ++j)
			{
				centroid[j] = float(sum[j] / counts[i]);
			}
		}
		UWARN("Clustering iteration %d/%d: inertia=%.3e, %d empty words, %.1fs.",
				iteration + 1, iterations, inertia, empty, timer.ticks());
	}
	return centroids;
}

bool VocabularyBuilder::save(const std::string & databasePath, const cv::Mat & words)
{
	if(words.empty())
	{
		UERROR("Nothing to save, the vocabulary is empty.");
		return false;
	}

	UTimer timer;
	DBDriver * driver = DBDriver::create();
	if(!driver->openConnection(databasePath, true))
	{
		UERROR("Could not create the vocabulary database \"%s\".", databasePath.c_str());
		delete driver;
		return false;
	}
	for(int i=0; i<words.rows; ++i)
	{
		driver->asyncSave(new VisualWord(VWDictionary::ID_START + i, words.row(i))); // the driver takes ownership
		if((i + 1) % 200000 == 0)
		{
			driver->emptyTrashes();
			UWARN("Saved %d/%d words...", i + 1, words.rows);
		}
	}
	driver->emptyTrashes();
	driver->closeConnection(true);
	delete driver;
	UWARN("Wrote %d words to \"%s\" in %.1fs.", words.rows, databasePath.c_str(), timer.ticks());
	return true;
}

bool VocabularyBuilder::build(
		const std::list<std::string> & databasePaths,
		const std::string & outputDatabasePath,
		int size,
		int samplesPerWord,
		int iterations,
		int checks,
		unsigned int seed)
{
	if(size <= 0 || samplesPerWord <= 0)
	{
		UERROR("Invalid vocabulary size (%d) or samples per word (%d).", size, samplesPerWord);
		return false;
	}

	// The sample has to be larger than the vocabulary for clustering to mean
	// anything, which samplesPerWord guarantees as long as the databases hold
	// that many descriptors.
	const double samples = double(size) * samplesPerWord;
	if(samples > double(std::numeric_limits<int>::max()))
	{
		UERROR("A vocabulary of %d words with %d samples per word needs more descriptors "
			   "than can be held at once.", size, samplesPerWord);
		return false;
	}

	UTimer timer;
	const cv::Mat descriptors = sampleDescriptors(databasePaths, int(samples), seed);
	if(descriptors.empty())
	{
		return false;
	}
	if(descriptors.rows <= size)
	{
		UERROR("Only %d descriptors could be sampled for a vocabulary of %d words: "
			   "ask for a smaller vocabulary.", descriptors.rows, size);
		return false;
	}

	const cv::Mat words = cluster(descriptors, size, iterations, checks, seed);
	if(words.empty())
	{
		return false;
	}
	if(!save(outputDatabasePath, words))
	{
		return false;
	}
	UWARN("Built a vocabulary of %d words in %.0fs.", words.rows, timer.elapsed());
	return true;
}

} // namespace rtabmap
