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

#ifndef CORELIB_INCLUDE_RTABMAP_CORE_VOCABULARYBUILDER_H_
#define CORELIB_INCLUDE_RTABMAP_CORE_VOCABULARYBUILDER_H_

#include "rtabmap/core/rtabmap_core_export.h" // DLL export/import defines

#include <opencv2/core/core.hpp>
#include <list>
#include <string>

namespace rtabmap {

/**
 * @class VocabularyBuilder
 * @brief Builds a fixed visual vocabulary out of the descriptors of existing databases
 *
 * The vocabulary is a plain set of centroids, written as the Word table of a
 * database that Kp/DictionaryPath accepts. Quantizing against it instead of
 * growing a dictionary online removes the whole vocabulary lifecycle from the
 * mapping loop (word insertion, pruning, index rebuilds, saving and reloading),
 * at the cost of a preprocessing pass over descriptors that are already stored.
 *
 * The centroids come from k-means, i.e. a classic (Sivic-Zisserman) bag of
 * words: descriptors are sampled from the input databases and clustered down to
 * the target vocabulary size. Clustering is done with Lloyd's algorithm, the
 * assignment step being an approximate kd-tree search rather than an exhaustive
 * one, which is what makes a vocabulary of a million words tractable. The
 * vocabulary is flat, unlike the hierarchical vocabulary trees of the DBoW
 * family: rtabmap searches words through a kd-tree index anyway, so there is
 * nothing to gain from a tree of centroids here.
 *
 * Only float descriptors are supported. Binary descriptors would need their
 * centroids to be binary too, which plain k-means cannot give.
 */
class RTABMAP_CORE_EXPORT VocabularyBuilder
{
public:
	/**
	 * @brief Read descriptors from the nodes of databases, uniformly at random
	 * @param databasePaths Databases to read, which must keep their raw
	 *        descriptors (Mem/RawDescriptorsKept)
	 * @param maxDescriptors Number of descriptors to stop at. Nodes are drawn
	 *        whole, so the count is only approached, never exceeded.
	 * @param seed RNG seed, so that the same databases give the same sample
	 * @return One descriptor per row, empty when none could be read
	 *
	 * Whole nodes are sampled rather than individual descriptors: the
	 * descriptors of a node are read in one indexed query, where drawing them
	 * one by one would mean scanning the feature table of every database.
	 */
	static cv::Mat sampleDescriptors(
			const std::list<std::string> & databasePaths,
			int maxDescriptors,
			unsigned int seed = 42);

	/**
	 * @brief Cluster descriptors into a vocabulary with Lloyd's algorithm
	 * @param descriptors One descriptor per row, CV_32FC1
	 * @param size Target vocabulary size, which must be under the number of
	 *        descriptors
	 * @param iterations Lloyd iterations. The first one already gives a usable
	 *        vocabulary, the following ones refine it with diminishing returns.
	 * @param checks Leaves visited by the approximate assignment search, see
	 *        FlannIndex::knnSearch()
	 * @param seed RNG seed used to draw the initial centroids
	 * @return One centroid per row, empty on invalid input
	 *
	 * Centroids left without any descriptor assigned to them are kept as they
	 * are rather than re-seeded: they are words that will simply never be
	 * matched, and dropping them would renumber the others.
	 */
	static cv::Mat cluster(
			const cv::Mat & descriptors,
			int size,
			int iterations = 4,
			int checks = 32,
			unsigned int seed = 42);

	/**
	 * @brief Write a vocabulary to a database usable as Kp/DictionaryPath
	 * @param databasePath Database to create, replacing any existing one
	 * @param words One word per row, as given by cluster()
	 * @return False if the database couldn't be created
	 */
	static bool save(const std::string & databasePath, const cv::Mat & words);

	/**
	 * @brief Sample, cluster and save in one pass, see the functions above
	 * @param databasePaths Databases to take the descriptors from
	 * @param outputDatabasePath Vocabulary database to create
	 * @param size Target vocabulary size
	 * @param samplesPerWord Descriptors to sample for each word of the target
	 *        size, the sample being what clustering time and memory depend on
	 * @param iterations Lloyd iterations
	 * @param checks Leaves visited by the approximate assignment search
	 * @param seed RNG seed
	 * @return False when the vocabulary couldn't be built or saved
	 */
	static bool build(
			const std::list<std::string> & databasePaths,
			const std::string & outputDatabasePath,
			int size,
			int samplesPerWord = 10,
			int iterations = 4,
			int checks = 32,
			unsigned int seed = 42);
};

} // namespace rtabmap

#endif /* CORELIB_INCLUDE_RTABMAP_CORE_VOCABULARYBUILDER_H_ */
