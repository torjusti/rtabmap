// COLMAP-style diminishing-return image coverage selection for global BA.
// Algorithm reference: COLMAP scene/reconstruction_pruning.cc,
// FindRedundantPoints3D (8x8 cells, marginal inverse-square-root gain).
// This helper contains no optimizer or camera assumptions. Bin IDs must be
// unique per image/camera/cell; each track must have at most one hit per image.
#ifndef RTABMAP_BA_LANDMARK_SELECTION_H
#define RTABMAP_BA_LANDMARK_SELECTION_H

#include <cmath>
#include <cstddef>
#include <functional>
#include <queue>
#include <utility>
#include <vector>

namespace rtabmap {
namespace ba {
struct CoverageTrack
{
	int id;
	std::vector<size_t> bins;
};

// Returns selected IDs in greedy order. Equal gains prefer larger IDs, as in
// COLMAP. Lazy updates are exact because counts only increase and gains fall.
inline std::vector<int> selectCoverageTracks(
		const std::vector<CoverageTrack> & tracks, size_t numBins, double minGain,
		const std::function<bool()> & keepGoing = std::function<bool()>())
{
	std::vector<size_t> counts(numBins, 0);
	auto gain = [&counts](const CoverageTrack & track) {
		double result = 0.0;
		for(size_t bin : track.bins)
		{
			const double n = double(counts.at(bin));
			result += 1.0/std::sqrt(n+1.0)-1.0/std::sqrt(n+2.0);
		}
		return result;
	};
	struct Candidate
	{
		double gain;
		int id;
		size_t index;
		bool operator<(const Candidate & other) const
		{
			return gain < other.gain || (gain == other.gain && id < other.id);
		}
	};
	std::vector<Candidate> initial;
	initial.reserve(tracks.size());
	for(size_t i=0; i<tracks.size(); ++i)
		initial.push_back(Candidate{gain(tracks[i]), tracks[i].id, i});
	std::priority_queue<Candidate> heap(std::less<Candidate>(), std::move(initial));
	std::vector<int> selected;
	size_t pops = 0;
	while(!heap.empty())
	{
		if(keepGoing && (++pops % 65536 == 0) && !keepGoing()) return {};
		Candidate candidate = heap.top();
		heap.pop();
		if(candidate.gain <= minGain) break;
		const CoverageTrack & track = tracks[candidate.index];
		const double currentGain = gain(track);
		if(currentGain < candidate.gain)
		{
			candidate.gain = currentGain;
			heap.push(candidate);
		}
		else
		{
			selected.push_back(candidate.id);
			for(size_t bin : track.bins) ++counts[bin];
		}
	}
	return selected;
}
} // namespace ba
} // namespace rtabmap
#endif
