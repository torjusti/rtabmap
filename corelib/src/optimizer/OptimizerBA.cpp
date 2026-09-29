#include <rtabmap/core/BAPairSelection.h>
#include <rtabmap/core/Optimizer.h>
#include <rtabmap/core/DBDriver.h>
#include <rtabmap/core/Graph.h>
#include <rtabmap/core/ProgressState.h>
#include <rtabmap/core/RegistrationVis.h>
#include <rtabmap/core/util3d.h>
#include <rtabmap/core/util3d_transforms.h>
#include <rtabmap/utilite/UConversion.h>
#include <rtabmap/utilite/ULogger.h>
#include <rtabmap/utilite/UMath.h>
#include <rtabmap/utilite/UStl.h>
#include <rtabmap/utilite/UTimer.h>
#include <algorithm>
#include <cmath>
#include <iterator>
#include <list>
#include <memory>
#include <queue>
#include <set>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace rtabmap {

struct KeyPointCompare
{
	bool operator() (const cv::KeyPoint& lhs, const cv::KeyPoint& rhs) const
	{
		return lhs.pt.x < rhs.pt.x || (lhs.pt.x == rhs.pt.x && lhs.pt.y < rhs.pt.y);
	}
};

class BAFrameSource
{
public:
	virtual ~BAFrameSource() {}

	/** The frames this source can serve, ascending. */
	virtual const std::set<int> & ids() const = 0;
	virtual bool has(int id) const = 0;

	// --- the resident half ---
	virtual int weight(int id) const = 0;
	virtual const std::vector<CameraModel> & cameraModels(int id) const = 0;
	virtual const std::vector<StereoCameraModel> & stereoModels(int id) const = 0;
	virtual const std::vector<cv::KeyPoint> & keypoints(int id) const = 0;
	virtual const std::vector<cv::Point3f> & words3(int id) const = 0;
	/**
	 * Union-find slot of each of the frame's features, -1 until that feature is first
	 * matched. One int per feature is what replaced a std::map keyed on the keypoint
	 * itself: same identity, 4 bytes instead of some 80, which on this map is the
	 * difference between 190MB and 3.7GB.
	 */
	virtual std::vector<int> & trackSlots(int id) = 0;

	// --- the transient half ---
	/**
	 * Make @p ids' descriptors resident and keep them so until the next call. Frames
	 * outside @p ids may be evicted to stay within budget; frames inside it never are.
	 */
	virtual void pin(const std::set<int> & ids) = 0;
	/** A pinned frame, or 0. Never loads, so it is safe to call concurrently. */
	virtual const Signature * pinned(int id) const = 0;
	/** A frame, loading it if it is not resident. Not safe to call concurrently. */
	virtual const Signature * fetch(int id) = 0;

	virtual size_t residentBytes() const {return 0;}
	virtual size_t loads() const {return 0;}
};

namespace {

/** The half of a frame that outlives the matching of any one pair. */
struct BAFrameGeometry
{
	BAFrameGeometry() : weight(0) {}
	std::vector<CameraModel> cameraModels;
	std::vector<StereoCameraModel> stereoModels;
	std::vector<cv::KeyPoint> kpts;
	std::vector<cv::Point3f> words3;
	std::vector<int> slots;
	int weight;
};

/**
 * Frames the caller already holds. Nothing is copied and nothing is ever evicted; the
 * source only adds the per-feature union-find slots, which the signatures have nowhere to
 * put. This is the path the local window refinements take, where the caller has already
 * bounded what it loaded.
 */
class MemoryFrameSource : public BAFrameSource
{
public:
	explicit MemoryFrameSource(const std::map<int, Signature> & signatures) :
		signatures_(signatures)
	{
		for(std::map<int, Signature>::const_iterator iter=signatures_.begin(); iter!=signatures_.end(); ++iter)
		{
			ids_.insert(iter->first);
		}
	}

	virtual const std::set<int> & ids() const {return ids_;}
	virtual bool has(int id) const {return signatures_.find(id) != signatures_.end();}
	virtual int weight(int id) const {return signatures_.at(id).getWeight();}
	virtual const std::vector<CameraModel> & cameraModels(int id) const
	{
		return signatures_.at(id).sensorData().cameraModels();
	}
	virtual const std::vector<StereoCameraModel> & stereoModels(int id) const
	{
		return signatures_.at(id).sensorData().stereoCameraModels();
	}
	virtual const std::vector<cv::KeyPoint> & keypoints(int id) const {return signatures_.at(id).getWordsKpts();}
	virtual const std::vector<cv::Point3f> & words3(int id) const {return signatures_.at(id).getWords3();}
	virtual std::vector<int> & trackSlots(int id)
	{
		std::map<int, std::vector<int> >::iterator iter = slots_.find(id);
		if(iter == slots_.end())
		{
			iter = slots_.insert(std::make_pair(id,
					std::vector<int>(signatures_.at(id).getWordsKpts().size(), -1))).first;
		}
		return iter->second;
	}
	virtual void pin(const std::set<int> &) {}
	virtual const Signature * pinned(int id) const
	{
		std::map<int, Signature>::const_iterator iter = signatures_.find(id);
		return iter == signatures_.end()? 0 : &iter->second;
	}
	virtual const Signature * fetch(int id) {return pinned(id);}

private:
	const std::map<int, Signature> & signatures_;
	std::set<int> ids_;
	std::map<int, std::vector<int> > slots_;
};

/** Bytes a resident signature costs, dominated in every real case by the descriptors. */
size_t signatureBytes(const Signature & s)
{
	const cv::Mat & d = s.getWordsDescriptors();
	return (size_t)d.total()*d.elemSize()
		 + s.getWordsKpts().size()*sizeof(cv::KeyPoint)
		 + s.getWords3().size()*sizeof(cv::Point3f)
		 + s.getWords().size()*48;  // std::multimap node
}

/**
 * Frames read from a database: geometry once, descriptors a batch at a time.
 *
 * The geometry pass reads every frame's keypoints and 3D words with the descriptor column
 * left out of the query entirely -- not read and discarded, never read -- because that
 * column is some 97% of the Feature table. What it builds is a few tens of bytes per
 * feature and stays for the whole run.
 *
 * Descriptors then arrive through @ref pin, which loads the batch of frames the caller is
 * about to match and evicts least-recently-pinned frames until the byte budget is met.
 * The pinned batch itself is exempt: a budget that could evict what the caller just asked
 * for would be a correctness bug rather than a memory policy, so a batch larger than the
 * budget simply overshoots it and says so.
 */
class DatabaseFrameSource : public BAFrameSource
{
public:
	DatabaseFrameSource(DBDriver * db, const std::set<int> & ids, size_t budgetBytes, Optimizer * progress = 0) :
		db_(db), progress_(progress), budgetBytes_(budgetBytes), residentBytes_(0), loads_(0), pinnedCount_(0),
		overBudgetWarned_(false)
	{
		UASSERT(db_ != 0);
		loadGeometry(ids);
	}

	virtual const std::set<int> & ids() const {return ids_;}
	virtual bool has(int id) const {return geometry_.find(id) != geometry_.end();}
	virtual int weight(int id) const {return geometry_.at(id).weight;}
	virtual const std::vector<CameraModel> & cameraModels(int id) const {return geometry_.at(id).cameraModels;}
	virtual const std::vector<StereoCameraModel> & stereoModels(int id) const {return geometry_.at(id).stereoModels;}
	virtual const std::vector<cv::KeyPoint> & keypoints(int id) const {return geometry_.at(id).kpts;}
	virtual const std::vector<cv::Point3f> & words3(int id) const {return geometry_.at(id).words3;}
	virtual std::vector<int> & trackSlots(int id) {return geometry_.at(id).slots;}

	virtual void pin(const std::set<int> & ids)
	{
		std::list<int> missing;
		for(std::set<int>::const_iterator iter=ids.begin(); iter!=ids.end(); ++iter)
		{
			if(has(*iter) && cache_.find(*iter) == cache_.end())
			{
				missing.push_back(*iter);
			}
		}
		load(missing);

		// Everything asked for goes to the front of the eviction order, so the loop below
		// can only ever reach frames from earlier batches.
		for(std::set<int>::const_iterator iter=ids.begin(); iter!=ids.end(); ++iter)
		{
			std::map<int, Entry>::iterator e = cache_.find(*iter);
			if(e != cache_.end())
			{
				order_.splice(order_.begin(), order_, e->second.position);
			}
		}
		pinnedCount_ = 0;
		for(std::set<int>::const_iterator iter=ids.begin(); iter!=ids.end(); ++iter)
		{
			if(cache_.find(*iter) != cache_.end())
			{
				++pinnedCount_;
			}
		}
		evict(pinnedCount_);
	}

	virtual const Signature * pinned(int id) const
	{
		std::map<int, Entry>::const_iterator iter = cache_.find(id);
		return iter == cache_.end()? 0 : &iter->second.signature;
	}

	virtual const Signature * fetch(int id)
	{
		std::map<int, Entry>::iterator iter = cache_.find(id);
		if(iter == cache_.end())
		{
			if(!has(id))
			{
				return 0;
			}
			std::list<int> one(1, id);
			load(one);
			iter = cache_.find(id);
			if(iter == cache_.end())
			{
				return 0;
			}
		}
		order_.splice(order_.begin(), order_, iter->second.position);
		// One protected entry: the frame just asked for. Without that, a single frame
		// larger than the whole budget would be evicted between being loaded and being
		// returned, and the caller would be handed a pointer into a destroyed entry.
		evict(1);
		return &iter->second.signature;
	}

	virtual size_t residentBytes() const {return residentBytes_;}
	virtual size_t loads() const {return loads_;}

private:
	struct Entry
	{
		Entry() : bytes(0) {}
		Signature signature;
		size_t bytes;
		std::list<int>::iterator position;
	};

	void loadGeometry(const std::set<int> & ids)
	{
		UTimer timer;
		size_t features = 0;
		int done = 0;
		const int total = (int)ids.size();
		double lastReport = -1.0;
		if(progress_ && !progress_->reportProgress(uFormat("Frame geometry 0/%d", total)))
		{
			progress_->cancel();
			return;
		}
		for(std::set<int>::const_iterator iter=ids.begin(); iter!=ids.end(); ++iter)
		{
			if(*iter <= 0)
			{
				continue;   // landmarks
			}
			++done;
			std::vector<CameraModel> models;
			std::vector<StereoCameraModel> stereoModels;
			db_->getCalibration(*iter, models, stereoModels);
			if(models.empty() && stereoModels.empty())
			{
				continue;   // uncalibrated frames can carry no observation
			}
			BAFrameGeometry & g = geometry_[*iter];
			g.cameraModels.swap(models);
			g.stereoModels.swap(stereoModels);
			db_->getWeight(*iter, g.weight);
			std::multimap<int, int> words;
			cv::Mat noDescriptors;
			db_->getLocalFeatures(*iter, words, g.kpts, g.words3, noDescriptors, false);
			g.slots.assign(g.kpts.size(), -1);
			features += g.kpts.size();
			ids_.insert(*iter);
			if(progress_ && (done == total || timer.elapsed() - lastReport > 0.05))
			{
				lastReport = timer.elapsed();
				if(!progress_->reportProgress(uFormat("Frame geometry %d/%d", done, total)))
				{
					progress_->cancel();
					return;
				}
			}
		}
		if(progress_)
		{
			if(!progress_->reportProgress(uFormat("Frame geometry %d/%d", done, total)))
			{
				progress_->cancel();
				return;
			}
		}
		UINFO("BA frames: geometry of %d frames / %d features read in %.2fs, descriptors excluded "
			  "(about %d MB resident; the same frames with descriptors would be far more).",
				(int)ids_.size(), (int)features, timer.elapsed(),
				(int)(features*(sizeof(cv::KeyPoint)+sizeof(cv::Point3f)+sizeof(int))/(1024*1024)));
		if(progress_)
		{
			progress_->reportProgress(uFormat(
					"Frame geometry loaded: %d frames, %d features (%.1fs)",
					(int)ids_.size(), (int)features, timer.elapsed()));
		}
	}

	void load(const std::list<int> & missing)
	{
		if(missing.empty())
		{
			return;
		}
		std::list<Signature *> loaded;
		db_->loadSignatures(missing, loaded);
		for(std::list<Signature *>::iterator iter=loaded.begin(); iter!=loaded.end(); ++iter)
		{
			if(*iter == 0)
			{
				continue;
			}
			const int id = (*iter)->id();
			Entry & e = cache_[id];
			e.signature = **iter;
			e.bytes = signatureBytes(e.signature);
			residentBytes_ += e.bytes;
			order_.push_front(id);
			e.position = order_.begin();
			delete *iter;
			++loads_;
		}
	}

	/** Evicts least-recently-pinned frames, never the @p protectedFront most recent ones. */
	void evict(size_t protectedFront)
	{
		while(residentBytes_ > budgetBytes_ && order_.size() > protectedFront)
		{
			const int victim = order_.back();
			order_.pop_back();
			std::map<int, Entry>::iterator e = cache_.find(victim);
			if(e != cache_.end())
			{
				residentBytes_ -= e->second.bytes;
				cache_.erase(e);
			}
		}
		if(residentBytes_ > budgetBytes_ && !overBudgetWarned_)
		{
			overBudgetWarned_ = true;
			UWARN("BA frames: one batch of pairs needs %d MB of descriptors, over the %d MB "
				  "budget of %s. The batch is kept -- evicting it would break the matching -- "
				  "so raise the budget or expect this peak.",
					(int)(residentBytes_/(1024*1024)), (int)(budgetBytes_/(1024*1024)),
					Parameters::kOptimizerBAFrameCacheMemory().c_str());
		}
	}

	DBDriver * db_;
	Optimizer * progress_;
	size_t budgetBytes_;
	size_t residentBytes_;
	size_t loads_;
	size_t pinnedCount_;
	bool overBudgetWarned_;
	std::set<int> ids_;
	std::map<int, BAFrameGeometry> geometry_;
	std::map<int, Entry> cache_;
	std::list<int> order_;   // most recently pinned first
};

/**
 * Union-find over observation nodes, with path halving and union by size.
 *
 * This is what the legacy pairwise builder lacks: there, when a correspondence
 * joins two keypoints that already belong to *different* words, the two words
 * are never merged and one landmark ends up split into two 3D points with
 * disjoint observations.
 */
class ObservationUnionFind
{
public:
	int add()
	{
		parent_.push_back((int)parent_.size());
		size_.push_back(1);
		return (int)parent_.size()-1;
	}
	int find(int x)
	{
		while(parent_[x] != x)
		{
			parent_[x] = parent_[parent_[x]]; // path halving
			x = parent_[x];
		}
		return x;
	}
	void unite(int a, int b)
	{
		a = find(a);
		b = find(b);
		if(a == b)
		{
			return;
		}
		if(size_[a] < size_[b])
		{
			std::swap(a, b);
		}
		parent_[b] = a;
		size_[a] += size_[b];
	}
private:
	std::vector<int> parent_;
	std::vector<int> size_;
};

/**
 * One observation: a frame, and which of that frame's features it is.
 *
 * Deliberately eight bytes and no more. The observation table is the one structure with
 * an entry per matched feature -- tens of millions of them on a large map -- so anything
 * carried here rather than looked up costs gigabytes. The previous version held a
 * keypoint, a depth, a camera index, a world point and a cv::Mat descriptor; the
 * descriptor in particular was a row view, which shares its parent's reference count and
 * so pinned that frame's *entire* descriptor matrix for as long as any one of its
 * features remained in any track.
 */
struct TrackObservation
{
	TrackObservation() : frameId(0), featureIndex(0) {}
	TrackObservation(int f, int i) : frameId(f), featureIndex(i) {}
	int frameId;
	int featureIndex;   // position in the frame's own feature array, i.e. in storage order
};

/** An observation resolved against its frame's geometry, ready for triangulation or BA. */
struct ResolvedObservation
{
	ResolvedObservation() : depth(0.0f), cameraIndex(0), hasDepth(false) {}
	cv::KeyPoint kpt;     // sub-image adjusted for multi-camera rigs
	float depth;          // in the observing camera's frame, 0 if unknown
	int cameraIndex;
	bool hasDepth;
};

/**
 * Resolve observation @p obs against the frame geometry held by @p frames.
 *
 * Unlike the legacy builder this does not require a valid 3D point: an observation with
 * no usable depth is still resolved (depth stays 0, which the back-ends route to a mono
 * reprojection edge) so it can still lengthen a track.
 */
bool resolveObservation(
		BAFrameSource & frames,
		const TrackObservation & obs,
		ResolvedObservation & out)
{
	if(!frames.has(obs.frameId))
	{
		return false;
	}
	const std::vector<cv::KeyPoint> & kpts = frames.keypoints(obs.frameId);
	if(obs.featureIndex < 0 || obs.featureIndex >= (int)kpts.size())
	{
		return false;
	}
	const std::vector<CameraModel> & models = frames.cameraModels(obs.frameId);
	const std::vector<StereoCameraModel> & stereo = frames.stereoModels(obs.frameId);

	cv::KeyPoint kpt = kpts[obs.featureIndex];
	int cameraIndex = 0;
	if(models.size() > 1 || stereo.size() > 1)
	{
		const float subImageWidth = models.size()>1?
				models[0].imageWidth() : stereo[0].left().imageWidth();
		if(subImageWidth <= 0.0f)
		{
			return false;
		}
		cameraIndex = int(kpt.pt.x / subImageWidth);
		kpt.pt.x = kpt.pt.x - (subImageWidth*float(cameraIndex));
	}

	Transform localTransform;
	if(!models.empty())
	{
		if(cameraIndex >= (int)models.size())
		{
			return false;
		}
		localTransform = models[cameraIndex].localTransform();
	}
	else
	{
		if(stereo.empty() || cameraIndex >= (int)stereo.size())
		{
			return false;
		}
		localTransform = stereo[cameraIndex].localTransform();
	}
	if(localTransform.isNull())
	{
		return false;
	}

	out.kpt = kpt;
	out.cameraIndex = cameraIndex;
	out.depth = 0.0f;

	// The legacy builder tests "p.x > 0", which encodes "the camera looks along +x" in the
	// base frame and silently drops every point of a rear- or side-facing camera on a
	// multi-camera rig. Test finiteness instead.
	const std::vector<cv::Point3f> & pts3 = frames.words3(obs.frameId);
	const cv::Point3f p = obs.featureIndex < (int)pts3.size()?
			pts3[obs.featureIndex] : cv::Point3f(0,0,0);
	out.hasDepth = util3d::isFinite(p) && !(p.x == 0.0f && p.y == 0.0f && p.z == 0.0f);
	if(out.hasDepth)
	{
		out.depth = util3d::transformPoint(p, localTransform.inverse()).z;
	}
	return true;
}

/**
 * Look up the union-find slot of (@p frameId, @p featureIndex), creating it the first time
 * that feature takes part in a correspondence.
 *
 * The frame's own feature index is the identity that survives across pairs: it is the
 * feature's position in the database, so it means the same thing in every call, whereas
 * the indices registration hands back address a filtered array it rebuilt for that one
 * call. Slots are created lazily because most features never match anything.
 */
int getOrCreateObservationNode(
		BAFrameSource & frames,
		int frameId,
		int featureIndex,
		std::vector<TrackObservation> & observations,
		ObservationUnionFind & uf)
{
	std::vector<int> & slots = frames.trackSlots(frameId);
	if(featureIndex < 0 || featureIndex >= (int)slots.size())
	{
		return -1;
	}
	if(slots[featureIndex] >= 0)
	{
		return slots[featureIndex];
	}
	const int id = uf.add();
	observations.push_back(TrackObservation(frameId, featureIndex));
	slots[featureIndex] = id;
	return id;
}

float medianOf(std::vector<float> & values)
{
	UASSERT(!values.empty());
	std::sort(values.begin(), values.end());
	return values[values.size()/2];
}

/**
 * Resolve the camera an observation was taken with.
 *
 * @param[out] model         intrinsics of the observing camera (already the
 *                           correct one for a multi-camera rig).
 * @param[out] cameraToWorld optical-frame camera pose in world coordinates,
 *                           i.e. @c pose*localTransform.
 * @return false if the frame has no pose or no usable calibration.
 */
bool getObservationCamera(
		BAFrameSource & frames,
		const std::map<int, Transform> & poses,
		int frameId,
		int cameraIndex,
		CameraModel & model,
		Transform & cameraToWorld)
{
	std::map<int, Transform>::const_iterator pIter = poses.find(frameId);
	if(!frames.has(frameId) || pIter == poses.end() || pIter->second.isNull())
	{
		return false;
	}
	const std::vector<CameraModel> & models = frames.cameraModels(frameId);
	const std::vector<StereoCameraModel> & stereo = frames.stereoModels(frameId);
	if(!models.empty())
	{
		if(cameraIndex >= (int)models.size())
		{
			return false;
		}
		model = models[cameraIndex];
	}
	else if(!stereo.empty())
	{
		if(cameraIndex >= (int)stereo.size())
		{
			return false;
		}
		model = stereo[cameraIndex].left();
	}
	else
	{
		return false;
	}
	if(!model.isValidForProjection() || model.localTransform().isNull())
	{
		return false;
	}
	cameraToWorld = pIter->second * model.localTransform();
	return true;
}

/** One view of a track, resolved and ready for triangulation. */
struct TrackView
{
	Transform worldToCamera;  // optical frame
	cv::Point3f center;       // camera centre in world
	float fx, fy, cx, cy;
	float u, v;               // observed keypoint
	float measuredDepth;      // 0 if unknown
};

/**
 * Refine a 3D point by Gauss-Newton on the reprojection error, poses fixed.
 *
 * The DLT below minimizes an algebraic cost, so its output can carry a
 * noticeably larger geometric error than the point deserves. Without this the
 * reprojection gate would reject perfectly good tracks.
 */
void refinePointGaussNewton(const std::vector<TrackView> & views, cv::Point3f & point)
{
	for(int it=0; it<5; ++it)
	{
		cv::Matx33d H = cv::Matx33d::zeros();
		cv::Matx31d b = cv::Matx31d::zeros();
		for(size_t i=0; i<views.size(); ++i)
		{
			const TrackView & view = views[i];
			const cv::Point3f pc = util3d::transformPoint(point, view.worldToCamera);
			if(pc.z <= 0.0f)
			{
				return; // behind a camera: leave the point to the cheirality gate
			}
			const double invZ = 1.0/(double)pc.z;
			const double ru = (double)view.fx*(double)pc.x*invZ + (double)view.cx - (double)view.u;
			const double rv = (double)view.fy*(double)pc.y*invZ + (double)view.cy - (double)view.v;

			// d(projection)/d(point in camera frame)
			const double j00 = (double)view.fx*invZ;
			const double j02 = -(double)view.fx*(double)pc.x*invZ*invZ;
			const double j11 = (double)view.fy*invZ;
			const double j12 = -(double)view.fy*(double)pc.y*invZ*invZ;

			// chain through the world->camera rotation
			const Transform & t = view.worldToCamera;
			const double r[3][3] = {
					{(double)t.r11(), (double)t.r12(), (double)t.r13()},
					{(double)t.r21(), (double)t.r22(), (double)t.r23()},
					{(double)t.r31(), (double)t.r32(), (double)t.r33()}};

			double ju[3], jv[3];
			for(int k=0; k<3; ++k)
			{
				ju[k] = j00*r[0][k] + j02*r[2][k];
				jv[k] = j11*r[1][k] + j12*r[2][k];
			}
			for(int a=0; a<3; ++a)
			{
				for(int c=0; c<3; ++c)
				{
					H(a,c) += ju[a]*ju[c] + jv[a]*jv[c];
				}
				b(a,0) += ju[a]*ru + jv[a]*rv;
			}
		}

		cv::Matx31d delta;
		if(!cv::solve(cv::Mat(H), cv::Mat(-b), delta, cv::DECOMP_CHOLESKY))
		{
			return;
		}
		point.x += (float)delta(0,0);
		point.y += (float)delta(1,0);
		point.z += (float)delta(2,0);
		if(std::sqrt(delta(0,0)*delta(0,0) + delta(1,0)*delta(1,0) + delta(2,0)*delta(2,0)) < 1e-6)
		{
			return;
		}
	}
}

/** Result of triangulating one track, for gating and reporting. */
struct TriangulationResult
{
	TriangulationResult() : valid(false), maxReprojError(0.0f), maxAngleDeg(0.0f), medianDepthError(0.0f), hasDepth(false) {}
	cv::Point3f point;
	bool valid;
	float maxReprojError;    // pixels, worst view
	float maxAngleDeg;       // widest parallax between two views
	float medianDepthError;  // relative, vs measured depth
	bool hasDepth;           // at least one view had a depth measurement
};

/**
 * Multi-view DLT triangulation followed by a Gauss-Newton refinement, then the
 * quality measures the caller gates on.
 *
 * The DLT is solved in normalized image coordinates so the system stays well
 * conditioned regardless of focal length.
 */
/**
 * Cheirality check plus worst-case reprojection error for a candidate point.
 *
 * @return false if the point falls behind any camera, in which case @p maxReproj
 *         is not meaningful.
 */
bool scoreTrackPoint(const std::vector<TrackView> & views, const cv::Point3f & point, float & maxReproj)
{
	maxReproj = 0.0f;
	for(size_t i=0; i<views.size(); ++i)
	{
		const TrackView & view = views[i];
		const cv::Point3f pc = util3d::transformPoint(point, view.worldToCamera);
		if(pc.z <= 0.0f)
		{
			return false;
		}
		const float du = view.fx*pc.x/pc.z + view.cx - view.u;
		const float dv = view.fy*pc.y/pc.z + view.cy - view.v;
		maxReproj = std::max(maxReproj, std::sqrt(du*du + dv*dv));
	}
	return true;
}

/**
 * Seed a track from what the depth sensor actually measured.
 *
 * The DLT below triangulates from bearing rays alone, so its conditioning is set
 * by the parallax between views and nothing else. On slow motion in a small room
 * -- EuRoC V1_01, TUM desk -- that parallax is small and the algebraic solution
 * is poor, while the stereo/RGB-D depth of the very same keypoint is perfectly
 * usable. The legacy pairwise builder has no such problem precisely because it
 * places each landmark from the depth of its link's "from" frame.
 *
 * Each view carrying a depth is back-projected to world coordinates; the
 * component-wise median is returned, which ignores the views whose depth reading
 * is wrong rather than averaging them in.
 */
bool depthSeedTrackPoint(const std::vector<TrackView> & views, cv::Point3f & seed)
{
	std::vector<float> xs, ys, zs;
	xs.reserve(views.size()); ys.reserve(views.size()); zs.reserve(views.size());
	for(size_t i=0; i<views.size(); ++i)
	{
		const TrackView & view = views[i];
		if(view.measuredDepth <= 0.0f || view.fx <= 0.0f || view.fy <= 0.0f)
		{
			continue;
		}
		const cv::Point3f pc(
				(view.u - view.cx)*view.measuredDepth/view.fx,
				(view.v - view.cy)*view.measuredDepth/view.fy,
				view.measuredDepth);
		const cv::Point3f pw = util3d::transformPoint(pc, view.worldToCamera.inverse());
		if(!util3d::isFinite(pw))
		{
			continue;
		}
		xs.push_back(pw.x); ys.push_back(pw.y); zs.push_back(pw.z);
	}
	if(xs.empty())
	{
		return false;
	}
	seed = cv::Point3f(medianOf(xs), medianOf(ys), medianOf(zs));
	return util3d::isFinite(seed);
}

TriangulationResult triangulateTrack(const std::vector<TrackView> & views, bool useDepthSeed)
{
	TriangulationResult result;
	if(views.size() < 2)
	{
		return result;
	}

	// Candidate 1: algebraic triangulation from bearing rays, geometrically refined.
	bool haveCandidate = false;
	cv::Point3f best;
	float bestReproj = 0.0f;

	cv::Mat A((int)views.size()*2, 4, CV_64FC1);
	for(size_t i=0; i<views.size(); ++i)
	{
		const TrackView & view = views[i];
		const Transform & t = view.worldToCamera;
		const double xn = ((double)view.u - (double)view.cx)/(double)view.fx;
		const double yn = ((double)view.v - (double)view.cy)/(double)view.fy;

		const double p1[4] = {(double)t.r11(), (double)t.r12(), (double)t.r13(), (double)t.x()};
		const double p2[4] = {(double)t.r21(), (double)t.r22(), (double)t.r23(), (double)t.y()};
		const double p3[4] = {(double)t.r31(), (double)t.r32(), (double)t.r33(), (double)t.z()};

		for(int c=0; c<4; ++c)
		{
			A.at<double>((int)i*2,   c) = xn*p3[c] - p1[c];
			A.at<double>((int)i*2+1, c) = yn*p3[c] - p2[c];
		}
	}

	cv::Mat x;
	cv::SVD::solveZ(A, x);
	if(x.rows == 4 && std::fabs(x.at<double>(3)) >= 1e-9)
	{
		const double w = x.at<double>(3);
		cv::Point3f dlt(
				(float)(x.at<double>(0)/w),
				(float)(x.at<double>(1)/w),
				(float)(x.at<double>(2)/w));
		if(util3d::isFinite(dlt))
		{
			refinePointGaussNewton(views, dlt);
			float e;
			if(util3d::isFinite(dlt) && scoreTrackPoint(views, dlt, e))
			{
				best = dlt; bestReproj = e; haveCandidate = true;
			}
		}
	}

	// Candidate 2: the sensor's own depth, refined the same way. Kept only if it
	// reprojects better than the algebraic solution, so this can add tracks the DLT
	// loses to poor parallax but can never replace a well-conditioned triangulation
	// with a worse one.
	if(useDepthSeed)
	{
		cv::Point3f seed;
		if(depthSeedTrackPoint(views, seed))
		{
			refinePointGaussNewton(views, seed);
			float e;
			if(util3d::isFinite(seed) && scoreTrackPoint(views, seed, e) &&
			   (!haveCandidate || e < bestReproj))
			{
				best = seed; bestReproj = e; haveCandidate = true;
			}
		}
	}

	if(!haveCandidate)
	{
		return result;
	}
	result.point = best;

	// Cheirality + reprojection error.
	std::vector<float> depthErrors;
	for(size_t i=0; i<views.size(); ++i)
	{
		const TrackView & view = views[i];
		const cv::Point3f pc = util3d::transformPoint(result.point, view.worldToCamera);
		if(pc.z <= 0.0f)
		{
			return result; // behind a camera
		}
		const float du = view.fx*pc.x/pc.z + view.cx - view.u;
		const float dv = view.fy*pc.y/pc.z + view.cy - view.v;
		result.maxReprojError = std::max(result.maxReprojError, std::sqrt(du*du + dv*dv));

		if(view.measuredDepth > 0.0f)
		{
			depthErrors.push_back(std::fabs(pc.z - view.measuredDepth)/view.measuredDepth);
		}
	}

	// Widest parallax between any two views.
	for(size_t i=0; i<views.size(); ++i)
	{
		const cv::Point3f ri = result.point - views[i].center;
		const float ni = std::sqrt(ri.x*ri.x + ri.y*ri.y + ri.z*ri.z);
		if(ni <= 0.0f)
		{
			continue;
		}
		for(size_t j=i+1; j<views.size(); ++j)
		{
			const cv::Point3f rj = result.point - views[j].center;
			const float nj = std::sqrt(rj.x*rj.x + rj.y*rj.y + rj.z*rj.z);
			if(nj <= 0.0f)
			{
				continue;
			}
			float c = (ri.x*rj.x + ri.y*rj.y + ri.z*rj.z)/(ni*nj);
			c = c > 1.0f ? 1.0f : (c < -1.0f ? -1.0f : c);
			result.maxAngleDeg = std::max(result.maxAngleDeg, (float)(std::acos(c)*180.0/M_PI));
		}
	}

	if(!depthErrors.empty())
	{
		result.hasDepth = true;
		result.medianDepthError = medianOf(depthErrors);
	}
	result.valid = true;
	return result;
}

/**
 * One frame pair worth attempting.
 *
 * Pair proposal and edge construction are separate stages: a candidate only says
 * "these two frames may see the same thing", descriptor matching decides whether
 * it yields any correspondence. No observation is ever fabricated from geometry.
 */
struct PairCandidate
{
	PairCandidate(int f, int t, const Transform & lt) : fromId(f), toId(t), linkTransform(lt) {}
	int fromId;
	int toId;
	Transform linkTransform; // null for frustum-proposed pairs, which have no link
};

/**
 * Orders candidates so that neighbouring pairs share frames.
 *
 * Only matters for how often a frame has to be read: a batch is matched with its frames
 * resident, so pairs that mention the same frames should fall in the same batch. The
 * tracks themselves do not depend on this -- union-find's partition is independent of the
 * order the unions arrive in.
 */
struct PairCandidateOrder
{
	bool operator()(const PairCandidate & a, const PairCandidate & b) const
	{
		if(a.fromId != b.fromId) return a.fromId < b.fromId;
		return a.toId < b.toId;
	}
};

/** Normalized (low,high) frame pair, so a pair is only ever attempted once. */
std::pair<int, int> orderedPair(int a, int b)
{
	return a < b? std::make_pair(a, b) : std::make_pair(b, a);
}

/**
 * A node's own feature indices, in a form that can be looked up by keypoint.
 *
 * Needed because RegistrationVis rebuilds the word maps while it matches: the indices it
 * leaves behind address a filtered, transient array, so they mean nothing once the call
 * returns. The unadjusted keypoint resolves back to the index the node stores, and that
 * index -- the feature's position in the database -- is the identity everything
 * downstream uses: the union-find slots, the Match table, the COLMAP writer.
 *
 * A sorted vector rather than a std::map, because one of these is built for every frame
 * of every batch: 12 contiguous bytes per feature beats some 80 scattered over nodes,
 * both to build and to search. Ties on (x,y) resolve to the lowest index, which is what
 * the map did -- filled in index order, first insert winning.
 */
class FeatureKeyIndex
{
public:
	FeatureKeyIndex() {}
	explicit FeatureKeyIndex(const std::vector<cv::KeyPoint> & kpts)
	{
		entries_.reserve(kpts.size());
		for(size_t i=0; i<kpts.size(); ++i)
		{
			entries_.push_back(Entry(kpts[i].pt.x, kpts[i].pt.y, (int)i));
		}
		std::sort(entries_.begin(), entries_.end());
	}
	/** Index of the feature at this keypoint's pixel, or -1 if the frame has none there. */
	int find(const cv::KeyPoint & kpt) const
	{
		const Entry key(kpt.pt.x, kpt.pt.y, -1);
		std::vector<Entry>::const_iterator iter =
				std::lower_bound(entries_.begin(), entries_.end(), key);
		if(iter != entries_.end() && iter->x == key.x && iter->y == key.y)
		{
			return iter->index;
		}
		return -1;
	}
	bool empty() const {return entries_.empty();}

private:
	struct Entry
	{
		Entry(float xIn, float yIn, int i) : x(xIn), y(yIn), index(i) {}
		bool operator<(const Entry & o) const
		{
			if(x != o.x) return x < o.x;
			if(y != o.y) return y < o.y;
			return index < o.index;
		}
		float x, y;
		int index;
	};
	std::vector<Entry> entries_;
};

/**
 * The raw descriptor matches of one frame pair, as feature index couples, plus which of
 * them the pose estimate of the moment kept. This is what the Match table stores.
 */
struct PairMatchRecord
{
	PairMatchRecord() : valid(false) {}
	std::vector<std::pair<int, int> > pairs;
	std::vector<bool> inlier;
	bool valid;
};

/**
 * Match one pair of frames and report the correspondences as the two frames' own feature
 * indices.
 *
 * Pure with respect to the track state: the only mutable thing it touches is @p reg,
 * which is why each thread is given its own. Matching a pair is independent of every
 * other pair; uniting the results is not, so the two are separate stages.
 *
 * @param guess            pose guess handed to @ref RegistrationVis. When the
 *                         registration is configured with a guess window, matching is
 *                         restricted to a reprojection window around the predicted pixel.
 * @param stripDescriptors drop the descriptors so RegistrationVis falls back to word-id
 *                         equality (the mapping-time matches). Same trick as
 *                         Memory::computeTransform(..., useKnownCorrespondencesIfPossible).
 * @param acceptMatches    keep every match instead of only the RANSAC PnP inliers.
 * @param record           optional: every match, with the surviving ones flagged, for the
 *                         Match table.
 * @param out              accepted correspondences as (from index, to index).
 * @return true if at least one correspondence came out.
 */
bool matchPair(
		const Signature & sFromIn,
		const Signature & sToIn,
		int fromId,
		int toId,
		const Transform & guess,
		bool stripDescriptors,
		bool acceptMatches,
		RegistrationVis & reg,
		const FeatureKeyIndex & keyIndexFrom,
		const FeatureKeyIndex & keyIndexTo,
		std::vector<std::pair<int, int> > & out,
		PairMatchRecord * record)
{
	out.clear();
	Signature sFrom = sFromIn;
	Signature sTo = sToIn;

	if((sFrom.sensorData().cameraModels().empty() && sFrom.sensorData().stereoCameraModels().empty()) ||
	   (sTo.sensorData().cameraModels().empty() && sTo.sensorData().stereoCameraModels().empty()))
	{
		UERROR("No camera models found");
		return false;
	}

	if(sFrom.getWordsKpts().empty() || sTo.getWordsKpts().empty() || sFrom.getWords3().empty())
	{
		return false;
	}

	if(stripDescriptors)
	{
		sFrom.setWordsDescriptors(cv::Mat());
		sTo.setWordsDescriptors(cv::Mat());
	}
	else if(sFrom.getWordsDescriptors().empty() || sTo.getWordsDescriptors().empty())
	{
		// Descriptor matching without descriptors would silently degrade to the
		// word-equality pass we already ran; skip instead of duplicating it.
		return false;
	}

	RegistrationInfo info;
	Transform t = reg.computeTransformationMod(sFrom, sTo, guess, &info);
	UDEBUG("%d->%d, inliers=%d matches=%d", fromId, toId, (int)info.inliersIDs.size(), (int)info.matchesIDs.size());

	if(record)
	{
		// Everything matched, with the surviving ones flagged. Raw rather than filtered:
		// a descriptor match is pose-independent, while the inlier verdict belongs to the
		// pose estimate that produced it, so only the former is safe to keep.
		const std::set<int> inlierSet(info.inliersIDs.begin(), info.inliersIDs.end());
		record->pairs.clear();
		record->inlier.clear();
		record->pairs.reserve(info.matchesIDs.size());
		record->inlier.reserve(info.matchesIDs.size());
		for(size_t i=0; i<info.matchesIDs.size(); ++i)
		{
			std::multimap<int, int>::const_iterator wF = sFrom.getWords().find(info.matchesIDs[i]);
			std::multimap<int, int>::const_iterator wT = sTo.getWords().find(info.matchesIDs[i]);
			if(wF == sFrom.getWords().end() || wT == sTo.getWords().end() ||
			   wF->second >= (int)sFrom.getWordsKpts().size() ||
			   wT->second >= (int)sTo.getWordsKpts().size())
			{
				continue;
			}
			const int iF = keyIndexFrom.find(sFrom.getWordsKpts()[wF->second]);
			const int iT = keyIndexTo.find(sTo.getWordsKpts()[wT->second]);
			if(iF >= 0 && iT >= 0)
			{
				record->pairs.push_back(std::make_pair(iF, iT));
				record->inlier.push_back(inlierSet.find(info.matchesIDs[i]) != inlierSet.end());
			}
		}
		record->valid = true;
	}

	if(!acceptMatches && t.isNull())
	{
		UDEBUG("Not enough inliers (%d) between %d and %d", (int)info.inliersIDs.size(), fromId, toId);
		return false;
	}
	const std::vector<int> & correspondences = acceptMatches? info.matchesIDs : info.inliersIDs;
	if(correspondences.empty())
	{
		return false;
	}

	out.reserve(correspondences.size());
	for(unsigned int i=0; i<correspondences.size(); ++i)
	{
		std::multimap<int, int>::const_iterator wFrom = sFrom.getWords().find(correspondences[i]);
		std::multimap<int, int>::const_iterator wTo = sTo.getWords().find(correspondences[i]);
		if(wFrom == sFrom.getWords().end() || wTo == sTo.getWords().end() ||
		   wFrom->second >= (int)sFrom.getWordsKpts().size() ||
		   wTo->second >= (int)sTo.getWordsKpts().size())
		{
			continue;
		}
		const int indexFrom = keyIndexFrom.find(sFrom.getWordsKpts()[wFrom->second]);
		const int indexTo = keyIndexTo.find(sTo.getWordsKpts()[wTo->second]);
		if(indexFrom >= 0 && indexTo >= 0)
		{
			out.push_back(std::make_pair(indexFrom, indexTo));
		}
	}
	return !out.empty();
}

/**
 * Re-key two frames so that the cached matches, and only those, are what word-equality
 * matching finds. Every other feature is given an id from a range the other frame cannot
 * hold, so it can pair with nothing.
 *
 * This is how a cache hit skips descriptor matching without skipping the geometry: the
 * pair still goes through @ref RegistrationVis, so PnP, the reprojection gate and the
 * inlier decision are all recomputed against the poses of the moment. Only the matching
 * -- the expensive, pose-independent half -- is reused. The same re-keying trick the
 * word-equality pass already relies on.
 */
void applyCachedMatches(
		Signature & sFrom,
		Signature & sTo,
		const std::vector<std::pair<int, int> > & cached)
{
	// Disjoint ranges, both far above any plausible cached-match count, so an unmatched
	// feature on one side can never collide with one on the other.
	const int kUnmatchedFrom = 1<<24;
	const int kUnmatchedTo   = 1<<25;

	std::vector<int> idFrom(sFrom.getWordsKpts().size(), -1);
	std::vector<int> idTo(sTo.getWordsKpts().size(), -1);
	for(size_t k=0; k<cached.size(); ++k)
	{
		const int i = cached[k].first;
		const int j = cached[k].second;
		if(i >= 0 && i < (int)idFrom.size() && j >= 0 && j < (int)idTo.size() &&
		   idFrom[i] < 0 && idTo[j] < 0)
		{
			// One id per couple; a feature already spoken for is left alone rather than
			// re-keyed, so a stale row cannot fabricate a many-to-one match.
			idFrom[i] = (int)k + 1;
			idTo[j] = (int)k + 1;
		}
	}

	std::multimap<int, int> wordsFrom;
	for(size_t i=0; i<idFrom.size(); ++i)
	{
		wordsFrom.insert(std::make_pair(idFrom[i] < 0? kUnmatchedFrom + (int)i : idFrom[i], (int)i));
	}
	std::multimap<int, int> wordsTo;
	for(size_t j=0; j<idTo.size(); ++j)
	{
		wordsTo.insert(std::make_pair(idTo[j] < 0? kUnmatchedTo + (int)j : idTo[j], (int)j));
	}

	sFrom.setWords(wordsFrom, sFrom.getWordsKpts(), sFrom.getWords3(), sFrom.getWordsDescriptors());
	sTo.setWords(wordsTo, sTo.getWordsKpts(), sTo.getWords3(), sTo.getWordsDescriptors());
}

/**
 * Unite one pair's correspondences into the track state. Serial by nature -- the
 * union-find and the observation table are shared.
 *
 * Order-independent all the same: union-find's partition is the same whatever order the
 * unions arrive in, so which batch matched which pair first cannot change the tracks.
 * Only the slot numbering follows encounter order, and nothing reads that.
 */
bool insertPairCorrespondences(
		BAFrameSource & frames,
		int fromId,
		int toId,
		const std::vector<std::pair<int, int> > & corrs,
		std::vector<TrackObservation> & observations,
		ObservationUnionFind & uf)
{
	bool added = false;
	for(size_t i=0; i<corrs.size(); ++i)
	{
		const int nodeFrom = getOrCreateObservationNode(frames, fromId, corrs[i].first, observations, uf);
		const int nodeTo = getOrCreateObservationNode(frames, toId, corrs[i].second, observations, uf);
		if(nodeFrom < 0 || nodeTo < 0)
		{
			continue;
		}
		uf.unite(nodeFrom, nodeTo);
		added = true;
	}
	return added;
}

/** A frame reduced to what the frustum test needs. */
struct FrustumFrame
{
	int id;
	CameraModel model;          // first camera of the rig
	Transform worldToCamera;    // optical frame
	cv::Point3f center;         // camera centre, world
	cv::Point3f axis;           // optical axis (camera +z), world, unit
	std::vector<cv::Point3f> points; // subsampled 3D words, world
};

/**
 * Fraction of @p a's 3D words that fall inside @p b's image.
 *
 * This is a real overlap measure rather than a frustum-intersection
 * approximation: it uses the scene points the frame actually observed, so it
 * accounts for the depth range as well as the field of view.
 */
float frustumOverlap(const FrustumFrame & a, const FrustumFrame & b)
{
	if(a.points.empty())
	{
		return 0.0f;
	}
	int inside = 0;
	for(size_t i=0; i<a.points.size(); ++i)
	{
		const cv::Point3f p = util3d::transformPoint(a.points[i], b.worldToCamera);
		if(p.z <= 0.0f)
		{
			continue;
		}
		float u, v;
		b.model.reproject(p.x, p.y, p.z, u, v);
		if(u >= 0.0f && v >= 0.0f && u < (float)b.model.imageWidth() && v < (float)b.model.imageHeight())
		{
			++inside;
		}
	}
	return (float)inside/(float)a.points.size();
}

// Build a cheap, spatially spread geometry sample without fetching descriptors
// or decoding depth images. First-camera geometry is only a ranking heuristic;
// it never vetoes a graph-link pair, and weak coverage gets a fallback pass.
FrustumFrame budgetFrame(int id, const Transform & pose, BAFrameSource & frames)
{
    FrustumFrame f; f.id = id;
    if(!frames.cameraModels(id).empty()) f.model = frames.cameraModels(id)[0];
    else if(!frames.stereoModels(id).empty()) f.model = frames.stereoModels(id)[0].left();
    if(!f.model.isValidForProjection()) return f;
    const Transform camera = pose*f.model.localTransform();
    f.worldToCamera = camera.inverse();
    f.center = cv::Point3f(camera.x(), camera.y(), camera.z());
    std::array<int, 48> cells; cells.fill(-1); // 8 x 6 image grid
    const auto & kpts = frames.keypoints(id);
    const auto & points = frames.words3(id);
    if(f.model.imageWidth() <= 0 || f.model.imageHeight() <= 0) return f;
    for(size_t i=0; i<std::min(kpts.size(),points.size()); ++i)
    {
        const cv::Point2f uv = kpts[i].pt;
        if(!util3d::isFinite(points[i]) || points[i] == cv::Point3f() ||
           uv.x < 0 || uv.y < 0 || uv.x >= f.model.imageWidth() || uv.y >= f.model.imageHeight()) continue;
        const int cell = int(uv.y*6/f.model.imageHeight())*8 + int(uv.x*8/f.model.imageWidth());
        if(cells[cell] < 0) cells[cell] = (int)i;
    }
    for(int index : cells)
        if(index >= 0) f.points.push_back(util3d::transformPoint(points[index], pose));
    return f;
}

double budgetQuality(const FrustumFrame & a, const FrustumFrame & b)
{
    if(a.points.empty() || b.points.empty()) return 0.5; // unknown, not a hard exclusion
    const double overlap = std::min(frustumOverlap(a,b), frustumOverlap(b,a));
    std::vector<double> angles;
    for(const auto & p : a.points)
    {
        const cv::Point3f u = p-a.center, v = p-b.center;
        const double denom = cv::norm(u)*cv::norm(v);
        if(denom > 1e-9) angles.push_back(std::acos(std::max(-1.0,std::min(1.0,u.dot(v)/denom))));
    }
    if(angles.empty()) return overlap;
    const size_t mid = angles.size()/2;
    std::nth_element(angles.begin(),angles.begin()+mid,angles.end());
    return overlap*std::min(1.0,angles[mid]/(1.5*M_PI/180.0));
}

// Count conflict-free tracks spanning >=3 frames before triangulation. This is
// a cheap fallback trigger, not a claim that the tracks will triangulate well.
std::set<int> weakBudgetFrames(const std::vector<TrackObservation> & observations,
    ObservationUnionFind & uf, BAFrameSource & frames)
{
    std::vector<std::pair<int,int> > groups;
    groups.reserve(observations.size());
    for(size_t i=0;i<observations.size();++i) groups.push_back({uf.find((int)i),(int)i});
    std::sort(groups.begin(),groups.end());
    std::map<int,int> support;
    std::map<int,std::set<int> > cells;
    for(size_t begin=0; begin<groups.size();)
    {
        size_t end=begin+1;
        while(end<groups.size() && groups[end].first==groups[begin].first) ++end;
        std::set<int> ids;
        bool conflict=false;
        for(size_t k=begin;k<end;++k)
            if(!ids.insert(observations[groups[k].second].frameId).second) conflict=true;
        if(!conflict && ids.size()>=3)
            for(size_t k=begin;k<end;++k)
            {
                const auto & obs=observations[groups[k].second];
                ++support[obs.frameId];
                const auto & models=frames.cameraModels(obs.frameId);
                const auto & stereo=frames.stereoModels(obs.frameId);
                int w=0,h=0;
                if(!models.empty()) { w=models[0].imageWidth()*(int)models.size();h=models[0].imageHeight(); }
                else if(!stereo.empty()) { w=stereo[0].left().imageWidth();h=stereo[0].left().imageHeight(); }
                const auto & kp=frames.keypoints(obs.frameId)[obs.featureIndex].pt;
                if(w>0 && h>0 && kp.x>=0 && kp.y>=0 && kp.x<w && kp.y<h)
                    cells[obs.frameId].insert(int(kp.y*4/h)*4+int(kp.x*4/w));
            }
        begin=end;
    }
    std::set<int> weak;
    for(int id : frames.ids()) if(support[id]<20 || cells[id].size()<4) weak.insert(id);
    return weak;
}

/** Straight-line distance between two poses. */
double poseDistance(const Transform & a, const Transform & b)
{
	const double dx = a.x()-b.x(), dy = a.y()-b.y(), dz = a.z()-b.z();
	return std::sqrt(dx*dx + dy*dy + dz*dz);
}

}  // namespace

/**
 * Propose extra frame pairs from frustum overlap on the optimized poses.
 *
 * Candidates are pre-filtered on camera-centre distance and optical-axis angle
 * (both cheap), then scored by @ref frustumOverlap and capped per frame. Pairs
 * already covered by @p existing are never re-proposed.
 */
std::vector<std::pair<int, int> > Optimizer::proposeWindowPairs(
		const std::map<int, Transform> & poses,
		const std::multimap<int, Link> & links,
		const std::set<std::pair<int, int> > & existing) const
{
	std::vector<std::pair<int, int> > proposals;
	if(baWindowGraphDepth_ <= 0)
	{
		return proposals;
	}

	// Odometry adjacency. Only neighbor links define a body: a body is a stretch of
	// trajectory whose internal geometry odometry is trusted to hold together.
	std::map<int, std::set<int> > odomAdj;
	std::vector<std::pair<int, int> > loops;
	for(std::multimap<int, Link>::const_iterator iter=links.lower_bound(1); iter!=links.end(); ++iter)
	{
		const Link & link = iter->second;
		if(link.from() == link.to() ||
		   !uContains(poses, link.from()) || !uContains(poses, link.to()))
		{
			continue;
		}
		if(link.type() == Link::kNeighbor || link.type() == Link::kNeighborMerged)
		{
			odomAdj[link.from()].insert(link.to());
			odomAdj[link.to()].insert(link.from());
		}
		else if(link.type() == Link::kGlobalClosure || link.type() == Link::kLocalSpaceClosure ||
				link.type() == Link::kLocalTimeClosure || link.type() == Link::kUserClosure)
		{
			loops.push_back(orderedPair(link.from(), link.to()));
		}
	}
	if(loops.empty())
	{
		return proposals;
	}
	std::sort(loops.begin(), loops.end());
	loops.erase(std::unique(loops.begin(), loops.end()), loops.end());

	std::set<std::pair<int, int> > emitted = existing;
	for(size_t i=0; i<loops.size(); ++i)
	{
		// Two bodies, grown by breadth-first search over odometry links only.
		std::set<int> body[2];
		const int seed[2] = {loops[i].first, loops[i].second};

		// How far the window should reach, in metres. Graph hops are the wrong unit for
		// this: what produces parallax is metric baseline, and two hops is several metres
		// of car travel on KITTI but a few centimetres of hand-held drift on EuRoC, so one
		// hop count cannot serve both.
		//
		// The loop's own baseline sets the scale. A revisit that lands almost on top of
		// itself needs only a small neighbourhood to gain parallax; a proximity closure
		// whose two frames stand two metres apart needs a correspondingly wider one to
		// bridge them. This is purely a property of the loop, so mapping more of the
		// building cannot change the window a given loop receives.
		double targetBaseline = 0.0;
		if(baWindowBaseline_ > 0.0f)
		{
			targetBaseline = baWindowBaseline_;
		}
		if(baWindowBaselineRatio_ > 0.0f)
		{
			const double loopBaseline = poseDistance(poses.at(seed[0]), poses.at(seed[1]));
			targetBaseline = std::max(targetBaseline, (double)baWindowBaselineRatio_*loopBaseline);
		}

		for(int b=0; b<2; ++b)
		{
			const Transform & seedPose = poses.at(seed[b]);

			std::map<int, int> depth;
			std::list<int> queue;
			depth.insert(std::make_pair(seed[b], 0));
			queue.push_back(seed[b]);
			while(!queue.empty())
			{
				const int id = queue.front();
				queue.pop_front();
				const int d = depth.at(id);
				body[b].insert(id);
				// The hop count is a hard cap so a metric target can never run away.
				if(d >= baWindowGraphDepth_)
				{
					continue;
				}
				if(targetBaseline > 0.0)
				{
					const Transform & here = poses.at(id);
					const double dx = here.x()-seedPose.x();
					const double dy = here.y()-seedPose.y();
					const double dz = here.z()-seedPose.z();
					if(std::sqrt(dx*dx + dy*dy + dz*dz) >= targetBaseline)
					{
						continue;   // far enough from the seed; stop growing this branch
					}
				}
				std::map<int, std::set<int> >::const_iterator adj = odomAdj.find(id);
				if(adj == odomAdj.end())
				{
					continue;
				}
				for(std::set<int>::const_iterator n=adj->second.begin(); n!=adj->second.end(); ++n)
				{
					if(depth.find(*n) == depth.end())
					{
						depth.insert(std::make_pair(*n, d+1));
						queue.push_back(*n);
					}
				}
			}
		}

		// A frame reachable from both ends within the window is not on one side of
		// the loop or the other, so a pair through it says nothing about the closure.
		std::vector<int> shared;
		std::set_intersection(body[0].begin(), body[0].end(), body[1].begin(), body[1].end(),
							  std::back_inserter(shared));
		for(size_t k=0; k<shared.size(); ++k)
		{
			body[0].erase(shared[k]);
			body[1].erase(shared[k]);
		}

		for(std::set<int>::const_iterator a=body[0].begin(); a!=body[0].end(); ++a)
		{
			for(std::set<int>::const_iterator b=body[1].begin(); b!=body[1].end(); ++b)
			{
				const std::pair<int, int> pair = orderedPair(*a, *b);
				// emitted carries the pairs already covered by the graph and by
				// earlier loops, so overlapping windows in a loop-rich region are
				// matched once between them rather than once per loop.
				if(emitted.insert(pair).second)
				{
					proposals.push_back(pair);
				}
			}
		}
	}

	UINFO("BA window: %d loop(s), graph depth %d -> %d cross-body pairs proposed",
			(int)loops.size(), baWindowGraphDepth_, (int)proposals.size());
	return proposals;
}

std::vector<std::pair<int, int> > Optimizer::proposeFrustumPairs(
		const std::map<int, Transform> & poses,
		BAFrameSource & frameSource,
		const std::set<std::pair<int, int> > & existing) const
{
	std::vector<std::pair<int, int> > proposed;

	// Reduce every frame to its first camera plus a subsample of its 3D words.
	const size_t maxPointsPerFrame = 50;
	std::vector<FrustumFrame> frames;
	frames.reserve(poses.size());
	for(std::map<int, Transform>::const_iterator iter=poses.begin(); iter!=poses.end(); ++iter)
	{
		if(iter->first <= 0 || iter->second.isNull())
		{
			continue; // landmarks and unposed nodes
		}
		if(!frameSource.has(iter->first) || frameSource.weight(iter->first) < 0)
		{
			continue;
		}
		FrustumFrame f;
		if(!frameSource.cameraModels(iter->first).empty())
		{
			f.model = frameSource.cameraModels(iter->first)[0];
		}
		else if(!frameSource.stereoModels(iter->first).empty())
		{
			f.model = frameSource.stereoModels(iter->first)[0].left();
		}
		else
		{
			continue;
		}
		if(!f.model.isValidForProjection() || f.model.localTransform().isNull() ||
		   f.model.imageWidth() <= 0 || f.model.imageHeight() <= 0)
		{
			continue;
		}
		const std::vector<cv::Point3f> & words3 = frameSource.words3(iter->first);
		if(words3.empty())
		{
			continue;
		}
		const Transform cameraToWorld = iter->second * f.model.localTransform();
		f.id = iter->first;
		f.worldToCamera = cameraToWorld.inverse();
		f.center = cv::Point3f(cameraToWorld.x(), cameraToWorld.y(), cameraToWorld.z());
		f.axis = cv::Point3f(cameraToWorld.r13(), cameraToWorld.r23(), cameraToWorld.r33());
		const size_t step = words3.size()>maxPointsPerFrame? words3.size()/maxPointsPerFrame : 1;
		for(size_t i=0; i<words3.size(); i+=step)
		{
			// Words are in the base frame of their own signature.
			const cv::Point3f p = util3d::transformPoint(words3[i], iter->second);
			if(util3d::isFinite(p) && !(words3[i].x == 0.0f && words3[i].y == 0.0f && words3[i].z == 0.0f))
			{
				f.points.push_back(p);
			}
		}
		if(!f.points.empty())
		{
			frames.push_back(f);
		}
	}

	const float maxRangeSqr = baProposeMaxRange_>0.0f? baProposeMaxRange_*baProposeMaxRange_ : 0.0f;
	const float minAxisCos = baProposeMaxAngle_>0.0f?
			(float)cos(std::min(baProposeMaxAngle_, 180.0f)*M_PI/180.0) : -2.0f;

	std::set<std::pair<int, int> > emitted;
	int prefiltered = 0;
	for(size_t i=0; i<frames.size(); ++i)
	{
		const FrustumFrame & a = frames[i];
		std::vector<std::pair<float, int> > scored; // <overlap, frame index>
		for(size_t j=0; j<frames.size(); ++j)
		{
			if(i == j)
			{
				continue;
			}
			const FrustumFrame & b = frames[j];
			if(existing.find(orderedPair(a.id, b.id)) != existing.end())
			{
				continue;
			}
			const cv::Point3f d = b.center - a.center;
			if(maxRangeSqr > 0.0f && d.x*d.x + d.y*d.y + d.z*d.z > maxRangeSqr)
			{
				continue;
			}
			if(minAxisCos > -2.0f &&
			   a.axis.x*b.axis.x + a.axis.y*b.axis.y + a.axis.z*b.axis.z < minAxisCos)
			{
				continue;
			}
			++prefiltered;
			const float overlap = frustumOverlap(a, b);
			if(overlap >= baProposeMinOverlap_)
			{
				scored.push_back(std::make_pair(overlap, (int)j));
			}
		}
		std::sort(scored.begin(), scored.end(), std::greater<std::pair<float, int> >());
		const size_t keep = baProposeMaxPairs_>0? std::min(scored.size(), (size_t)baProposeMaxPairs_) : scored.size();
		for(size_t k=0; k<keep; ++k)
		{
			const std::pair<int, int> pair = orderedPair(a.id, frames[scored[k].second].id);
			if(emitted.insert(pair).second)
			{
				proposed.push_back(pair);
			}
		}
	}

	UINFO("BA pair proposal: %d frames, %d pairs passed the distance/angle prefilter, %d pairs proposed.",
			(int)frames.size(), prefiltered, (int)proposed.size());
	return proposed;
}

void Optimizer::computeBACorrespondencesFrom(
		const std::map<int, Transform> & poses,
		const std::multimap<int, Link> & links,
		BAFrameSource & frames,
		std::map<int, cv::Point3f> & points3DMap,
		std::map<int, std::map<int, FeatureBA> > & wordReferences,
		bool rematchFeatures,
		bool useLinkTransformAsGuess,
		const ParametersMap & registrationParameters)
{
	UTimer totalTimer;
	reportProgress(uFormat("Building BA correspondences (strategy %d)...", baStrategy_));
	if(baStrategy_ == 1)
	{
		computeBACorrespondencesTracks(
				poses, links, frames, points3DMap, wordReferences,
				rematchFeatures, useLinkTransformAsGuess, registrationParameters);
	}
	else
	{
		computeBACorrespondencesPairwise(
				poses, links, frames, points3DMap, wordReferences,
				rematchFeatures, useLinkTransformAsGuess, registrationParameters);
	}
	if(isCanceled())
	{
		return;
	}

	// Pure reprojection BA: drop every depth measurement, which routes each
	// observation to a mono reprojection edge in the back-ends. Scale is still
	// pinned by the metric pose-graph links, which optimizeBA() adds as real
	// pose-pose edges -- this is not the sim(3) gauge freedom of monocular SfM.
	// Applied after either strategy so the two stay directly comparable.
	if(baDepth_ == 0)
	{
		int zeroed = 0;
		for(std::map<int, std::map<int, FeatureBA> >::iterator iter=wordReferences.begin(); iter!=wordReferences.end(); ++iter)
		{
			for(std::map<int, FeatureBA>::iterator jter=iter->second.begin(); jter!=iter->second.end(); ++jter)
			{
				if(jter->second.depth != 0.0f)
				{
					jter->second.depth = 0.0f;
					++zeroed;
				}
			}
		}
		UINFO("BA depth disabled: %d observations converted to mono reprojection.", zeroed);
	}
	UINFO("BA correspondences: strategy=%d, %d tracks, done in %.2fs.",
			baStrategy_, (int)wordReferences.size(), totalTimer.elapsed());
	reportProgress(uFormat("BA correspondences built in %.1fs", totalTimer.elapsed()));

	reportBALeverStats(links, wordReferences);
}

namespace {

double medianOfDouble(std::vector<double> & v)
{
	if(v.empty()) return 0.0;
	std::sort(v.begin(), v.end());
	return v[v.size()/2];
}

/** Per-link-category accumulator for the leverage report. */
struct LeverBucket
{
	LeverBucket() : links(0), linksWithTracks(0) {}
	int links;
	int linksWithTracks;
	std::vector<double> sharedPerLink;   // tracks visible in both frames
	std::vector<double> meanLenPerLink;  // mean length of those tracks
};

void printLever(const char * label, LeverBucket & b)
{
	if(b.links == 0)
	{
		return;
	}
	double sumShared = 0.0, sumLen = 0.0;
	for(size_t i=0; i<b.sharedPerLink.size(); ++i) sumShared += b.sharedPerLink[i];
	for(size_t i=0; i<b.meanLenPerLink.size(); ++i) sumLen += b.meanLenPerLink[i];
	const double meanShared = b.sharedPerLink.empty()?0.0:sumShared/(double)b.sharedPerLink.size();
	const double meanLen = b.meanLenPerLink.empty()?0.0:sumLen/(double)b.meanLenPerLink.size();
	UINFO("BA lever [%s]: links=%d (%d with shared tracks) | shared tracks/link mean=%.1f median=%.1f | "
		  "track length mean=%.2f median=%.2f | total shared-track incidences=%.0f",
			label, b.links, b.linksWithTracks,
			meanShared, medianOfDouble(b.sharedPerLink),
			meanLen, medianOfDouble(b.meanLenPerLink),
			sumShared);
}

}  // namespace

void Optimizer::reportBALeverStats(
		const std::multimap<int, Link> & links,
		const std::map<int, std::map<int, FeatureBA> > & wordReferences) const
{
	if(wordReferences.empty())
	{
		return;
	}

	// frame -> tracks it observes
	std::map<int, std::set<int> > frameTracks;
	for(std::map<int, std::map<int, FeatureBA> >::const_iterator iter=wordReferences.begin(); iter!=wordReferences.end(); ++iter)
	{
		for(std::map<int, FeatureBA>::const_iterator jter=iter->second.begin(); jter!=iter->second.end(); ++jter)
		{
			frameTracks[jter->first].insert(iter->first);
		}
	}

	LeverBucket odom, loop;
	std::set<int> tracksOnLoops;   // tracks witnessed by at least one loop-closure link

	for(std::multimap<int, Link>::const_iterator iter=links.lower_bound(1); iter!=links.end(); ++iter)
	{
		const Link & link = iter->second;
		if(link.from() == link.to())
		{
			continue;
		}
		const bool isOdom = link.type() == Link::kNeighbor || link.type() == Link::kNeighborMerged;
		const bool isLoop = link.type() == Link::kGlobalClosure || link.type() == Link::kLocalSpaceClosure ||
							link.type() == Link::kLocalTimeClosure || link.type() == Link::kUserClosure;
		if(!isOdom && !isLoop)
		{
			continue;
		}
		LeverBucket & bucket = isOdom? odom : loop;
		++bucket.links;

		std::map<int, std::set<int> >::const_iterator f = frameTracks.find(link.from());
		std::map<int, std::set<int> >::const_iterator t = frameTracks.find(link.to());
		if(f == frameTracks.end() || t == frameTracks.end())
		{
			bucket.sharedPerLink.push_back(0.0);
			continue;
		}
		std::vector<int> shared;
		std::set_intersection(f->second.begin(), f->second.end(),
							  t->second.begin(), t->second.end(),
							  std::back_inserter(shared));
		bucket.sharedPerLink.push_back((double)shared.size());
		if(shared.empty())
		{
			continue;
		}
		++bucket.linksWithTracks;

		double lenSum = 0.0;
		for(size_t i=0; i<shared.size(); ++i)
		{
			lenSum += (double)wordReferences.at(shared[i]).size();
			if(isLoop)
			{
				tracksOnLoops.insert(shared[i]);
			}
		}
		bucket.meanLenPerLink.push_back(lenSum/(double)shared.size());
	}

	// Overall track-length distribution, for reference.
	std::vector<double> allLen;
	allLen.reserve(wordReferences.size());
	for(std::map<int, std::map<int, FeatureBA> >::const_iterator iter=wordReferences.begin(); iter!=wordReferences.end(); ++iter)
	{
		allLen.push_back((double)iter->second.size());
	}
	double lenSum = 0.0;
	for(size_t i=0; i<allLen.size(); ++i) lenSum += allLen[i];

	UINFO("BA lever: %d tracks total, length mean=%.2f median=%.2f",
			(int)wordReferences.size(), lenSum/(double)allLen.size(), medianOfDouble(allLen));
	printLever("odom", odom);
	printLever("loop", loop);
	UINFO("BA lever: %d/%d tracks (%.1f%%) are witnessed by at least one loop-closure link",
			(int)tracksOnLoops.size(), (int)wordReferences.size(),
			100.0*(double)tracksOnLoops.size()/(double)wordReferences.size());
}

void Optimizer::computeBACorrespondencesPairwise(
		const std::map<int, Transform> & poses,
		const std::multimap<int, Link> & links,
		BAFrameSource & frames,
		std::map<int, cv::Point3f> & points3DMap,
		std::map<int, std::map<int, FeatureBA> > & wordReferences,
		bool rematchFeatures,
		bool useLinkTransformAsGuess,
		ParametersMap registrationParameters)
{
	UDEBUG("rematchFeatures=%d", rematchFeatures?1:0);

	int wordCount = 0;
	int edgeWithWordsAdded = 0;

	// Some defaults if not provided
	registrationParameters.insert(ParametersPair(Parameters::kVisEstimationType(), "1"));
	registrationParameters.insert(ParametersPair(Parameters::kVisPnPReprojError(), "5"));
	registrationParameters.insert(ParametersPair(Parameters::kVisMinInliers(), "6"));
	registrationParameters.insert(ParametersPair(Parameters::kVisCorNNDR(), "0.6"));
	RegistrationVis reg(registrationParameters);

	std::map<int, std::map<cv::KeyPoint, int, KeyPointCompare> > frameToWordMap; // <FrameId, <Keypoint, wordId> >
	UTimer pairTimer;
	double lastPairReport = -1.0;
	int processed = 0;
	const int totalLinks = (int)std::distance(links.lower_bound(1), links.end());
	if(!reportProgress(uFormat("Correspondences 0/%d", totalLinks)))
	{
		canceled_ = true;
		return;
	}
	for(std::multimap<int, Link>::const_iterator iter=links.lower_bound(1); iter!=links.end(); ++iter)
	{
		++processed;
		if(processed == totalLinks || pairTimer.elapsed() - lastPairReport > 0.05)
		{
			lastPairReport = pairTimer.elapsed();
			if(!reportProgress(uFormat("Correspondences %d/%d", processed, totalLinks)))
			{
				canceled_ = true;
				points3DMap.clear();
				wordReferences.clear();
				return;
			}
		}
		Link link = iter->second;
		if(link.from() <= 0 || link.to() <= 0 ||
		   link.type() == Link::kPosePrior || link.type() == Link::kLandmark ||
		   link.type() == Link::kGravity || link.type() == Link::kVirtualClosure)
		{
			continue;
		}
		if(link.to() < link.from())
		{
			link = link.inverse();
		}
		if(link.to() != link.from() &&
		   frames.has(link.from()) &&
		   frames.has(link.to()) &&
		   uContains(poses, link.from()))
		{
			// Frames one pair at a time, so this strategy is no more resident than the
			// track builder even though it has no batching of its own to do.
			const Signature * sFromPtr = frames.fetch(link.from());
			if(sFromPtr == 0)
			{
				continue;
			}
			Signature sFrom = *sFromPtr;
			if(sFrom.getWeight() >= 0) // ignore intermediate links
			{
				const Signature * sToPtr = frames.fetch(link.to());
				if(sToPtr == 0)
				{
					continue;
				}
				Signature sTo = *sToPtr;

				if((sFrom.sensorData().cameraModels().empty() && sFrom.sensorData().stereoCameraModels().empty()) ||
				   (sTo.sensorData().cameraModels().empty() && sTo.sensorData().stereoCameraModels().empty()))
				{
					UERROR("No camera models found");
					continue;
				}

				if(sTo.getWeight() < 0)
				{
					for(std::multimap<int, Link>::const_iterator jter=links.find(sTo.id());
						sTo.getWeight() < 0 && jter!=links.end() &&  frames.has(jter->second.to());
						++jter)
					{
						const Signature * hop = frames.fetch(jter->second.to());
						if(hop == 0)
						{
							break;
						}
						sTo = *hop;
					}
				}

				// Kept before the descriptors are stripped below, because the frame they
				// came from may well have been evicted by the time they are put back. A
				// cv::Mat copy is a reference, so this costs nothing and keeps the data
				// alive exactly as long as it is needed.
				const cv::Mat originalDescriptorsFrom = sFrom.getWordsDescriptors();
				const cv::Mat originalDescriptorsTo = sTo.getWordsDescriptors();

				if(sFrom.getWordsKpts().size() &&
					sTo.getWordsKpts().size() &&
					sFrom.getWords3().size())
				{
					if(!rematchFeatures)
					{
						sFrom.setWordsDescriptors(cv::Mat());
						sTo.setWordsDescriptors(cv::Mat());
					}
					else if(sFrom.getWordsDescriptors().empty() && sTo.getWordsDescriptors().empty())
					{
						UWARN("Rematching features is enabled but signatures (%d and %d) don't have word descriptors!? "
							  "Features won't be rematched. If it is an old database, do rtabmap-reprocess "
							  "so that signatures contain word desriptors.",
							  sFrom.id(), sTo.id());
					}

					RegistrationInfo info;
					Transform t = reg.computeTransformationMod(sFrom, sTo, useLinkTransformAsGuess?iter->second.transform():Transform(), &info);
					UDEBUG("%d->%d, inliers=%d",sFrom.id(), sTo.id(), (int)info.inliersIDs.size());

					if(!t.isNull())
					{
						if(!rematchFeatures)
						{
							// set descriptors for the output
							if(sFrom.getWords().size() &&
							   sFrom.getWordsDescriptors().empty() &&
							   (int)sFrom.getWords().size() == originalDescriptorsFrom.rows)
							{
								sFrom.setWordsDescriptors(originalDescriptorsFrom);
							}
							if(sTo.getWords().size() &&
							   sTo.getWordsDescriptors().empty() &&
							   (int)sTo.getWords().size() == originalDescriptorsTo.rows)
							{
								sTo.setWordsDescriptors(originalDescriptorsTo);
							}
						}

						Transform pose = poses.at(sFrom.id());
						UASSERT(!pose.isNull());
						for(unsigned int i=0; i<info.inliersIDs.size(); ++i)
						{
							int indexFrom = sFrom.getWords().lower_bound(info.inliersIDs[i])->second;
							cv::Point3f p = sFrom.getWords3()[indexFrom];
							if(p.x > 0.0f) // make sure the point is valid
							{
								cv::KeyPoint ptFrom = sFrom.getWordsKpts()[indexFrom];
								int indexTo = sTo.getWords().lower_bound(info.inliersIDs[i])->second;
								cv::KeyPoint ptTo = sTo.getWordsKpts()[indexTo];

								int wordId = -1;

								// find if the word is already added
								std::map<int, std::map<cv::KeyPoint, int, KeyPointCompare> >::iterator fromIter = frameToWordMap.find(sFrom.id());
								std::map<int, std::map<cv::KeyPoint, int, KeyPointCompare> >::iterator toIter = frameToWordMap.find(sTo.id());
								bool fromAlreadyAdded = false;
								bool toAlreadyAdded = false;
								if( fromIter != frameToWordMap.end() &&
									fromIter->second.find(ptFrom) != fromIter->second.end())
								{
									wordId = fromIter->second.at(ptFrom);
									fromAlreadyAdded = true;
								}
								if( toIter != frameToWordMap.end() &&
									toIter->second.find(ptTo) != toIter->second.end())
								{
									wordId = toIter->second.at(ptTo);
									toAlreadyAdded = true;
								}

								if(wordId == -1)
								{
									wordId = ++wordCount;
									wordReferences.insert(std::make_pair(wordId, std::map<int, FeatureBA>()));

									points3DMap.insert(std::make_pair(wordId, util3d::transformPoint(p, pose)));
								}
								else
								{
									UASSERT(wordReferences.find(wordId) != wordReferences.end());
									UASSERT(points3DMap.find(wordId) != points3DMap.end());
								}

								if(!fromAlreadyAdded)
								{
									cv::Mat descriptorFrom;
									if(!sFrom.getWordsDescriptors().empty())
									{
										UASSERT(indexFrom < sFrom.getWordsDescriptors().rows);
										descriptorFrom = sFrom.getWordsDescriptors().row(indexFrom);
									}
									int cameraIndex = 0;
									if(sFrom.sensorData().cameraModels().size()>1 || sFrom.sensorData().stereoCameraModels().size()>1)
									{
										float subImageWidth = sFrom.sensorData().cameraModels().size()>1?sFrom.sensorData().cameraModels()[0].imageWidth():sFrom.sensorData().stereoCameraModels()[0].left().imageWidth();
										cameraIndex = int(ptFrom.pt.x / subImageWidth);
										ptFrom.pt.x = ptFrom.pt.x - (subImageWidth*float(cameraIndex));
									}
									
									float depth = 0.0f;
									if(!sFrom.sensorData().cameraModels().empty())
									{
										depth = util3d::transformPoint(p, sFrom.sensorData().cameraModels()[cameraIndex].localTransform().inverse()).z;
									}
									else
									{
										UASSERT(!sFrom.sensorData().stereoCameraModels().empty());
										depth = util3d::transformPoint(p, sFrom.sensorData().stereoCameraModels()[cameraIndex].localTransform().inverse()).z;
									}


									wordReferences.at(wordId).insert(std::make_pair(sFrom.id(), FeatureBA(ptFrom, depth, descriptorFrom, cameraIndex)));
									frameToWordMap.insert(std::make_pair(sFrom.id(), std::map<cv::KeyPoint, int, KeyPointCompare>()));
									frameToWordMap.at(sFrom.id()).insert(std::make_pair(ptFrom, wordId));
								}

								if(!toAlreadyAdded)
								{
									cv::Mat descriptorTo;
									if(!sTo.getWordsDescriptors().empty())
									{
										UASSERT(indexTo < sTo.getWordsDescriptors().rows);
										descriptorTo = sTo.getWordsDescriptors().row(indexTo);
									}

									int cameraIndex = 0;
									if(sTo.sensorData().cameraModels().size()>1 || sTo.sensorData().stereoCameraModels().size()>1)
									{
										float subImageWidth = sTo.sensorData().cameraModels().size()>1?sTo.sensorData().cameraModels()[0].imageWidth():sTo.sensorData().stereoCameraModels()[0].left().imageWidth();
										cameraIndex = int(ptTo.pt.x / subImageWidth);
										ptTo.pt.x = ptTo.pt.x - (subImageWidth*float(cameraIndex));
									}

									float depth = 0.0f;
									if(!sTo.getWords3().empty())
									{
										UASSERT(indexTo < (int)sTo.getWords3().size());
										const cv::Point3f & pt = sTo.getWords3()[indexTo];
										if(!sTo.sensorData().cameraModels().empty())
										{
											depth = util3d::transformPoint(pt, sTo.sensorData().cameraModels()[cameraIndex].localTransform().inverse()).z;
										}
										else
										{
											UASSERT(!sTo.sensorData().stereoCameraModels().empty());
											depth = util3d::transformPoint(pt, sTo.sensorData().stereoCameraModels()[cameraIndex].localTransform().inverse()).z;
										}
									}

									wordReferences.at(wordId).insert(std::make_pair(sTo.id(), FeatureBA(ptTo, depth, descriptorTo, cameraIndex)));
									frameToWordMap.insert(std::make_pair(sTo.id(), std::map<cv::KeyPoint, int, KeyPointCompare>()));
									frameToWordMap.at(sTo.id()).insert(std::make_pair(ptTo, wordId));
								}
							}
						}
						++edgeWithWordsAdded;
					}
					else
					{
						UWARN("Not enough inliers (%d) between %d and %d", (int)info.inliersIDs.size(), sFrom.id(), sTo.id());
					}
				}
			}
		}
	}
	size_t observationsTotal = 0;
	for(std::map<int, std::map<int, FeatureBA> >::const_iterator iter=wordReferences.begin(); iter!=wordReferences.end(); ++iter)
	{
		observationsTotal += iter->second.size();
	}
	// Mirrors the track builder's summary so the two strategies can be read
	// against each other straight from the logs.
	UINFO("BA words: %d words / %d observations from %d/%d links (%.2fs)",
			wordCount, (int)observationsTotal, edgeWithWordsAdded, (int)links.size(), pairTimer.elapsed());
	reportProgress(uFormat("Correspondences matched: %d of %d links produced words (%.1fs)",
			edgeWithWordsAdded, (int)links.size(), pairTimer.elapsed()));
	if(links.empty())
	{
		UERROR("No links found for BA?!");
	}
	else if(wordCount == 0)
	{
		UERROR("No words added for BA?!");
	}
}

void Optimizer::computeBACorrespondencesTracks(
		const std::map<int, Transform> & poses,
		const std::multimap<int, Link> & links,
		BAFrameSource & frames,
		std::map<int, cv::Point3f> & points3DMap,
		std::map<int, std::map<int, FeatureBA> > & wordReferences,
		bool rematchFeatures,
		bool useLinkTransformAsGuess,
		ParametersMap registrationParameters)
{
	UDEBUG("rematchFeatures=%d minTrackLength=%d", rematchFeatures?1:0, baMinTrackLength_);

	// Same defaults as the legacy builder, so the two strategies stay comparable.
	registrationParameters.insert(ParametersPair(Parameters::kVisEstimationType(), "1"));
	registrationParameters.insert(ParametersPair(Parameters::kVisPnPReprojError(), "5"));
	registrationParameters.insert(ParametersPair(Parameters::kVisMinInliers(), "6"));
	registrationParameters.insert(ParametersPair(Parameters::kVisCorNNDR(), "0.6"));
	int matchThreads = 1;
	int nnType = Parameters::defaultVisCorNNType();
	Parameters::parse(registrationParameters, Parameters::kVisCorNNType(), nnType);
#ifdef _OPENMP
	if(nnType != 4)
	{
		matchThreads = omp_get_max_threads();
	}
	else
	{
		UINFO("BA matching is serial: %s=4 (GPU brute-force) is not OpenMP-safe.",
				Parameters::kVisCorNNType().c_str());
	}
#endif

	// Source 2 of the edge construction: same entry point, but a pose guess is
	// supplied, which makes RegistrationVis project the from-frame's 3D words into
	// the to-image and restrict candidates to a radius around the predicted pixel.
	// The ratio test is then computed only among candidates inside that window, so
	// a distractor elsewhere in the image cannot veto a true match and the ratio can
	// be relaxed. Nothing photometric is involved: the gate protects descriptor
	// power rather than replacing it.
	ParametersMap guidedParameters = registrationParameters;
	uInsert(guidedParameters, ParametersPair(Parameters::kVisCorNNDR(), uNumber2Str(baGuidedNNDR_)));
	uInsert(guidedParameters, ParametersPair(Parameters::kVisCorGuessWinSize(), uNumber2Str(baGuidedWinSize_)));

	bool guidedMatching = baGuidedMatching_;

	ObservationUnionFind uf;
	std::vector<TrackObservation> observations;

	//
	// Stage (a): pair proposal. The graph links come first; frustum overlap on the
	// optimized poses may add more. A candidate creates no correspondence by itself.
	//
	std::vector<PairCandidate> candidates;
	std::set<std::pair<int, int> > candidateSet;
	for(std::multimap<int, Link>::const_iterator iter=links.lower_bound(1); iter!=links.end(); ++iter)
	{
		Link link = iter->second;
		if(link.from() <= 0 || link.to() <= 0 ||
		   link.type() == Link::kPosePrior || link.type() == Link::kLandmark ||
		   link.type() == Link::kGravity || link.type() == Link::kVirtualClosure)
		{
			continue;
		}
		if(link.to() < link.from())
		{
			link = link.inverse();
		}
		if(link.to() == link.from() ||
		   !frames.has(link.from()) ||
		   !frames.has(link.to()) ||
		   !uContains(poses, link.from()))
		{
			continue;
		}

		if(frames.weight(link.from()) < 0) // ignore intermediate links
		{
			continue;
		}

		int toId = link.to();
		bool transformValid = true;
		if(frames.weight(toId) < 0)
		{
			// Hop over intermediate nodes to the next real one. The link's stored
			// transform no longer describes the resulting pair, so drop it.
			for(std::multimap<int, Link>::const_iterator jter=links.find(toId);
				jter!=links.end() && frames.has(jter->second.to()) &&
				frames.weight(toId) < 0;
				++jter)
			{
				toId = jter->second.to();
				transformValid = false;
			}
			if(frames.weight(toId) < 0)
			{
				continue;
			}
		}

		// The legacy builder never checks that the "to" frame has a pose, so it can
		// emit observations for frames outside the problem. The hop above can also
		// land on such a frame.
		if(!uContains(poses, toId))
		{
			UDEBUG("Skipping link %d->%d: no pose for %d", link.from(), toId, toId);
			continue;
		}

		if(candidateSet.insert(orderedPair(link.from(), toId)).second)
		{
			candidates.push_back(PairCandidate(link.from(), toId,
					(useLinkTransformAsGuess && transformValid)? link.transform() : Transform()));
		}
	}
	const int linkCandidates = (int)candidates.size();

	// Cross-body pairs around each loop closure. These carry the parallax the loop
	// pair itself does not have: its two frames are nearly co-located, which is what
	// made them a loop closure in the first place.
	{
		const std::vector<std::pair<int, int> > windowed = proposeWindowPairs(poses, links, candidateSet);
		for(size_t i=0; i<windowed.size(); ++i)
		{
			if(frames.has(windowed[i].first) && frames.has(windowed[i].second) &&
			   candidateSet.insert(windowed[i]).second)
			{
				// No link, hence no stored transform: the guess can only come from
				// the optimized poses, which guided matching does anyway.
				candidates.push_back(PairCandidate(windowed[i].first, windowed[i].second, Transform()));
			}
		}
	}
	const int windowCandidates = (int)candidates.size() - linkCandidates;

	if(baProposePairs_)
	{
		const std::vector<std::pair<int, int> > proposed = proposeFrustumPairs(poses, frames, candidateSet);
		for(size_t i=0; i<proposed.size(); ++i)
		{
			// No link, hence no stored transform: the guess can only come from the
			// optimized poses, which guided matching does anyway.
			candidates.push_back(PairCandidate(proposed[i].first, proposed[i].second, Transform()));
		}
	}

	const int allProposedPairs = (int)candidates.size();
	const int frustumCandidates = allProposedPairs-linkCandidates-windowCandidates;
	std::vector<PairCandidate> deferred;
	int fallbackPairs = 0;
	if(baExtraPairBudget_ > 0 && candidates.size() > (size_t)linkCandidates)
	{
		UTimer selectionTimer;
		std::map<int, FrustumFrame> geometry;
		std::map<int, ba::PairGeometry> selectionGeometry;
		std::vector<std::pair<int,int> > protectedPairs;
		for(const auto & c : candidates)
			for(int id : {c.fromId,c.toId})
				if(!geometry.count(id))
				{
					FrustumFrame f=budgetFrame(id,poses.at(id),frames);
					std::vector<double> depths;
					for(const auto & point : f.points) depths.push_back(cv::norm(point-f.center));
					double depth=1.0;
					if(!depths.empty())
					{
						std::nth_element(depths.begin(),depths.begin()+depths.size()/2,depths.end());
						depth=depths[depths.size()/2];
					}
					selectionGeometry[id]={{f.center.x,f.center.y,f.center.z},depth};
					geometry.emplace(id,std::move(f));
				}
		for(int i=0;i<linkCandidates;++i) protectedPairs.push_back({candidates[i].fromId,candidates[i].toId});
		std::vector<ba::ExtraPair> extras;
		for(size_t i=linkCandidates;i<candidates.size();++i)
		{
			const auto & c=candidates[i];
			extras.push_back({c.fromId,c.toId,budgetQuality(geometry.at(c.fromId),geometry.at(c.toId))});
		}
		const auto selected=ba::selectExtraPairs(extras,protectedPairs,selectionGeometry,baExtraPairBudget_);
		std::vector<PairCandidate> kept(candidates.begin(),candidates.begin()+linkCandidates);
		for(size_t i=0;i<selected.size();++i)
			(selected[i]?kept:deferred).push_back(candidates[linkCandidates+i]);
		candidates.swap(kept);
		UINFO("BA pair budget: budget=%d proposed=%d protected=%d selected=%d deferred=%d selection_seconds=%.3f",
			baExtraPairBudget_,allProposedPairs,linkCandidates,(int)candidates.size(),(int)deferred.size(),selectionTimer.elapsed());
	}

	// Matched in pair order rather than proposal order, so that consecutive batches ask
	// for overlapping frames and a frame's descriptors are read once instead of once per
	// link that touches it. Union-find does not care: its partition is the same whatever
	// order the unions arrive in.
	std::sort(candidates.begin(), candidates.end(), PairCandidateOrder());

	//
	// Stage (b): edge construction. Every accepted correspondence unites two
	// observation nodes; no landmark is created yet.
	//
	int edgeWithWordsAdded = 0;
	int guidedEdgesAdded = 0;

	// Matching a pair is independent of every other pair and is the dominant cost of this
	// function; uniting the results is not independent, because the union-find and the
	// observation table are shared. So the two are separated: match in parallel, then
	// insert serially.
	//
	// The chunk is also the unit of residency: the frames one chunk needs are read
	// together, matched, and then allowed to fall out of the frame source. That is what
	// keeps descriptors -- 48GB of them on the map this was sized against -- from having
	// to be resident all at once.
	const int chunkSize = 256;
	std::vector<std::vector<std::pair<int, int> > > wordCorrs(chunkSize);
	std::vector<std::vector<std::pair<int, int> > > guidedCorrs(chunkSize);

	// Confirm descriptors exist before building matchers. SuperPoint + rematch + guided
	// would otherwise construct two RegistrationVis pools and then match every pair twice.
	if(guidedMatching && !candidates.empty())
	{
		std::set<int> firstFrames;
		const size_t firstEnd = std::min(candidates.size(), (size_t)chunkSize);
		for(size_t i=0; i<firstEnd; ++i)
		{
			firstFrames.insert(candidates[i].fromId);
			firstFrames.insert(candidates[i].toId);
		}
		frames.pin(firstFrames);
		bool anyDescriptors = false;
		for(std::set<int>::const_iterator iter=firstFrames.begin();
				!anyDescriptors && iter!=firstFrames.end(); ++iter)
		{
			const Signature * s = frames.pinned(*iter);
			anyDescriptors = s && !s->getWordsDescriptors().empty();
		}
		if(!anyDescriptors)
		{
			UWARN("%s is enabled but no signature has word descriptors, guided matching is disabled. "
				  "If it is an old database, do rtabmap-reprocess so that signatures contain word descriptors.",
				  Parameters::kOptimizerBAGuidedMatching().c_str());
			guidedMatching = false;
		}
	}
	// rematch + guided is a second, pose-windowed match of the same descriptors. The
	// unguided global rematch (legacy source 1) is redundant for SuperPoint, whose word
	// ids are unique and cannot be matched by BoW equality.
	const bool doSource1 = !rematchFeatures || !guidedMatching;
	UINFO("BA matching: %d pairs, rematch=%d guided=%d source1=%d threads=%d nnType=%d.",
			(int)candidates.size(), rematchFeatures?1:0, guidedMatching?1:0, doSource1?1:0,
			matchThreads, nnType);

	// RegistrationVis carries state between calls, so a thread cannot share one. Built
	// once here rather than per chunk, since construction parses parameters and builds
	// matchers. Pools the pass will not use are skipped.
	std::vector<std::shared_ptr<RegistrationVis> > regPool;
	if(doSource1)
	{
		for(int t=0; t<matchThreads; ++t)
		{
			regPool.push_back(std::shared_ptr<RegistrationVis>(new RegistrationVis(registrationParameters)));
		}
	}
	std::vector<std::shared_ptr<RegistrationVis> > regGuidedPool;
	if(guidedMatching)
	{
		for(int t=0; t<matchThreads; ++t)
		{
			regGuidedPool.push_back(std::shared_ptr<RegistrationVis>(new RegistrationVis(guidedParameters)));
		}
	}

	// The persisted match store, when the caller supplied a database and the database is
	// new enough to have the table. Only the guided pass is cached: word equality does no
	// descriptor matching, and its matches are already on disk as the word ids themselves.
	//
	// Reads happen before the parallel region and writes after it, never inside: the
	// sqlite handle is serialized by a mutex, so touching it from every thread would
	// trade the matching cost for lock contention.
	DBDriver * store = 0;
	std::vector<PairMatchRecord> cached(chunkSize);
	std::vector<PairMatchRecord> fresh(chunkSize);
	int cacheHits = 0;
	int cacheWrites = 0;

	UTimer matchTimer;
	double lastReport = 0.0;
	if(!reportProgress(uFormat("Correspondences 0/%d", (int)candidates.size())))
	{
		canceled_ = true;
		return;
	}
	for(size_t begin=0; begin<candidates.size(); )
	{
		const size_t end = std::min(candidates.size(), begin+(size_t)chunkSize);
		for(size_t i=begin; i<end; ++i)
		{
			wordCorrs[i-begin].clear();
			guidedCorrs[i-begin].clear();
			cached[i-begin] = PairMatchRecord();
			fresh[i-begin] = PairMatchRecord();
		}

		// Read this chunk's frames, and let the previous chunk's go.
		std::set<int> chunkFrames;
		for(size_t i=begin; i<end; ++i)
		{
			chunkFrames.insert(candidates[i].fromId);
			chunkFrames.insert(candidates[i].toId);
		}
		frames.pin(chunkFrames);

		// Resolving a match back to the node's own feature index needs this, and it is
		// only ever built for the frames of the chunk in hand.
		std::map<int, FeatureKeyIndex> keyIndex;
		for(std::set<int>::const_iterator iter=chunkFrames.begin(); iter!=chunkFrames.end(); ++iter)
		{
			keyIndex.insert(std::make_pair(*iter, FeatureKeyIndex(frames.keypoints(*iter))));
		}

		if(store && guidedMatching)
		{
			(void)cacheHits;
		}

#ifdef _OPENMP
		#pragma omp parallel for schedule(dynamic) num_threads(matchThreads)
#endif
		for(int i=(int)begin; i<(int)end; ++i)
		{
			int thread = 0;
#ifdef _OPENMP
			thread = omp_get_thread_num();
#endif
			// An exception escaping an OpenMP region terminates the process; one bad pair
			// must not take the whole build with it.
			try
			{
				const PairCandidate & candidate = candidates[i];
				const Signature * sFrom = frames.pinned(candidate.fromId);
				const Signature * sTo = frames.pinned(candidate.toId);
				if(sFrom == 0 || sTo == 0)
				{
					continue;   // frame could not be read; the pair simply contributes nothing
				}
				const FeatureKeyIndex & kiFrom = keyIndex.at(candidate.fromId);
				const FeatureKeyIndex & kiTo = keyIndex.at(candidate.toId);

				// Source 1 -- word equality: words sharing an id were already matched at
				// mapping time. Free, high precision, and they survived the dataset's real
				// illumination conditions. (When rematchFeatures is set without guided
				// matching, this pass matches descriptors without a guess, as the legacy
				// builder does.) Skipped when rematch and guided are both on: SuperPoint
				// word ids are unique, so the unguided rematch is a second full match of
				// the same pair that the pose-windowed pass already covers.
				if(doSource1)
				{
					matchPair(*sFrom, *sTo, candidate.fromId, candidate.toId,
							candidate.linkTransform, !rematchFeatures, false,
							*regPool[thread], kiFrom, kiTo, wordCorrs[i-(int)begin], 0);
				}

				// Source 2 -- guided descriptor matching. Recovers the landmarks word
				// equality misses: pairs never matched at mapping time (every proposed
				// pair, for a start) and landmarks split by vocabulary quantization. The
				// optimized relative pose is a strictly better guess than any single
				// link's transform.
				if(guidedMatching)
				{
					const Transform guess =
							poses.at(candidate.fromId).inverse() * poses.at(candidate.toId);
					const PairMatchRecord & hit = cached[i-(int)begin];
					if(hit.valid)
					{
						// Replay: matching is skipped, the geometry is not. The pair still
						// goes through registration, so PnP and the inlier decision are
						// recomputed against the current poses.
						Signature reFrom = *sFrom;
						Signature reTo = *sTo;
						applyCachedMatches(reFrom, reTo, hit.pairs);
						matchPair(reFrom, reTo, candidate.fromId, candidate.toId,
								guess, true, baGuidedSkipPnP_,
								*regGuidedPool[thread], kiFrom, kiTo,
								guidedCorrs[i-(int)begin], 0);
					}
					else
					{
						matchPair(*sFrom, *sTo, candidate.fromId, candidate.toId,
								guess, false, baGuidedSkipPnP_,
								*regGuidedPool[thread], kiFrom, kiTo,
								guidedCorrs[i-(int)begin],
								store? &fresh[i-(int)begin] : 0);
					}
				}
			}
			catch(const std::exception & e)
			{
				UERROR("Matching pair %d->%d threw (%s), pair dropped.",
						candidates[i].fromId, candidates[i].toId, e.what());
			}
			catch(...)
			{
				UERROR("Matching pair %d->%d threw, pair dropped.",
						candidates[i].fromId, candidates[i].toId);
			}
		}

		for(size_t i=begin; i<end; ++i)
		{
			bool added = false;
			if(insertPairCorrespondences(frames, candidates[i].fromId, candidates[i].toId,
					wordCorrs[i-begin], observations, uf))
			{
				added = true;
			}
			if(insertPairCorrespondences(frames, candidates[i].fromId, candidates[i].toId,
					guidedCorrs[i-begin], observations, uf))
			{
				++guidedEdgesAdded;
				added = true;
			}
			if(added)
			{
				++edgeWithWordsAdded;
			}
		}

		if(store)
		{
			(void)cacheWrites;
		}

		if(!reportProgress(uFormat("Correspondences %d/%d", (int)end, (int)candidates.size())))
		{
			canceled_ = true;
			points3DMap.clear();
			wordReferences.clear();
			return;
		}

		if(matchTimer.elapsed() - lastReport > 60.0)
		{
			lastReport = matchTimer.elapsed();
			UINFO("BA pairs: %d/%d matched (%.0fs), %d observations, %d MB of descriptors resident.",
					(int)end, (int)candidates.size(), lastReport, (int)observations.size(),
					(int)(frames.residentBytes()/(1024*1024)));
		}
		begin = end;
		if(begin == candidates.size() && !deferred.empty())
		{
			UTimer fallbackTimer;
			const auto weak=weakBudgetFrames(observations,uf,frames);
			std::sort(deferred.begin(),deferred.end(),PairCandidateOrder());
			for(const auto & c : deferred)
				if(weak.count(c.fromId) || weak.count(c.toId))
				{
					candidates.push_back(c);
					++fallbackPairs;
				}
			UINFO("BA pair budget fallback: weak_frames=%d restored=%d final_pairs=%d coverage_seconds=%.3f",
				(int)weak.size(),fallbackPairs,(int)candidates.size(),fallbackTimer.elapsed());
			deferred.clear(); // exactly one coverage fallback, bounded by original candidates
		}
	}

	if(store)
	{
		UINFO("BA match cache: %d of %d candidate pairs replayed from the database, "
			  "%d newly stored.", cacheHits, (int)candidates.size(), cacheWrites);
	}

	UINFO("BA pairs: %d matched candidates (proposal sources: %d links, %d cross-body windows, %d frustum), "
		  "%d produced correspondences, %d of them through guided matching. "
		  "%d frame reads for %d frames, %.2fs (%.1f pairs/s).",
		  (int)candidates.size(), linkCandidates, windowCandidates,
		  frustumCandidates,
		  edgeWithWordsAdded, guidedEdgesAdded,
		  (int)frames.loads(), (int)frames.ids().size(), matchTimer.elapsed(),
		  matchTimer.elapsed() > 0.0 ? (double)candidates.size()/matchTimer.elapsed() : 0.0);
	reportProgress(uFormat("Correspondences matched: %d of %d pairs in %.1fs",
			edgeWithWordsAdded, (int)candidates.size(), matchTimer.elapsed()));

	//
	// Pass 2: group observations into tracks, then emit one landmark per track.
	//
	// Grouped by sorting rather than by a map keyed on the root: there is one entry per
	// observation and there can be tens of millions of them, where a std::map node costs
	// some 48 bytes of overhead on top of the 4 the value needs.
	//
	std::vector<std::pair<int, int> > byRoot;   // <union-find root, observation index>
	byRoot.reserve(observations.size());
	for(size_t i=0; i<observations.size(); ++i)
	{
		byRoot.push_back(std::make_pair(uf.find((int)i), (int)i));
	}
	std::sort(byRoot.begin(), byRoot.end());

	int wordCount = 0;
	int rejectedConflict = 0;
	int rejectedShort = 0;
	int rejectedTriangulation = 0;
	int rejectedAngle = 0;
	int rejectedReprojection = 0;
	int rejectedDepthMismatch = 0;
	size_t observationsKept = 0;
	std::map<int, int> lengthHistogram; // <track length, count>

	std::vector<int> nodes;
	std::vector<int> kept;
	std::vector<ResolvedObservation> resolved;
	UTimer trackTimer;
	double lastTrackReport = -1.0;
	if(!reportProgress(uFormat("Tracks 0/%d", (int)byRoot.size())))
	{
		canceled_ = true;
		points3DMap.clear();
		wordReferences.clear();
		return;
	}
	for(size_t begin=0; begin<byRoot.size(); )
	{
		size_t end = begin;
		while(end < byRoot.size() && byRoot[end].first == byRoot[begin].first)
		{
			++end;
		}
		nodes.clear();
		for(size_t i=begin; i<end; ++i)
		{
			nodes.push_back(byRoot[i].second);
		}
		begin = end;
		if(begin == byRoot.size() || trackTimer.elapsed() - lastTrackReport > 0.05)
		{
			lastTrackReport = trackTimer.elapsed();
			if(!reportProgress(uFormat("Tracks %d/%d", (int)begin, (int)byRoot.size())))
			{
				canceled_ = true;
				points3DMap.clear();
				wordReferences.clear();
				return;
			}
		}

		// A track observing the same frame twice is a contradiction: one landmark
		// cannot project to two places in one image. It means the union chain
		// merged through a bad match, so drop the whole track.
		std::set<int> frameIds;
		bool conflict = false;
		for(size_t i=0; i<nodes.size(); ++i)
		{
			if(!frameIds.insert(observations[nodes[i]].frameId).second)
			{
				conflict = true;
				break;
			}
		}
		if(conflict)
		{
			++rejectedConflict;
			continue;
		}

		// Resolve every view before judging the track's length, and triangulate from all
		// of them -- unlike the legacy builder the position is not taken from a single
		// frame's depth. An observation whose frame lost its calibration, or whose
		// feature index falls outside that frame, cannot be part of the problem, so it is
		// dropped here rather than triangulated around and then emitted anyway.
		resolved.clear();
		kept.clear();
		std::vector<TrackView> views;
		views.reserve(nodes.size());
		for(size_t i=0; i<nodes.size(); ++i)
		{
			const TrackObservation & obs = observations[nodes[i]];
			ResolvedObservation r;
			if(!resolveObservation(frames, obs, r))
			{
				continue;
			}
			CameraModel model;
			Transform cameraToWorld;
			if(!getObservationCamera(frames, poses, obs.frameId, r.cameraIndex,
					model, cameraToWorld))
			{
				continue;
			}
			TrackView view;
			view.worldToCamera = cameraToWorld.inverse();
			view.center = cv::Point3f(cameraToWorld.x(), cameraToWorld.y(), cameraToWorld.z());
			view.fx = (float)model.fx();
			view.fy = (float)model.fy();
			view.cx = (float)model.cx();
			view.cy = (float)model.cy();
			view.u = r.kpt.pt.x;
			view.v = r.kpt.pt.y;
			view.measuredDepth = r.hasDepth? r.depth : 0.0f;
			views.push_back(view);
			resolved.push_back(r);
			kept.push_back(nodes[i]);
		}

		if((int)kept.size() < baMinTrackLength_ || kept.size() < 2)
		{
			++rejectedShort;
			continue;
		}

		const TriangulationResult tri = triangulateTrack(views, baTriangulationDepthSeed_);
		if(!tri.valid)
		{
			++rejectedTriangulation;
			continue;
		}
		if(baTriangulationMinAngle_ > 0.0f && tri.maxAngleDeg < baTriangulationMinAngle_)
		{
			++rejectedAngle;
			continue;
		}
		if(baTriangulationMaxReprojError_ > 0.0f && tri.maxReprojError > baTriangulationMaxReprojError_)
		{
			++rejectedReprojection;
			continue;
		}
		// The consistency test the legacy builder cannot do: does the geometry
		// agree with what the depth sensor reported? Disagreement means either a
		// bad match or a bad depth reading, and both should drop the track.
		if(baTriangulationMaxDepthError_ > 0.0f && tri.hasDepth &&
		   tri.medianDepthError > baTriangulationMaxDepthError_)
		{
			++rejectedDepthMismatch;
			continue;
		}

		int wordId = ++wordCount;
		points3DMap.insert(points3DMap.end(), std::make_pair(wordId, tri.point));
		std::map<int, FeatureBA> & refs =
				wordReferences.insert(wordReferences.end(),
						std::make_pair(wordId, std::map<int, FeatureBA>()))->second;
		for(size_t i=0; i<kept.size(); ++i)
		{
			const TrackObservation & obs = observations[kept[i]];
			// No descriptor: no back-end reads one, and a descriptor here is a cv::Mat row
			// view that would pin its whole frame's descriptor matrix for the life of the
			// track. The feature index is carried instead, which is what lets a consumer
			// find its way back to the frame's own keypoints.
			refs.insert(refs.end(), std::make_pair(obs.frameId,
					FeatureBA(resolved[i].kpt, resolved[i].depth, cv::Mat(),
							resolved[i].cameraIndex, obs.featureIndex)));
		}
		observationsKept += kept.size();
		++lengthHistogram[(int)kept.size()];
	}

	UINFO("BA tracks: %d tracks / %d observations from %d/%d pairs in %.2fs "
		  "(rejected: %d too short, %d frame conflict, %d triangulation failed, "
		  "%d low parallax, %d reprojection error, %d depth mismatch)",
		  wordCount, (int)observationsKept, edgeWithWordsAdded, (int)candidates.size(),
		  trackTimer.elapsed(),
		  rejectedShort, rejectedConflict, rejectedTriangulation,
		  rejectedAngle, rejectedReprojection, rejectedDepthMismatch);
	reportProgress(uFormat("Tracks built: %d tracks, %d observations (%.1fs)",
			wordCount, (int)observationsKept, trackTimer.elapsed()));
	for(std::map<int, int>::const_iterator iter=lengthHistogram.begin(); iter!=lengthHistogram.end(); ++iter)
	{
		UINFO("BA tracks:   length %d: %d tracks (%.1f%%)",
			  iter->first, iter->second, wordCount>0?100.0f*float(iter->second)/float(wordCount):0.0f);
	}

	std::map<int,int> supportedFrames;
	int threeViewTracks=0;
	for(const auto & track : wordReferences)
		if(track.second.size() >= 3)
		{
			++threeViewTracks;
			for(const auto & view : track.second) ++supportedFrames[view.first];
		}
	int wellSupported=0;
	for(const auto & f : supportedFrames) if(f.second>=20) ++wellSupported;
	UINFO("BA coverage: budget=%d proposed_pairs=%d matched_pairs=%d fallback_pairs=%d three_view_tracks=%d frames_with_three_view_tracks=%d frames_with_20_tracks=%d total_frames=%d",
		baExtraPairBudget_,allProposedPairs,(int)candidates.size(),fallbackPairs,threeViewTracks,
		(int)supportedFrames.size(),wellSupported,(int)frames.ids().size());

	applyLoopScope(links, points3DMap, wordReferences);

	if(links.empty())
	{
		UERROR("No links found for BA?!");
	}
	else if(wordReferences.empty())
	{
		UERROR("No words added for BA?!");
	}
}

void Optimizer::applyLoopScope(
		const std::multimap<int, Link> & links,
		std::map<int, cv::Point3f> & points3DMap,
		std::map<int, std::map<int, FeatureBA> > & wordReferences) const
{
	if(baLoopScope_ <= 0 || wordReferences.empty())
	{
		return;
	}

	// frame -> tracks it observes
	std::map<int, std::set<int> > frameTracks;
	for(std::map<int, std::map<int, FeatureBA> >::const_iterator iter=wordReferences.begin(); iter!=wordReferences.end(); ++iter)
	{
		for(std::map<int, FeatureBA>::const_iterator jter=iter->second.begin(); jter!=iter->second.end(); ++jter)
		{
			frameTracks[jter->first].insert(iter->first);
		}
	}

	// Every non-neighbor loop closure induces its own cluster: the link's tracks
	// are the ones both its ends observe -- the landmarks the link is actually a
	// statement about -- and a frame joins the cluster when it witnesses enough of
	// them. The BA problem is the union of those clusters. Nothing is hop-counted:
	// a frame either sees what the loop sees or it does not.
	std::set<int> cluster;
	int loopLinks = 0, loopLinksWithTracks = 0;
	for(std::multimap<int, Link>::const_iterator iter=links.lower_bound(1); iter!=links.end(); ++iter)
	{
		const Link & link = iter->second;
		if(link.from() == link.to() ||
		   link.type() == Link::kNeighbor || link.type() == Link::kNeighborMerged)
		{
			continue;
		}
		if(link.type() != Link::kGlobalClosure && link.type() != Link::kLocalSpaceClosure &&
		   link.type() != Link::kLocalTimeClosure && link.type() != Link::kUserClosure)
		{
			continue;
		}
		++loopLinks;

		std::map<int, std::set<int> >::const_iterator f = frameTracks.find(link.from());
		std::map<int, std::set<int> >::const_iterator t = frameTracks.find(link.to());
		if(f == frameTracks.end() || t == frameTracks.end())
		{
			continue;
		}
		std::vector<int> linkTracks;
		std::set_intersection(f->second.begin(), f->second.end(),
							  t->second.begin(), t->second.end(),
							  std::back_inserter(linkTracks));
		if(linkTracks.empty())
		{
			continue;
		}
		++loopLinksWithTracks;

		// Count, per frame, how many of this link's tracks it witnesses. Walking
		// the link's tracks rather than all frames keeps this proportional to the
		// loop's own observations instead of to the size of the map.
		std::map<int, int> witnessed;
		for(size_t i=0; i<linkTracks.size(); ++i)
		{
			const std::map<int, FeatureBA> & refs = wordReferences.at(linkTracks[i]);
			for(std::map<int, FeatureBA>::const_iterator r=refs.begin(); r!=refs.end(); ++r)
			{
				++witnessed[r->first];
			}
		}
		for(std::map<int, int>::const_iterator w=witnessed.begin(); w!=witnessed.end(); ++w)
		{
			if(w->second >= baLoopScopeMinTracks_)
			{
				cluster.insert(w->first);
			}
		}
		// The link's own two frames are in its cluster by definition, whatever the
		// threshold: they are what the link constrains.
		cluster.insert(link.from());
		cluster.insert(link.to());
	}

	// Trim: an observation survives only if its frame is in the cluster. This runs
	// after triangulation on purpose -- a landmark that survives was still
	// positioned using every one of its views, so what is dropped is reprojection
	// residuals away from any loop, not the geometry that placed the point.
	const int minLength = baMinTrackLength_ > 2? baMinTrackLength_ : 2;
	size_t observationsBefore = 0, observationsAfter = 0;
	const size_t tracksBefore = wordReferences.size();
	for(std::map<int, std::map<int, FeatureBA> >::iterator iter=wordReferences.begin(); iter!=wordReferences.end(); )
	{
		observationsBefore += iter->second.size();
		for(std::map<int, FeatureBA>::iterator jter=iter->second.begin(); jter!=iter->second.end(); )
		{
			if(cluster.find(jter->first) == cluster.end())
			{
				iter->second.erase(jter++);
			}
			else
			{
				++jter;
			}
		}
		if((int)iter->second.size() < minLength)
		{
			points3DMap.erase(iter->first);
			wordReferences.erase(iter++);
		}
		else
		{
			observationsAfter += iter->second.size();
			++iter;
		}
	}

	UINFO("BA loop scope: %d loop links (%d with shared tracks) induced a cluster of "
		  "%d frames of %d (>=%d witnessed tracks). Kept %d/%d tracks, %d/%d observations.",
			loopLinks, loopLinksWithTracks, (int)cluster.size(), (int)frameTracks.size(),
			baLoopScopeMinTracks_,
			(int)wordReferences.size(), (int)tracksBefore,
			(int)observationsAfter, (int)observationsBefore);
}

void Optimizer::computeBACorrespondences(
		const std::map<int, Transform> & poses,
		const std::multimap<int, Link> & links,
		const std::map<int, Signature> & signatures,
		std::map<int, cv::Point3f> & points3DMap,
		std::map<int, std::map<int, FeatureBA> > & wordReferences,
		bool rematchFeatures,
		bool useLinkTransformAsGuess,
		ParametersMap registrationParameters,
		DBDriver * dbDriver)
{
	UDEBUG("rematchFeatures=%d strategy=%d dbDriver=%d signatures=%d",
			rematchFeatures?1:0, baStrategy_, dbDriver?1:0, (int)signatures.size());
	points3DMap.clear();
	wordReferences.clear();
	if(!signatures.empty())
	{
		MemoryFrameSource frames(signatures);
		computeBACorrespondencesFrom(poses, links, frames, points3DMap, wordReferences,
				rematchFeatures, useLinkTransformAsGuess, registrationParameters);
		return;
	}
	if(dbDriver == 0)
	{
		UERROR("No signatures and no database driver: cannot build BA correspondences.");
		return;
	}
	std::set<int> ids;
	for(std::map<int, Transform>::const_iterator iter=poses.lower_bound(1); iter!=poses.end(); ++iter)
	{
		ids.insert(iter->first);
	}
	DatabaseFrameSource frames(dbDriver, ids, (size_t)std::max(1, baFrameCacheMemory_)*1024*1024, this);
	if(isCanceled())
	{
		return;
	}
	computeBACorrespondencesFrom(poses, links, frames, points3DMap, wordReferences,
			rematchFeatures, useLinkTransformAsGuess, registrationParameters);
}

} /* namespace rtabmap */
