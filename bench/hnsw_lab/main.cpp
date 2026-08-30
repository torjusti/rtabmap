// FLANN (rtabmap's exact FlannIndex path) vs hnswlib on real SuperPoint
// descriptors, replicating the VWDictionary workload:
//   - build index over N words
//   - per-node quantization: knnSearch k=2 batches (checks=32 for FLANN)
//   - WM-saturation churn: add W words one by one + remove W words per round
// Data file: raw float32, row-major, dim columns (see extract_descriptors.py).

#include <rtabmap/core/FlannIndex.h>
#include <opencv2/opencv.hpp>
#include <hnswlib/hnswlib.h>
#include <omp.h>
#include <chrono>
#include <random>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static double now()
{
	return std::chrono::duration<double>(
		std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Args
{
	std::string file;
	int dim = 256;
	long nbase = 1000000;
	long nquery = 20000;
	int rounds = 100;
	int nodeWords = 620;    // words added AND removed per churn round
	int threads = 0;        // 0 = all
	int gtn = 200;          // queries used for recall ground truth
	int ef = 64, M = 16, efc = 200;
	bool skipFlann = false, skipHnsw = false;
};

int main(int argc, char** argv)
{
	Args a;
	if(argc < 4)
	{
		printf("usage: hnsw_lab <words.bin> <dim> <Nbase> [--queries N] [--rounds R] "
		       "[--node-words W] [--threads T] [--gt N] [--ef E] [--M m] [--efc E] "
		       "[--skip-flann] [--skip-hnsw]\n");
		return 1;
	}
	a.file = argv[1]; a.dim = atoi(argv[2]); a.nbase = atol(argv[3]);
	for(int i=4; i<argc; ++i)
	{
		std::string s = argv[i];
		if(s=="--queries") a.nquery = atol(argv[++i]);
		else if(s=="--rounds") a.rounds = atoi(argv[++i]);
		else if(s=="--node-words") a.nodeWords = atoi(argv[++i]);
		else if(s=="--threads") a.threads = atoi(argv[++i]);
		else if(s=="--gt") a.gtn = atoi(argv[++i]);
		else if(s=="--ef") a.ef = atoi(argv[++i]);
		else if(s=="--M") a.M = atoi(argv[++i]);
		else if(s=="--efc") a.efc = atoi(argv[++i]);
		else if(s=="--skip-flann") a.skipFlann = true;
		else if(s=="--skip-hnsw") a.skipHnsw = true;
	}
	if(a.threads <= 0) a.threads = omp_get_max_threads();
	omp_set_num_threads(a.threads);

	const long poolN = (long)a.rounds * a.nodeWords;
	const long need = a.nbase + a.nquery + poolN;

	FILE* f = fopen(a.file.c_str(), "rb");
	if(!f) { printf("cannot open %s\n", a.file.c_str()); return 1; }
	fseek(f, 0, SEEK_END);
	long avail = ftell(f) / (a.dim * 4);
	fseek(f, 0, SEEK_SET);
	if(need > avail)
	{
		printf("need %ld vectors but file has %ld\n", need, avail);
		fclose(f);
		return 1;
	}
	cv::Mat all((int)need, a.dim, CV_32F);
	size_t rd = fread(all.data, 4, (size_t)need * a.dim, f);
	fclose(f);
	if(rd != (size_t)need * a.dim) { printf("short read\n"); return 1; }

	cv::Mat base = all.rowRange(0, (int)a.nbase);
	cv::Mat queries = all.rowRange((int)a.nbase, (int)(a.nbase + a.nquery));
	cv::Mat pool = all.rowRange((int)(a.nbase + a.nquery), (int)need);
	cv::Mat probe = queries.rowRange(0, std::min<long>(1000, a.nquery));

	printf("data: base=%ld queries=%ld churnPool=%ld dim=%d threads=%d\n",
		a.nbase, a.nquery, poolN, a.dim, a.threads);

	// Ground truth (true NN over base) for the first gtn queries.
	std::vector<int> gt(a.gtn, -1);
	{
		double t0 = now();
		#pragma omp parallel for schedule(dynamic)
		for(int q=0; q<a.gtn; ++q)
		{
			const float* Q = queries.ptr<float>(q);
			float best = 1e30f; int bi = -1;
			for(long i=0; i<a.nbase; ++i)
			{
				const float* P = base.ptr<float>((int)i);
				float d = 0.f;
				for(int c=0; c<a.dim; ++c) { float x = Q[c]-P[c]; d += x*x; }
				if(d < best) { best = d; bi = (int)i; }
			}
			gt[q] = bi;
		}
		printf("ground truth (%d queries brute force): %.1fs\n", a.gtn, now()-t0);
	}

	std::mt19937 rng(42);

	//==================== FLANN (rtabmap FlannIndex) ====================
	if(!a.skipFlann)
	{
		printf("\n===== FLANN kd-tree (rtabmap FlannIndex, VWDictionary settings) =====\n");
		rtabmap::FlannIndex idx;
		double t0 = now();
		idx.buildIndex(rtabmap::FlannIndex::FLANN_INDEX_KDTREE, base, false, 2.0f);
		printf("build %ld pts: %.2fs\n", a.nbase, now()-t0);

		cv::Mat ind, dist;
		idx.knnSearch(probe, ind, dist, 2, 32, 0.0f, true, a.threads); // warm-up
		for(int checks : {32, 128, 512, 2048})
		{
			t0 = now();
			idx.knnSearch(queries, ind, dist, 2, checks, 0.0f, true, a.threads);
			double dt = now()-t0;
			int hit = 0;
			for(int q=0; q<a.gtn; ++q) if(ind.at<int>(q,0) == gt[q]) ++hit;
			printf("query %ld x k2 checks=%-5d: %7.3fs -> %8.2f ms/1k   recall@1=%.3f\n",
				a.nquery, checks, dt, 1000.0*dt/(a.nquery/1000.0), (double)hit/a.gtn);
		}

		// Churn: one addPoints(1 row) per word (as VWDictionary does) + removals.
		std::vector<unsigned int> live((size_t)a.nbase);
		for(long i=0; i<a.nbase; ++i) live[(size_t)i] = (unsigned int)i;
		printf("churn: %d rounds x (add %d + remove %d)\n", a.rounds, a.nodeWords, a.nodeWords);
		printf("%8s %12s %12s %14s %16s\n", "round", "add_ms", "rm_ms", "probe_ms/1k", "worst_add_ms");
		for(int r=0; r<a.rounds; ++r)
		{
			double addT = 0, worstAdd = 0, rmT = 0;
			for(int w=0; w<a.nodeWords; ++w)
			{
				cv::Mat row = pool.row(r*a.nodeWords + w);
				t0 = now();
				std::vector<unsigned int> ids = idx.addPoints(row);
				double dt = now()-t0;
				addT += dt; if(dt > worstAdd) worstAdd = dt;
				live.push_back(ids.front());
			}
			for(int w=0; w<a.nodeWords; ++w)
			{
				size_t pick = rng() % live.size();
				t0 = now();
				idx.removePoint(live[pick]);
				rmT += now()-t0;
				live[pick] = live.back(); live.pop_back();
			}
			if(r % 10 == 9 || r == 0)
			{
				t0 = now();
				idx.knnSearch(probe, ind, dist, 2, 32, 0.0f, true, a.threads);
				double qms = 1000.0*(now()-t0)/(probe.rows/1000.0);
				printf("%8d %12.1f %12.1f %14.1f %16.1f\n",
					r+1, 1000.0*addT, 1000.0*rmT, qms, 1000.0*worstAdd);
			}
		}
		printf("flann memory used: %.1f MB (indexed=%zu)\n",
			idx.memoryUsed()/1e6, idx.indexedFeatures());
	}

	//==================== HNSW (hnswlib) ====================
	if(!a.skipHnsw)
	{
		printf("\n===== HNSW (hnswlib, M=%d efc=%d ef=%d, replace_deleted) =====\n",
			a.M, a.efc, a.ef);
		hnswlib::L2Space space(a.dim);
		hnswlib::HierarchicalNSW<float> idx(&space, (size_t)need, a.M, a.efc, 100, true);
		double t0 = now();
		#pragma omp parallel for schedule(dynamic, 256)
		for(long i=0; i<a.nbase; ++i)
			idx.addPoint(base.ptr<float>((int)i), (size_t)i, false);
		printf("build %ld pts (parallel insert): %.2fs\n", a.nbase, now()-t0);

		std::vector<int> res((size_t)a.nquery, -1);
		for(int ef : {8, 16, 32, 64, 128})
		{
			idx.setEf((size_t)ef);
			t0 = now();
			#pragma omp parallel for schedule(dynamic, 64)
			for(long q=0; q<a.nquery; ++q)
			{
				auto pq = idx.searchKnn(queries.ptr<float>((int)q), 2);
				while(pq.size() > 1) pq.pop();
				res[(size_t)q] = (int)pq.top().second;
			}
			double dt = now()-t0;
			int hit = 0;
			for(int q=0; q<a.gtn; ++q) if(res[(size_t)q] == gt[q]) ++hit;
			printf("query %ld x k2 ef=%-5d: %7.3fs -> %8.2f ms/1k   recall@1=%.3f\n",
				a.nquery, ef, dt, 1000.0*dt/(a.nquery/1000.0), (double)hit/a.gtn);
		}
		idx.setEf((size_t)a.ef);

		std::vector<size_t> live((size_t)a.nbase);
		for(long i=0; i<a.nbase; ++i) live[(size_t)i] = (size_t)i;
		size_t nextLabel = (size_t)a.nbase;
		printf("churn: %d rounds x (parallel add %d + remove %d)\n", a.rounds, a.nodeWords, a.nodeWords);
		printf("%8s %14s %12s %14s\n", "round", "add_wall_ms", "rm_ms", "probe_ms/1k");
		std::vector<int> pres(probe.rows, -1);
		for(int r=0; r<a.rounds; ++r)
		{
			// One node's worth of new words, inserted as a parallel batch
			// (as an RTAB-Map integration would do once per node).
			size_t baseLabel = nextLabel;
			t0 = now();
			#pragma omp parallel for schedule(dynamic, 8)
			for(int w=0; w<a.nodeWords; ++w)
				idx.addPoint(pool.ptr<float>(r*a.nodeWords + w), baseLabel + w, true /*replace_deleted*/);
			double addWall = now()-t0;
			for(int w=0; w<a.nodeWords; ++w) live.push_back(nextLabel++);
			double rmT = 0;
			for(int w=0; w<a.nodeWords; ++w)
			{
				size_t pick = rng() % live.size();
				t0 = now();
				idx.markDelete(live[pick]);
				rmT += now()-t0;
				live[pick] = live.back(); live.pop_back();
			}
			if(r % 10 == 9 || r == 0)
			{
				t0 = now();
				#pragma omp parallel for schedule(dynamic, 64)
				for(int q=0; q<probe.rows; ++q)
				{
					auto pq = idx.searchKnn(probe.ptr<float>(q), 2);
					while(pq.size() > 1) pq.pop();
					pres[q] = (int)pq.top().second;
				}
				double qms = 1000.0*(now()-t0)/(probe.rows/1000.0);
				printf("%8d %14.1f %12.1f %14.1f\n",
					r+1, 1000.0*addWall, 1000.0*rmT, qms);
			}
		}
	}
	printf("\ndone\n");
	return 0;
}
