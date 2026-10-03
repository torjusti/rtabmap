/*
Copyright (c) 2010-2026, Mathieu Labbe - IntRoLab - Universite de Sherbrooke
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

#include <rtabmap/core/DBDriver.h>
#include <rtabmap/core/LoopPruning.h>
#include <rtabmap/core/Optimizer.h>
#include <rtabmap/core/ProgressState.h>
#include <rtabmap/utilite/UConversion.h>
#include <rtabmap/utilite/UFile.h>
#include <rtabmap/utilite/ULogger.h>
#include <rtabmap/utilite/UTimer.h>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <memory>
#include <signal.h>
#include <stdio.h>

using namespace rtabmap;

void showUsage()
{
	loopPruning::Options defaults;
	printf("\nUsage:\n"
			"rtabmap-pruneLoopClosures [options] database.db\n"
			"\n"
			"Removes inconsistent loop closures (greedy: optimize, remove the worst,\n"
			"repeat) and redundant ones (nodes already joined by a short path, keeping\n"
			"the loop closures that agree best with the map). Only loop closure links are\n"
			"removed; odometry, priors, gravity and landmark links are kept. Links are\n"
			"deleted from the database: make a copy first, or use --dry-run.\n"
			"\n"
			"Options:\n"
			"    --max_error #       Max translation residual of a loop closure (m, default %g, 0 disables).\n"
			"    --max_angle #       Max rotation residual of a loop closure (deg, default %g, 0 disables).\n"
			"    --max_ratio #       Max residual over the link's own std-dev (default %g, 0 disables).\n"
			"    --batch_radius #    Per greedy round, also remove inconsistent loop closures farther\n"
			"                        than this from those already removed (m, default %g). 0 removes\n"
			"                        all at once, -1 removes one per round.\n"
			"    --max_rounds #      Max greedy rounds (default %d).\n"
			"    --stall_rounds #    Stop greedy pruning when the worst error improved less than\n"
			"                        --stall_improvement over this many rounds, and put back the\n"
			"                        loop closures removed during them (default %d, 0 disables).\n"
			"    --stall_improvement #\n"
			"                        See --stall_rounds (%%, default %g).\n"
			"    --full_solve_every #\n"
			"                        With GTSAM, optimize incrementally (iSAM2) and solve from\n"
			"                        scratch every this many optimizations (default %d). The final\n"
			"                        optimization is always solved from scratch. Pass\n"
			"                        --GTSAM/Incremental false to always solve from scratch.\n"
			"    --min_graph_distance #\n"
			"                        Remove loop closures whose nodes are already joined by a\n"
			"                        shorter path through odometry and better loop closures (m,\n"
			"                        default %g, 0 disables). Same rule as detect-more's -d.\n"
			"    --keep_user         Never remove user loop closures. Detect-more stores its loop\n"
			"                        closures as user loop closures too, so they are removable\n"
			"                        by default.\n"
			"    --csv file          Write the removed loop closures to this CSV file.\n"
			"    --dry-run           Do not modify the database.\n"
			"\n"
			"Optimizer parameters (e.g. --Optimizer/Strategy 2 --Optimizer/PriorsIgnored false)\n"
			"override those stored in the database.\n"
			"\n%s",
			defaults.maxLinearError,
			defaults.maxAngularError*180.0/M_PI,
			defaults.maxErrorRatio,
			defaults.batchRadius,
			defaults.maxRounds,
			defaults.stallRounds,
			defaults.stallImprovement*100.0f,
			defaults.fullSolveInterval,
			defaults.minGraphDistance,
			Parameters::showUsage());
	exit(1);
}

class PrintProgressState : public ProgressState
{
public:
	PrintProgressState() :
		stamp_(UTimer::now())
	{}
	virtual bool callback(const std::string & msg) const
	{
		if(!msg.empty())
		{
			printf("[%.1fs] %s\n", UTimer::now()-stamp_, msg.c_str());
			fflush(stdout);
		}
		return !isCanceled();
	}
private:
	double stamp_;
};
PrintProgressState progress;

void sighandler(int sig)
{
	printf("\nSignal %d caught, stopping after the current optimization...\n", sig);
	progress.setCanceled(true);
}

int main(int argc, char * argv[])
{
	signal(SIGABRT, &sighandler);
	signal(SIGTERM, &sighandler);
	signal(SIGINT, &sighandler);

	ULogger::setType(ULogger::kTypeConsole);
	ULogger::setLevel(ULogger::kWarning);

	if(argc < 2)
	{
		showUsage();
	}

	loopPruning::Options options;
	bool dryRun = false;
	std::string csvPath;
	for(int i=1; i<argc-1; ++i)
	{
		auto value = [&]() {
			++i;
			if(i >= argc-1)
			{
				showUsage();
			}
			return argv[i];
		};
		if(std::strcmp(argv[i], "--help") == 0)
		{
			showUsage();
		}
		else if(std::strcmp(argv[i], "--max_error") == 0)
		{
			options.maxLinearError = uStr2Float(value());
		}
		else if(std::strcmp(argv[i], "--max_angle") == 0)
		{
			options.maxAngularError = uStr2Float(value())*M_PI/180.0;
		}
		else if(std::strcmp(argv[i], "--max_ratio") == 0)
		{
			options.maxErrorRatio = uStr2Float(value());
		}
		else if(std::strcmp(argv[i], "--batch_radius") == 0)
		{
			options.batchRadius = uStr2Float(value());
		}
		else if(std::strcmp(argv[i], "--max_rounds") == 0)
		{
			options.maxRounds = uStr2Int(value());
		}
		else if(std::strcmp(argv[i], "--stall_rounds") == 0)
		{
			options.stallRounds = uStr2Int(value());
		}
		else if(std::strcmp(argv[i], "--stall_improvement") == 0)
		{
			options.stallImprovement = uStr2Float(value())/100.0f;
		}
		else if(std::strcmp(argv[i], "--full_solve_every") == 0)
		{
			options.fullSolveInterval = uStr2Int(value());
		}
		else if(std::strcmp(argv[i], "--min_graph_distance") == 0)
		{
			options.minGraphDistance = uStr2Float(value());
		}
		else if(std::strcmp(argv[i], "--keep_user") == 0)
		{
			options.includeUserClosures = false;
		}
		else if(std::strcmp(argv[i], "--csv") == 0)
		{
			csvPath = value();
		}
		else if(std::strcmp(argv[i], "--dry-run") == 0)
		{
			dryRun = true;
		}
		else if(std::strncmp(argv[i], "--", 2) == 0 &&
				std::strchr(argv[i], '/') == 0 &&
				std::strncmp(argv[i], "--u", 3) != 0 &&
				std::strncmp(argv[i], "--log", 5) != 0 &&
				std::strcmp(argv[i], "--nolog") != 0 &&
				std::strcmp(argv[i], "--version") != 0)
		{
			// RTAB-Map parameters always contain '/', so this is a typo that
			// would otherwise be ignored silently.
			printf("Unknown option \"%s\". See --help.\n", argv[i]);
			return 1;
		}
	}
	ParametersMap inputParams = Parameters::parseArguments(argc, argv);

	std::string dbPath = argv[argc-1];
	if(!UFile::exists(dbPath))
	{
		printf("Database %s doesn't exist!\n", dbPath.c_str());
		return 1;
	}

	std::unique_ptr<DBDriver> driver(DBDriver::create());
	if(!driver->openConnection(dbPath, false, dryRun))
	{
		printf("Cannot open database %s!\n", dbPath.c_str());
		return 1;
	}

	ParametersMap parameters = driver->getLastParameters();
	for(ParametersMap::iterator iter=inputParams.begin(); iter!=inputParams.end(); ++iter)
	{
		printf("Using parameter \"%s=%s\" from arguments\n", iter->first.c_str(), iter->second.c_str());
	}
	uInsert(parameters, inputParams);
	if(inputParams.find(Parameters::kGTSAMIncremental()) == inputParams.end())
	{
		uInsert(parameters, ParametersPair(Parameters::kGTSAMIncremental(), "true"));
	}

	printf("Database: %s%s\n", dbPath.c_str(), dryRun?" (dry run)":"");
	printf("Greedy pruning: max error %g m, %g deg, ratio %g (0=off), batch radius %g m, max %d rounds, "
			"stop below %g%% improvement over %d rounds\n",
			options.maxLinearError, options.maxAngularError*180.0/M_PI, options.maxErrorRatio,
			options.batchRadius, options.maxRounds, options.stallImprovement*100.0f, options.stallRounds);
	printf("Redundancy min graph distance: %g m (0=off)%s\n", options.minGraphDistance,
			options.includeUserClosures?"":", user loop closures kept");

	UTimer timer;
	std::map<int, Transform> odomPoses;
	driver->getAllOdomPoses(odomPoses, true, false);
	std::multimap<int, Link> links;
	driver->getAllLinks(links, true, true);
	printf("Loaded %d poses and %d links (%.1fs)\n", (int)odomPoses.size(), (int)links.size(), timer.ticks());

	std::unique_ptr<Optimizer> optimizer(Optimizer::create(parameters));
	printf("Optimizer: %s, %d iterations, priors %s, robust %s\n",
			optimizer->type() == Optimizer::kTypeGTSAM?(optimizer->isIncremental()?
					uFormat("GTSAM incremental (full solve every %d)", options.fullSolveInterval).c_str():"GTSAM"):
			optimizer->type() == Optimizer::kTypeG2O?"g2o":
			optimizer->type() == Optimizer::kTypeCeres?"Ceres":
			optimizer->type() == Optimizer::kTypeTORO?"TORO":"other",
			optimizer->iterations(),
			optimizer->priorsIgnored()?"ignored":"used",
			optimizer->isRobust()?"on":"off");
	if(optimizer->isRobust())
	{
		printf("Warning: Optimizer/Robust is on. Switchable constraints hide inconsistent loop closures "
			   "from the residual check; consider --Optimizer/Robust false.\n");
	}

	loopPruning::Result result = loopPruning::prune(*optimizer, odomPoses, links, options, &progress);
	if(!result.ok)
	{
		printf("%s. Database not modified.\n", progress.isCanceled()?"Canceled":"Pruning failed");
		driver->closeConnection(false);
		return 1;
	}

	int inconsistent = 0;
	int redundant = 0;
	for(size_t i=0; i<result.removed.size(); ++i)
	{
		(result.removed[i].reason == loopPruning::kInconsistent ? inconsistent : redundant)++;
	}
	printf("\nOptimized component: %d of %d poses (root %d)\n",
			result.componentPoses, (int)odomPoses.size(), result.rootId);
	printf("Loop closures: %d -> %d (%d inconsistent, %d redundant removed, %d optimizations)\n",
			result.loopsBefore, result.loopsAfter, inconsistent, redundant, result.rounds);
	if(!result.bridges.empty())
	{
		printf("Kept %d inconsistent loop closures because removing them would split the graph:\n", (int)result.bridges.size());
		for(std::set<std::pair<int, int> >::iterator iter=result.bridges.begin(); iter!=result.bridges.end(); ++iter)
		{
			printf("  %d <-> %d\n", iter->first, iter->second);
		}
	}

	if(!result.unresolved.empty())
	{
		std::vector<loopPruning::LoopResidual> unresolved = result.unresolved;
		std::sort(unresolved.begin(), unresolved.end(), [](const loopPruning::LoopResidual & a, const loopPruning::LoopResidual & b) {
			return a.linear > b.linear;
		});
		printf("Stopped with %d inconsistent loop closures kept; the disagreement is likely odometry or priors near them:\n",
				(int)unresolved.size());
		for(size_t i=0; i<unresolved.size() && i<20; ++i)
		{
			printf("  %d -> %d: %.3f m, %.2f deg\n", unresolved[i].link.from(), unresolved[i].link.to(),
					unresolved[i].linear, unresolved[i].angular*180.0/M_PI);
		}
		if(unresolved.size() > 20)
		{
			printf("  ... (%d more%s)\n", (int)unresolved.size()-20, csvPath.empty()?", see --csv":" in the CSV");
		}
	}

	if(!csvPath.empty())
	{
		FILE * file = fopen(csvPath.c_str(), "w");
		if(file)
		{
			fprintf(file, "from,to,type,reason,round,linear_m,angular_deg,linear_ratio,angular_ratio\n");
			for(size_t i=0; i<result.removed.size(); ++i)
			{
				const loopPruning::RemovedLoop & r = result.removed[i];
				fprintf(file, "%d,%d,%s,%s,%d,%f,%f,%f,%f\n",
						r.link.from(), r.link.to(), r.link.typeName().c_str(),
						r.reason == loopPruning::kInconsistent?"inconsistent":"redundant",
						r.round, r.residual.linear, r.residual.angular*180.0/M_PI,
						r.residual.linearRatio, r.residual.angularRatio);
			}
			for(size_t i=0; i<result.unresolved.size(); ++i)
			{
				const loopPruning::LoopResidual & r = result.unresolved[i];
				fprintf(file, "%d,%d,%s,unresolved,0,%f,%f,%f,%f\n",
						r.link.from(), r.link.to(), r.link.typeName().c_str(),
						r.linear, r.angular*180.0/M_PI, r.linearRatio, r.angularRatio);
			}
			fclose(file);
			printf("Removed and unresolved loop closures written to %s\n", csvPath.c_str());
		}
		else
		{
			printf("Cannot write %s\n", csvPath.c_str());
		}
	}

	if(dryRun || result.removed.empty())
	{
		printf("%s\n", dryRun?"Dry run: database not modified.":"Nothing to remove.");
		driver->closeConnection(false);
		return 0;
	}

	for(size_t i=0; i<result.removed.size(); ++i)
	{
		const Link & link = result.removed[i].link;
		driver->removeLink(link.from(), link.to());
		driver->removeLink(link.to(), link.from());
	}
	// Rtabmap::init() reuses the saved optimized poses, which would still
	// reflect the removed links.
	Transform lastLocalizationPose;
	driver->loadOptimizedPoses(&lastLocalizationPose);
	driver->saveOptimizedPoses(result.optimizedPoses, lastLocalizationPose);

	float xMin, yMin, cellSize;
	if(!driver->load2DMap(xMin, yMin, cellSize).empty())
	{
		printf("Warning: the saved global occupancy grid (and mesh, if any) was built with the removed "
			   "loop closures; regenerate it from the database viewer.\n");
	}

	driver->closeConnection(true);
	printf("Removed %d loop closures from the database and saved the new optimized poses.\n", (int)result.removed.size());
	return 0;
}
