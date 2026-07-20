/*
 * Benchmarks the stages of the DatabaseViewer "Export clouds" pipeline
 * (ExportCloudsDialog::getClouds) on a real database, one stage at a time:
 *
 *   1. SQLite blob read           (DBDriver::getNodeData)
 *   2. RGB decompression          (JPEG decode)
 *   3. Depth decompression        (PNG or RVL decode, whatever is in the db)
 *   4. Depth re-coded as RVL      (decode time comparison vs PNG)
 *   5. Back-projection            (util3d::cloudRGBFromSensorData)
 *   6. Normal estimation          (util3d::computeNormals, K=20 like the dialog default)
 *   7. Per-cloud voxel filtering  (util3d::voxelize)
 *
 * Usage: bench_export database.db [maxNodes=50] [decimation=1] [voxel=0.01]
 */

#include <rtabmap/core/DBDriver.h>
#include <rtabmap/core/SensorData.h>
#include <rtabmap/core/Compression.h>
#include <rtabmap/core/util3d.h>
#include <rtabmap/core/util3d_surface.h>
#include <rtabmap/core/util3d_filtering.h>
#include <rtabmap/utilite/UTimer.h>
#include <rtabmap/utilite/ULogger.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <cstdio>
#include <string>
#include <vector>

using namespace rtabmap;

int main(int argc, char ** argv)
{
	if(argc < 2)
	{
		printf("Usage: bench_export database.db [maxNodes=50] [decimation=1] [voxel=0.01]\n");
		return 1;
	}
	std::string dbPath = argv[1];
	int maxNodes = argc > 2 ? atoi(argv[2]) : 50;
	int decimation = argc > 3 ? atoi(argv[3]) : 1;
	float voxel = argc > 4 ? (float)atof(argv[4]) : 0.01f;

	ULogger::setType(ULogger::kTypeConsole);
	ULogger::setLevel(ULogger::kError);

	DBDriver * driver = DBDriver::create();
	if(!driver->openConnection(dbPath))
	{
		printf("Failed to open %s\n", dbPath.c_str());
		return 1;
	}

	std::set<int> ids;
	driver->getAllNodeIds(ids, false, true, true);
	printf("Database: %s (%d nodes, benchmarking first %d, decimation=%d, voxel=%.3f)\n\n",
			dbPath.c_str(), (int)ids.size(), maxNodes, decimation, voxel);

	double tDbRead = 0, tRgbDecode = 0, tDepthDecode = 0, tRvlDecode = 0,
		   tProject = 0, tNormals = 0, tVoxel = 0;
	size_t bytesPng = 0, bytesRvl = 0;
	long long totalPoints = 0, totalVoxelPoints = 0;
	int n = 0;
	bool depthIsRvl = false;

	UTimer total;
	for(std::set<int>::iterator iter = ids.begin(); iter != ids.end() && n < maxNodes; ++iter)
	{
		UTimer t;
		SensorData data;
		driver->getNodeData(*iter, data, true /*images*/, false, false, false);
		tDbRead += t.ticks();

		if(data.imageCompressed().empty() || data.depthOrRightCompressed().empty())
		{
			continue;
		}

		cv::Mat rgb = uncompressImage(data.imageCompressed());
		tRgbDecode += t.ticks();

		cv::Mat depth = uncompressImage(data.depthOrRightCompressed());
		tDepthDecode += t.ticks();

		if(rgb.empty() || depth.empty() || depth.type() == CV_8UC3 || depth.type() == CV_8UC1)
		{
			// stereo db, right image instead of depth: skip depth codec comparison
			continue;
		}

		if(n == 0)
		{
			printf("rgb: %dx%d, depth: %dx%d type=%s stored as %s\n\n",
					rgb.cols, rgb.rows, depth.cols, depth.rows,
					depth.type() == CV_16UC1 ? "16UC1" : depth.type() == CV_32FC1 ? "32FC1" : "?",
					compressedDepthFormat(data.depthOrRightCompressed()).c_str());
		}

		// Compare depth codec: re-encode to RVL (if 16UC1) and decode
		if(depth.type() == CV_16UC1)
		{
			depthIsRvl = compressedDepthFormat(data.depthOrRightCompressed()) == ".rvl";
			bytesPng += data.depthOrRightCompressed().total();
			cv::Mat rvlBytes = compressImage2(depth, ".rvl");
			bytesRvl += rvlBytes.total();
			t.ticks(); // don't count the re-encode
			cv::Mat depthRvl = uncompressImage(rvlBytes);
			tRvlDecode += t.ticks();
			UASSERT(cv::countNonZero(depthRvl != depth) == 0);
		}

		data.setRGBDImage(rgb, depth, data.cameraModels());
		t.ticks();

		std::vector<int> validIndices;
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = util3d::cloudRGBFromSensorData(
				data, decimation, 0.0f, 0.0f, &validIndices);
		tProject += t.ticks();
		totalPoints += validIndices.size();

		pcl::IndicesPtr indices(new std::vector<int>(validIndices));
		Eigen::Vector3f viewPoint(0, 0, 0);
		if(!data.cameraModels().empty() && !data.cameraModels()[0].localTransform().isNull())
		{
			viewPoint[0] = data.cameraModels()[0].localTransform().x();
			viewPoint[1] = data.cameraModels()[0].localTransform().y();
			viewPoint[2] = data.cameraModels()[0].localTransform().z();
		}
		pcl::PointCloud<pcl::Normal>::Ptr normals =
				util3d::computeNormals(cloud, indices, 20, 0.0f, viewPoint);
		tNormals += t.ticks();

		if(voxel > 0.0f)
		{
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr voxelized =
					util3d::voxelize(cloud, indices, voxel);
			tVoxel += t.ticks();
			totalVoxelPoints += voxelized->size();
		}

		++n;
	}
	double tTotal = total.ticks();
	driver->closeConnection();
	delete driver;

	if(n == 0)
	{
		printf("No usable RGB-D nodes found.\n");
		return 1;
	}

	double sum = tDbRead + tRgbDecode + tDepthDecode + tProject + tNormals + tVoxel;
	printf("Processed %d nodes, %.1fs wall (%.0f ms/node), avg %lld pts/cloud (%lld after voxel)\n\n",
			n, tTotal, 1000.0 * tTotal / n, totalPoints / n, totalVoxelPoints / n);
	printf("%-34s %10s %12s %8s\n", "stage", "total (s)", "ms/node", "share");
	const char * depthLabel = depthIsRvl ? "depth decode (RVL, as stored)" : "depth decode (PNG, as stored)";
	struct { const char * name; double t; } rows[] = {
		{"db read (sqlite, compressed)", tDbRead},
		{"rgb decode (jpeg)", tRgbDecode},
		{depthLabel, tDepthDecode},
		{"back-projection to cloud", tProject},
		{"normals (K=20, kd-tree)", tNormals},
		{"voxel filter (per-cloud)", tVoxel},
	};
	for(auto & r : rows)
	{
		printf("%-34s %10.2f %12.1f %7.1f%%\n", r.name, r.t, 1000.0 * r.t / n, 100.0 * r.t / sum);
	}
	if(bytesPng > 0)
	{
		printf("\nDepth codec comparison (not counted in pipeline shares above):\n");
		printf("  RVL decode of same depth images:  %.2fs total, %.2f ms/node (vs %.2f ms/node as stored)\n",
				tRvlDecode, 1000.0 * tRvlDecode / n, 1000.0 * tDepthDecode / n);
		printf("  compressed size: stored=%.1f KB/node, rvl=%.1f KB/node (%.2fx)\n",
				bytesPng / 1024.0 / n, bytesRvl / 1024.0 / n, (double)bytesRvl / (double)bytesPng);
	}
	else
	{
		printf("\nDepth is 32FC1 (float): always PNG-coded, RVL not applicable to this db.\n");
	}

	// --- Parallelization experiment ---------------------------------------
	// Serial db read of compressed blobs, then process nodes concurrently
	// (decode + project + normals + voxel), like a parallelized getClouds().
	{
		DBDriver * drv = DBDriver::create();
		drv->openConnection(dbPath);
		std::vector<SensorData> datas;
		datas.reserve(maxNodes);
		UTimer t;
		for(std::set<int>::iterator iter = ids.begin(); iter != ids.end() && (int)datas.size() < maxNodes; ++iter)
		{
			SensorData d;
			drv->getNodeData(*iter, d, true, false, false, false);
			if(!d.imageCompressed().empty() && !d.depthOrRightCompressed().empty())
			{
				datas.push_back(d);
			}
		}
		double tRead = t.ticks();

		#pragma omp parallel for schedule(dynamic)
		for(int i = 0; i < (int)datas.size(); ++i)
		{
			SensorData & d = datas[i];
			cv::Mat rgb = uncompressImage(d.imageCompressed());
			cv::Mat depth = uncompressImage(d.depthOrRightCompressed());
			if(rgb.empty() || depth.empty() || depth.type() == CV_8UC3 || depth.type() == CV_8UC1)
			{
				continue;
			}
			d.setRGBDImage(rgb, depth, d.cameraModels());
			std::vector<int> validIndices;
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = util3d::cloudRGBFromSensorData(
					d, decimation, 0.0f, 0.0f, &validIndices);
			pcl::IndicesPtr indices(new std::vector<int>(validIndices));
			Eigen::Vector3f viewPoint(0, 0, 0);
			if(!d.cameraModels().empty() && !d.cameraModels()[0].localTransform().isNull())
			{
				viewPoint[0] = d.cameraModels()[0].localTransform().x();
				viewPoint[1] = d.cameraModels()[0].localTransform().y();
				viewPoint[2] = d.cameraModels()[0].localTransform().z();
			}
			pcl::PointCloud<pcl::Normal>::Ptr normals =
					util3d::computeNormals(cloud, indices, 20, 0.0f, viewPoint);
			if(voxel > 0.0f)
			{
				util3d::voxelize(cloud, indices, voxel);
			}
		}
		double tPar = t.ticks();
		drv->closeConnection();
		delete drv;

		printf("\nParallelization experiment (%d nodes, omp over nodes):\n", (int)datas.size());
		printf("  serial db read of compressed blobs: %.2fs\n", tRead);
		printf("  parallel decode+project+normals+voxel: %.2fs wall (%.1f ms/node)\n",
				tPar, 1000.0 * tPar / datas.size());
		printf("  serial equivalent above was: %.2fs (%.1f ms/node) -> speedup %.1fx\n",
				sum - tDbRead, 1000.0 * (sum - tDbRead) / n, (sum - tDbRead) / tPar);
	}
	return 0;
}
