/*
 * Headless smoke test of the parallelized ExportCloudsDialog::getClouds(),
 * exercising the same code path as DatabaseViewer's "Export clouds" and
 * "Edit > View 3D map" (db driver + empty caches).
 *
 * Usage: QT_QPA_PLATFORM=offscreen gui_test database.db [maxNodes=100]
 */

#include <rtabmap/gui/ExportCloudsDialog.h>
#include <rtabmap/core/DBDriver.h>
#include <rtabmap/utilite/ULogger.h>
#include <rtabmap/utilite/UTimer.h>

#include <QApplication>
#include <QTimer>
#include <QDialog>
#include <QDir>

using namespace rtabmap;

int main(int argc, char ** argv)
{
	QApplication app(argc, argv);
	if(argc < 2)
	{
		printf("Usage: gui_test database.db [maxNodes=100]\n");
		return 1;
	}
	int maxNodes = argc > 2 ? atoi(argv[2]) : 100;

	ULogger::setType(ULogger::kTypeConsole);
	ULogger::setLevel(ULogger::kError);

	DBDriver * driver = DBDriver::create();
	if(!driver->openConnection(argv[1]))
	{
		printf("Failed to open %s\n", argv[1]);
		return 1;
	}

	std::map<int, Transform> allPoses;
	driver->getAllOdomPoses(allPoses, true, true);
	std::map<int, Transform> poses;
	for(std::map<int, Transform>::iterator iter=allPoses.begin();
		iter!=allPoses.end() && (int)poses.size() < maxNodes; ++iter)
	{
		poses.insert(*iter);
	}
	printf("Using %d/%d poses\n", (int)poses.size(), (int)allPoses.size());

	ExportCloudsDialog dialog;
	dialog.setDBDriver(driver);
	dialog.restoreDefaults();

	// getExportedClouds() shows the dialog with exec(); accept it as soon as
	// the event loop runs.
	QTimer::singleShot(0, &dialog, &QDialog::accept);

	std::map<int, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> clouds;
	std::map<int, pcl::PolygonMesh::Ptr> meshes;
	std::map<int, pcl::TextureMesh::Ptr> textureMeshes;
	std::vector<std::map<int, pcl::PointXY> > textureVertexToPixels;

	UTimer timer;
	bool ok = dialog.getExportedClouds(
			poses,
			std::multimap<int, Link>(),
			std::map<int, int>(),
			QMap<int, Signature>(),
			std::map<int, std::pair<pcl::PointCloud<pcl::PointXYZRGB>::Ptr, pcl::IndicesPtr> >(),
			std::map<int, LaserScan>(),
			QDir::tempPath(),
			ParametersMap(),
			clouds,
			meshes,
			textureMeshes,
			textureVertexToPixels);
	double elapsed = timer.ticks();

	long long totalPoints = 0;
	for(std::map<int, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr>::iterator iter=clouds.begin(); iter!=clouds.end(); ++iter)
	{
		totalPoints += iter->second->size();
	}
	printf("getExportedClouds: %s, %.2fs, %d cloud(s), %lld total points\n",
			ok?"OK":"FAILED", elapsed, (int)clouds.size(), totalPoints);

	driver->closeConnection();
	delete driver;
	return ok ? 0 : 1;
}
