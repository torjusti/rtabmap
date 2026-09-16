#include <gtest/gtest.h>
#include <rtabmap/core/Colmap.h>
#include <rtabmap/core/Signature.h>
#include <rtabmap/core/SensorData.h>
#include <rtabmap/core/CameraModel.h>
#include <rtabmap/core/Link.h>
#include <rtabmap/core/Optimizer.h>
#include <rtabmap/core/util3d_transforms.h>
#include <rtabmap/utilite/UDirectory.h>
#include <rtabmap/utilite/UFile.h>
#include <opencv2/core.hpp>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace rtabmap;

namespace {

std::string makeTempDir()
{
	static int n = 0;
	std::ostringstream ss;
	ss << "test_colmap_export_" << ++n;
	const std::string dir = ss.str();
	UDirectory::makeDir(dir);
	return dir;
}

CameraModel makeModel(double k1 = 0.12)
{
	const cv::Size size(64, 48);
	cv::Mat K = (cv::Mat_<double>(3,3) << 50.0, 0, 32.0, 0, 50.0, 24.0, 0, 0, 1);
	cv::Mat D = (cv::Mat_<double>(1,5) << k1, 0.01, 0.0, 0.0, 0.0);
	cv::Mat R = cv::Mat::eye(3, 3, CV_64FC1);
	cv::Mat P = cv::Mat::zeros(3, 4, CV_64FC1);
	K.copyTo(P(cv::Rect(0, 0, 3, 3)));
	return CameraModel("cam", size, K, D, R, P, CameraModel::opticalRotation());
}

Signature makeSignature(int id, const CameraModel & model, bool fast = false)
{
	cv::Mat image(48, 64, CV_8UC3);
	image.setTo(cv::Scalar(id * 20, 80, 160));
	cv::Mat depth(48, 64, CV_32FC1, cv::Scalar(2.0f));
	SensorData data(image, depth, model, id, 0.0);
	Signature s(id, 0, 0, 0.0, "", Transform::getIdentity(), Transform(), data);
	if(fast)
	{
		s.setVelocity(5.0f, 0, 0, 0, 0, 0);
	}

	const int n = 8;
	std::multimap<int, int> words;
	std::vector<cv::KeyPoint> kpts;
	std::vector<cv::Point3f> words3;
	cv::Mat descriptors(n, 8, CV_32F);
	cv::randu(descriptors, 0, 1);
	for(int i=0; i<n; ++i)
	{
		const float y = -0.15f + 0.04f * float(i);
		const cv::Point3f pBase(2.0f, y, 0.05f);
		const cv::Point3f pCam = util3d::transformPoint(pBase, model.localTransform().inverse());
		const float u = float(model.fx() * pCam.x / pCam.z + model.cx());
		const float v = float(model.fy() * pCam.y / pCam.z + model.cy());
		kpts.push_back(cv::KeyPoint(u, v, 3.0f));
		words3.push_back(pBase);
		words.insert(std::make_pair(100 + i, i));
	}
	s.setWords(words, kpts, words3, descriptors);
	return s;
}

std::map<int, std::string> readNonCommentLines(const std::string & path)
{
	std::map<int, std::string> lines;
	std::ifstream in(path.c_str());
	std::string line;
	int i = 0;
	while(std::getline(in, line))
	{
		if(!line.empty() && line[0] != '#')
		{
			lines[i++] = line;
		}
	}
	return lines;
}

std::vector<std::string> split(const std::string & line)
{
	std::istringstream ss(line);
	std::vector<std::string> out;
	std::string tok;
	while(ss >> tok)
	{
		out.push_back(tok);
	}
	return out;
}

} // namespace

TEST(ColmapExport, DefaultsUseOpenCVModel)
{
	EXPECT_EQ(ColmapExportOptions().cameraModel, kColmapOpenCV);
	EXPECT_TRUE(ColmapExportOptions().exportPoints);
}

TEST(ColmapExport, FailedBALeavesPosesUntouched)
{
	CameraModel model = makeModel();
	std::map<int, Signature> signatures;
	signatures.insert(std::make_pair(1, makeSignature(1, model)));
	signatures.insert(std::make_pair(2, makeSignature(2, model)));

	std::map<int, Transform> poses;
	poses[1] = Transform::getIdentity();
	poses[2] = Transform(1, 0, 0, 0, 0, 0);
	const std::map<int, Transform> posesBefore = poses;

	std::multimap<int, Link> links;
	links.insert(std::make_pair(1, Link(1, 2, Link::kNeighbor, Transform(1, 0, 0, 0, 0, 0))));

	ColmapExportOptions options;
	options.exportPoints = true;
	options.exportImages = false;
	options.sbaType = Optimizer::kTypeTORO;
	options.sbaIterations = 5;

	const std::string dir = makeTempDir();
	ColmapExportResult result = exportColmap(poses, links, signatures, 0, ParametersMap(), options, dir);
	EXPECT_FALSE(result.ok);
	EXPECT_EQ(poses.size(), posesBefore.size());
	EXPECT_EQ(poses.at(1), posesBefore.at(1));
	EXPECT_EQ(poses.at(2), posesBefore.at(2));
}

TEST(ColmapExport, WritesOpenCVDistortionAndMatchingPointIds)
{
	CameraModel model = makeModel(0.12);
	std::map<int, Signature> signatures;
	signatures.insert(std::make_pair(1, makeSignature(1, model)));
	signatures.insert(std::make_pair(2, makeSignature(2, model)));

	std::map<int, Transform> poses;
	poses[1] = Transform::getIdentity();
	poses[2] = Transform(0.4f, 0, 0, 0, 0, 0);

	std::multimap<int, Link> links;
	links.insert(std::make_pair(1, Link(1, 2, Link::kNeighbor, Transform(0.4f, 0, 0, 0, 0, 0))));

	ColmapExportOptions options;
	options.cameraModel = kColmapOpenCV;
	options.exportPoints = true;
	options.exportImages = true;
	options.allKeypoints = true;
	options.sbaIterations = 0; // correspondences only
	options.sbaRematchFeatures = false;

	ParametersMap parameters;
	parameters.insert(ParametersPair(Parameters::kVisMinInliers(), "2"));

	const std::string dir = makeTempDir();
	ColmapExportResult result = exportColmap(poses, links, signatures, 0, parameters, options, dir);
	ASSERT_TRUE(result.ok) << result.error;
	EXPECT_EQ(result.camerasExported, 2);

	const std::map<int, std::string> camLines = readNonCommentLines(dir + "/cameras.txt");
	ASSERT_FALSE(camLines.empty());
	const std::vector<std::string> cam = split(camLines.begin()->second);
	ASSERT_GE(cam.size(), 12u);
	EXPECT_EQ(cam[1], "OPENCV");
	EXPECT_NEAR(std::atof(cam[8].c_str()), 0.12, 1e-6);

	const std::map<int, std::string> imgLines = readNonCommentLines(dir + "/images.txt");
	// Two lines per image.
	ASSERT_EQ(imgLines.size(), 4u);

	std::set<int> imagePointIds;
	for(std::map<int, std::string>::const_iterator iter=imgLines.begin(); iter!=imgLines.end(); ++iter)
	{
		if(iter->first % 2 == 0)
		{
			continue; // pose line
		}
		const std::vector<std::string> toks = split(iter->second);
		ASSERT_EQ(toks.size() % 3, 0u);
		for(size_t i=2; i<toks.size(); i+=3)
		{
			const int id = atoi(toks[i].c_str());
			if(id >= 0)
			{
				imagePointIds.insert(id);
			}
		}
	}

	const std::map<int, std::string> ptLines = readNonCommentLines(dir + "/points3D.txt");
	std::set<int> filePointIds;
	for(std::map<int, std::string>::const_iterator iter=ptLines.begin(); iter!=ptLines.end(); ++iter)
	{
		const std::vector<std::string> toks = split(iter->second);
		ASSERT_GE(toks.size(), 8u);
		filePointIds.insert(atoi(toks[0].c_str()));
	}

	ASSERT_FALSE(filePointIds.empty());
	EXPECT_GT(result.pointsExported, 0);
	EXPECT_EQ(imagePointIds, filePointIds);
	EXPECT_EQ(*filePointIds.begin(), 1);
	EXPECT_EQ(result.pointsExported, (int)filePointIds.size());
}

TEST(ColmapExport, DroppedCameraDoesNotLeaveDanglingPointIds)
{
	CameraModel model = makeModel();
	std::map<int, Signature> signatures;
	signatures.insert(std::make_pair(1, makeSignature(1, model, false)));
	signatures.insert(std::make_pair(2, makeSignature(2, model, true))); // fast, will be filtered
	signatures.insert(std::make_pair(3, makeSignature(3, model, false)));

	std::map<int, Transform> poses;
	poses[1] = Transform::getIdentity();
	poses[2] = Transform(0.3f, 0, 0, 0, 0, 0);
	poses[3] = Transform(0.6f, 0, 0, 0, 0, 0);

	std::multimap<int, Link> links;
	links.insert(std::make_pair(1, Link(1, 2, Link::kNeighbor, Transform(0.3f, 0, 0, 0, 0, 0))));
	links.insert(std::make_pair(2, Link(2, 3, Link::kNeighbor, Transform(0.3f, 0, 0, 0, 0, 0))));

	ColmapExportOptions options;
	options.exportPoints = true;
	options.exportImages = false;
	options.allKeypoints = false;
	options.sbaIterations = 0;
	options.sbaRematchFeatures = false;
	options.maxLinearVelocity = 1.0; // signature 2 is at 5 m/s

	ParametersMap parameters;
	parameters.insert(ParametersPair(Parameters::kVisMinInliers(), "2"));

	const std::string dir = makeTempDir();
	ColmapExportResult result = exportColmap(poses, links, signatures, 0, parameters, options, dir);
	ASSERT_TRUE(result.ok) << result.error;
	EXPECT_EQ(result.camerasExported, 2);

	std::set<int> imagePointIds;
	const std::map<int, std::string> imgLines = readNonCommentLines(dir + "/images.txt");
	for(std::map<int, std::string>::const_iterator iter=imgLines.begin(); iter!=imgLines.end(); ++iter)
	{
		if(iter->first % 2 == 0)
		{
			continue;
		}
		const std::vector<std::string> toks = split(iter->second);
		for(size_t i=2; i<toks.size(); i+=3)
		{
			const int id = atoi(toks[i].c_str());
			if(id >= 0)
			{
				imagePointIds.insert(id);
			}
		}
	}

	std::set<int> filePointIds;
	const std::map<int, std::string> ptLines = readNonCommentLines(dir + "/points3D.txt");
	for(std::map<int, std::string>::const_iterator iter=ptLines.begin(); iter!=ptLines.end(); ++iter)
	{
		filePointIds.insert(atoi(split(iter->second)[0].c_str()));
	}
	EXPECT_EQ(imagePointIds, filePointIds);
}

TEST(ColmapExport, EmptyInputFailsWithoutTouchingPoses)
{
	std::map<int, Transform> poses;
	std::multimap<int, Link> links;
	std::map<int, Signature> signatures;
	ColmapExportResult result = exportColmap(
			poses, links, signatures, 0, ParametersMap(), ColmapExportOptions(), "test_colmap_empty");
	EXPECT_FALSE(result.ok);
}
