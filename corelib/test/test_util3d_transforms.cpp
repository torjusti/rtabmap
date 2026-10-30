#include "gtest/gtest.h"
#include "rtabmap/core/util3d_transforms.h"
#include "rtabmap/utilite/UException.h"

using namespace rtabmap;

TEST(Util3dTransformsTest, TransformLaserScanBasicXYZTransform)
{
    // Create a simple 3D laser scan with 3 points
    cv::Mat scanData(1, 3, CV_32FC3); // 3D points (X,Y,Z)
    scanData.at<cv::Vec3f>(0,0) = cv::Vec3f(1.0f, 0.0f, 0.0f);
    scanData.at<cv::Vec3f>(0,1) = cv::Vec3f(0.0f, 1.0f, 0.0f);
    scanData.at<cv::Vec3f>(0,2) = cv::Vec3f(0.0f, 0.0f, 1.0f);

    LaserScan scan(scanData, 0, 10.0f, LaserScan::kXYZ, Transform::getIdentity());

    // Create a transform: translate (1, 2, 3) and rotate 90 degrees about Z
    Transform transform(0, -1, 0, 1,
                        1,  0, 0, 2,
                        0,  0, 1, 3);

    // Apply transform
    LaserScan result = util3d::transformLaserScan(scan, transform);

    // Expected points after transformation
    std::vector<Eigen::Vector3f> expectedPoints = {
        Eigen::Vector3f(1.0f, 3.0f, 3.0f), // Rotated (1,0,0) + translated
        Eigen::Vector3f(0.0f, 2.0f, 3.0f), // Rotated (0,1,0) + translated
        Eigen::Vector3f(1.0f, 2.0f, 4.0f)  // Rotated (0,0,1) + translated
    };

    ASSERT_EQ(result.size(), 3);
    for (int i = 0; i < 3; ++i)
    {
        EXPECT_NEAR(result.field(i, 0), expectedPoints[i].x(), 1e-5f);
        EXPECT_NEAR(result.field(i, 1), expectedPoints[i].y(), 1e-5f);
        EXPECT_NEAR(result.field(i, 2), expectedPoints[i].z(), 1e-5f);
    }
}

TEST(Util3dTransformsTest, TransformLaserScanXYZNormalTransform)
{
    // One point: position (1, 0, 0), normal (0, 1, 0)
    cv::Mat scanData(1, 1, CV_32FC(6)); // XYZ + normal_x, normal_y, normal_z
    float* ptr = scanData.ptr<float>(0, 0);
    ptr[0] = 1.0f; // x
    ptr[1] = 0.0f; // y
    ptr[2] = 0.0f; // z
    ptr[3] = 0.0f; // normal_x
    ptr[4] = 1.0f; // normal_y
    ptr[5] = 0.0f; // normal_z

    LaserScan scan(scanData, 0, 10.0f, LaserScan::kXYZNormal, Transform::getIdentity());

    // Transform: rotate 90 deg around Z axis + translate (0, 1, 0)
    Transform transform(0, -1, 0, 1,
                        1,  0, 0, 2,
                        0,  0, 1, 3);

    // Apply transform
    LaserScan result = util3d::transformLaserScan(scan, transform);

    ASSERT_EQ(result.size(), 1);
    const float* out = result.data().ptr<float>(0, 0);

    // Expected:
    // Point (1,0,0) → rotated to (0,1,0) + translated → (1,2,3)
    // Normal (0,1,0) → rotated to (-1,0,0) (normals are rotated, not translated)
    EXPECT_NEAR(out[0], 1.0f, 1e-5);  // x
    EXPECT_NEAR(out[1], 3.0f, 1e-5);  // y
    EXPECT_NEAR(out[2], 3.0f, 1e-5);  // z

    EXPECT_NEAR(out[3], -1.0f, 1e-5); // normal_x
    EXPECT_NEAR(out[4],  0.0f, 1e-5); // normal_y
    EXPECT_NEAR(out[5],  0.0f, 1e-5); // normal_z
}

TEST(Util3dTransformsTest, TransformPointCloudXYZ)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointXYZ pt(1.0f, 2.0f, 3.0f);
    cloud->push_back(pt);

    Transform tf(1, 0, 0, 1,
                 0, 1, 0, 2,
                 0, 0, 1, 3); // Translate (1,2,3)

    auto result = util3d::transformPointCloud(cloud, tf);

    ASSERT_EQ(result->size(), 1);
    EXPECT_FLOAT_EQ(result->at(0).x, 2.0f);
    EXPECT_FLOAT_EQ(result->at(0).y, 4.0f);
    EXPECT_FLOAT_EQ(result->at(0).z, 6.0f);

    // test indices
    pcl::PointXYZ pt2(4.0f, 5.0f, 6.0f);
    cloud->push_back(pt2);
    pcl::IndicesPtr indices(new std::vector<int>(1,1)); // will transform only second point
    result = util3d::transformPointCloud(cloud, indices, tf);
    ASSERT_EQ(result->size(), 1);
    EXPECT_FLOAT_EQ(result->at(0).x, 5.0f);
    EXPECT_FLOAT_EQ(result->at(0).y, 7.0f);
    EXPECT_FLOAT_EQ(result->at(0).z, 9.0f);
}

TEST(Util3dTransformsTest, TransformPointCloudXYZI)
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointXYZI pt;
    pt.x = 1; pt.y = 1; pt.z = 1; pt.intensity = 42.0f;
    cloud->push_back(pt);

    Transform tf(1, 0, 0, 0,
                 0, 1, 0, 0,
                 0, 0, 1, 1); // Translate (0,0,1)

    auto result = util3d::transformPointCloud(cloud, tf);
    ASSERT_EQ(result->size(), 1);
    EXPECT_FLOAT_EQ(result->at(0).z, 2.0f);
    EXPECT_FLOAT_EQ(result->at(0).intensity, 42.0f);
    
    // test indices
    pt.intensity = 52.0f;
    cloud->push_back(pt);
    pcl::IndicesPtr indices(new std::vector<int>(1,1)); // will transform only second point
    result = util3d::transformPointCloud(cloud, indices, tf);
    ASSERT_EQ(result->size(), 1);
    EXPECT_FLOAT_EQ(result->at(0).intensity, 52.0f);
}

TEST(Util3dTransformsTest, TransformPointCloudXYZRGB)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointXYZRGB pt;
    pt.x = 0; pt.y = 0; pt.z = 0;
    pt.r = 255; pt.g = 100; pt.b = 50;
    cloud->push_back(pt);

    Transform tf(1, 0, 0, 0,
                 0, 1, 0, 0,
                 0, 0, 1, 0);

    auto result = util3d::transformPointCloud(cloud, tf);
    ASSERT_EQ(result->size(), 1);
    EXPECT_EQ(result->at(0).r, 255);
    EXPECT_EQ(result->at(0).g, 100);
    EXPECT_EQ(result->at(0).b, 50);

    // test indices
    pt.r = 250; pt.g = 105; pt.b = 55;
    cloud->push_back(pt);
    pcl::IndicesPtr indices(new std::vector<int>(1,1)); // will transform only second point
    result = util3d::transformPointCloud(cloud, indices, tf);
    ASSERT_EQ(result->size(), 1);
    EXPECT_EQ(result->at(0).r, 250);
    EXPECT_EQ(result->at(0).g, 105);
    EXPECT_EQ(result->at(0).b, 55);
}

TEST(Util3dTransformsTest, TransformPointCloudPointNormal)
{
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud(new pcl::PointCloud<pcl::PointNormal>);
    pcl::PointNormal pt;
    pt.x = 1; pt.y = 0; pt.z = 0;
    pt.normal_x = 0; pt.normal_y = 1; pt.normal_z = 0;
    cloud->push_back(pt);

    Transform tf(0, -1, 0, 0,
                 1,  0, 0, 0,
                 0,  0, 1, 0); // 90 deg rotation around Z

    auto result = util3d::transformPointCloud(cloud, tf);
    ASSERT_EQ(result->size(), 1);
    EXPECT_NEAR(result->at(0).x, 0.0f, 1e-6);
    EXPECT_NEAR(result->at(0).y, 1.0f, 1e-6);
    EXPECT_NEAR(result->at(0).normal_x, -1.0f, 1e-6);
    EXPECT_NEAR(result->at(0).normal_y, 0.0f, 1e-6);

    // test indices
    pt.x = 0; pt.y = 1; pt.z = 0;
    pt.normal_x = 1; pt.normal_y = 0; pt.normal_z = 0;
    cloud->push_back(pt);
    pcl::IndicesPtr indices(new std::vector<int>(1,1)); // will transform only second point
    result = util3d::transformPointCloud(cloud, indices, tf);
    ASSERT_EQ(result->size(), 1);
    EXPECT_NEAR(result->at(0).x, -1.0f, 1e-6);
    EXPECT_NEAR(result->at(0).y, 0.0f, 1e-6);
    EXPECT_NEAR(result->at(0).normal_x, 0.0f, 1e-6);
    EXPECT_NEAR(result->at(0).normal_y, 1.0f, 1e-6);
}

TEST(Util3dTransformsTest, TransformPointCloudXYZINormal)
{
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZINormal>);
    pcl::PointXYZINormal pt;
    pt.x = 0; pt.y = 1; pt.z = 0; pt.intensity = 42.0f;
    pt.normal_x = 1; pt.normal_y = 0; pt.normal_z = 0;
    cloud->push_back(pt);

    Transform tf(0, -1, 0, 0,
                 1,  0, 0, 0,
                 0,  0, 1, 0);

    auto result = util3d::transformPointCloud(cloud, tf);
    ASSERT_EQ(result->size(), 1);
    EXPECT_FLOAT_EQ(result->at(0).intensity, 42.0f);
    EXPECT_NEAR(result->at(0).normal_x, 0.0f, 1e-6);
    EXPECT_NEAR(result->at(0).normal_y, 1.0f, 1e-6);

    // test indices
    pt.intensity = 52.0f;
    cloud->push_back(pt);
    pcl::IndicesPtr indices(new std::vector<int>(1,1)); // will transform only second point
    result = util3d::transformPointCloud(cloud, indices, tf);
    ASSERT_EQ(result->size(), 1);
    EXPECT_FLOAT_EQ(result->at(0).intensity, 52.0f);
}

TEST(Util3dTransformsTest, TransformPointCloudXYZRGBNormal)
{
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::PointXYZRGBNormal pt;
    pt.x = 0; pt.y = 1; pt.z = 0;
    pt.r = 10; pt.g = 20; pt.b = 30;
    pt.normal_x = 1; pt.normal_y = 0; pt.normal_z = 0;
    cloud->push_back(pt);

    Transform tf(0, -1, 0, 0,
                 1,  0, 0, 0,
                 0,  0, 1, 0);

    auto result = util3d::transformPointCloud(cloud, tf);
    ASSERT_EQ(result->size(), 1);
    EXPECT_EQ(result->at(0).r, 10);
    EXPECT_EQ(result->at(0).g, 20);
    EXPECT_EQ(result->at(0).b, 30);
    EXPECT_NEAR(result->at(0).normal_x, 0.0f, 1e-6);
    EXPECT_NEAR(result->at(0).normal_y, 1.0f, 1e-6);

    // test indices
    pt.r = 250; pt.g = 105; pt.b = 55;
    cloud->push_back(pt);
    pcl::IndicesPtr indices(new std::vector<int>(1,1)); // will transform only second point
    result = util3d::transformPointCloud(cloud, indices, tf);
    ASSERT_EQ(result->size(), 1);
    EXPECT_EQ(result->at(0).r, 250);
    EXPECT_EQ(result->at(0).g, 105);
    EXPECT_EQ(result->at(0).b, 55);
}

TEST(Util3dTransformsTest, TransformCVPointF)
{
    cv::Point3f pt(1.0f, 2.0f, 3.0f);

    Transform tf(1, 0, 0, 1,
                 0, 1, 0, 2,
                 0, 0, 1, 3); // Translate (1,2,3)

    auto result = util3d::transformPoint(pt, tf);

    EXPECT_FLOAT_EQ(result.x, 2.0f);
    EXPECT_FLOAT_EQ(result.y, 4.0f);
    EXPECT_FLOAT_EQ(result.z, 6.0f);
}

TEST(Util3dTransformsTest, TransformCVPointD)
{
    cv::Point3d pt(1.0f, 2.0f, 3.0f);

    Transform tf(1, 0, 0, 1,
                 0, 1, 0, 2,
                 0, 0, 1, 3); // Translate (1,2,3)

    auto result = util3d::transformPoint(pt, tf);

    EXPECT_FLOAT_EQ(result.x, 2.0);
    EXPECT_FLOAT_EQ(result.y, 4.0);
    EXPECT_FLOAT_EQ(result.z, 6.0);
}

TEST(Util3dTransformsTest, TransformPointXYZ)
{
    pcl::PointXYZ pt(1.0f, 2.0f, 3.0f);

    Transform tf(1, 0, 0, 1,
                 0, 1, 0, 2,
                 0, 0, 1, 3); // Translate (1,2,3)

    auto result = util3d::transformPoint(pt, tf);

    EXPECT_FLOAT_EQ(result.x, 2.0f);
    EXPECT_FLOAT_EQ(result.y, 4.0f);
    EXPECT_FLOAT_EQ(result.z, 6.0f);
}

TEST(Util3dTransformsTest, TransformPointXYZI)
{
    pcl::PointXYZI pt;
    pt.x = 1; pt.y = 1; pt.z = 1; pt.intensity = 42.0f;

    Transform tf(1, 0, 0, 0,
                 0, 1, 0, 0,
                 0, 0, 1, 1); // Translate (0,0,1)

    auto result = util3d::transformPoint(pt, tf);
    EXPECT_FLOAT_EQ(result.z, 2.0f);
    EXPECT_FLOAT_EQ(result.intensity, 42.0f);
}

TEST(Util3dTransformsTest, TransformPointXYZRGB)
{
    pcl::PointXYZRGB pt;
    pt.x = 0; pt.y = 0; pt.z = 0;
    pt.r = 255; pt.g = 100; pt.b = 50;

    Transform tf(1, 0, 0, 0,
                 0, 1, 0, 0,
                 0, 0, 1, 0);

    auto result = util3d::transformPoint(pt, tf);
    EXPECT_EQ(result.r, 255);
    EXPECT_EQ(result.g, 100);
    EXPECT_EQ(result.b, 50);
}

TEST(Util3dTransformsTest, TransformPointPointNormal)
{
    pcl::PointNormal pt;
    pt.x = 1; pt.y = 0; pt.z = 0;
    pt.normal_x = 0; pt.normal_y = 1; pt.normal_z = 0;

    Transform tf(0, -1, 0, 0,
                 1,  0, 0, 0,
                 0,  0, 1, 0); // 90 deg rotation around Z

    auto result = util3d::transformPoint(pt, tf);
    EXPECT_NEAR(result.x, 0.0f, 1e-6);
    EXPECT_NEAR(result.y, 1.0f, 1e-6);
    EXPECT_NEAR(result.normal_x, -1.0f, 1e-6);
    EXPECT_NEAR(result.normal_y, 0.0f, 1e-6);
}

TEST(Util3dTransformsTest, TransformPointXYZINormal)
{
    pcl::PointXYZINormal pt;
    pt.x = 0; pt.y = 1; pt.z = 0; pt.intensity = 42.0f;
    pt.normal_x = 1; pt.normal_y = 0; pt.normal_z = 0;

    Transform tf(0, -1, 0, 0,
                 1,  0, 0, 0,
                 0,  0, 1, 0);

    auto result = util3d::transformPoint(pt, tf);
    EXPECT_FLOAT_EQ(result.intensity, 42.0f);
    EXPECT_NEAR(result.normal_x, 0.0f, 1e-6);
    EXPECT_NEAR(result.normal_y, 1.0f, 1e-6);
}

TEST(Util3dTransformsTest, TransformPointXYZRGBNormal)
{
    pcl::PointXYZRGBNormal pt;
    pt.x = 0; pt.y = 1; pt.z = 0;
    pt.r = 10; pt.g = 20; pt.b = 30;
    pt.normal_x = 1; pt.normal_y = 0; pt.normal_z = 0;

    Transform tf(0, -1, 0, 0,
                 1,  0, 0, 0,
                 0,  0, 1, 0);

    auto result = util3d::transformPoint(pt, tf);
    EXPECT_EQ(result.r, 10);
    EXPECT_EQ(result.g, 20);
    EXPECT_EQ(result.b, 30);
    EXPECT_NEAR(result.normal_x, 0.0f, 1e-6);
    EXPECT_NEAR(result.normal_y, 1.0f, 1e-6);
}
// PCL 1.12 transformPointCloud()/transformPointCloudWithNormals() divide by the
// cloud width, which crashed (SIGFPE) on empty clouds (issue #1799).
TEST(Util3dTransformsTest, TransformEmptyPointClouds)
{
    const Transform t(1, 2, 3, 0.1f, 0.2f, 0.3f);
    EXPECT_TRUE(util3d::transformPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>), t)->empty());
    EXPECT_TRUE(util3d::transformPointCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>), t)->empty());
    EXPECT_TRUE(util3d::transformPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>), t)->empty());
    EXPECT_TRUE(util3d::transformPointCloud(pcl::PointCloud<pcl::PointNormal>::Ptr(new pcl::PointCloud<pcl::PointNormal>), t)->empty());
    EXPECT_TRUE(util3d::transformPointCloud(pcl::PointCloud<pcl::PointXYZINormal>::Ptr(new pcl::PointCloud<pcl::PointXYZINormal>), t)->empty());
    EXPECT_TRUE(util3d::transformPointCloud(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>), t)->empty());
}

// ---------------------------------------------------------------------------
// transformAndMergePointClouds
// ---------------------------------------------------------------------------

#include <cmath>
#include <limits>
#include <random>

namespace {

pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr randomCloud(std::mt19937 & rng, size_t size)
{
    std::uniform_real_distribution<float> coord(-20.0f, 20.0f);
    std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
    std::uniform_int_distribution<int> byte(0, 255);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    cloud->resize(size);
    for(size_t i=0; i<size; ++i)
    {
        pcl::PointXYZRGBNormal & p = cloud->points[i];
        p.x = coord(rng); p.y = coord(rng); p.z = coord(rng);
        p.normal_x = unit(rng); p.normal_y = unit(rng); p.normal_z = unit(rng);
        p.curvature = unit(rng);
        p.r = byte(rng); p.g = byte(rng); p.b = byte(rng); p.a = byte(rng);
    }
    return cloud;
}

void expectSamePoints(const pcl::PointCloud<pcl::PointXYZRGBNormal> & expected,
                      const pcl::PointCloud<pcl::PointXYZRGBNormal> & actual)
{
    ASSERT_EQ(expected.size(), actual.size());
    for(size_t i=0; i<expected.size(); ++i)
    {
        const pcl::PointXYZRGBNormal & e = expected.points[i];
        const pcl::PointXYZRGBNormal & a = actual.points[i];
        EXPECT_NEAR(e.x, a.x, 1e-4f) << "point " << i;
        EXPECT_NEAR(e.y, a.y, 1e-4f) << "point " << i;
        EXPECT_NEAR(e.z, a.z, 1e-4f) << "point " << i;
        EXPECT_NEAR(e.normal_x, a.normal_x, 1e-5f) << "point " << i;
        EXPECT_NEAR(e.normal_y, a.normal_y, 1e-5f) << "point " << i;
        EXPECT_NEAR(e.normal_z, a.normal_z, 1e-5f) << "point " << i;
        EXPECT_FLOAT_EQ(e.curvature, a.curvature) << "point " << i;
        EXPECT_EQ(e.rgba, a.rgba) << "point " << i;
    }
}

}  // namespace

TEST(Util3dTransformsTest, TransformAndMergeMatchesSequentialTransformThenAppend)
{
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> pos(-50.0f, 50.0f);
    std::uniform_real_distribution<float> ang(-3.0f, 3.0f);

    std::vector<std::pair<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr, Transform> > clouds;
    pcl::PointCloud<pcl::PointXYZRGBNormal> expected;
    const size_t sizes[] = {1, 500, 0, 3000, 17, 250, 1200, 64};
    for(size_t size : sizes)
    {
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud = randomCloud(rng, size);
        Transform pose(pos(rng), pos(rng), pos(rng), ang(rng), ang(rng), ang(rng));
        if(!cloud->empty()) // the PCL transform behind transformPointCloud() divides by zero on an empty cloud
        {
            expected += *util3d::transformPointCloud(cloud, pose);
        }
        clouds.push_back(std::make_pair(cloud, pose));
    }

    // One thread, a few, and one per core must all give the same cloud, in the same order.
    for(unsigned int threads : {1u, 3u, 0u})
    {
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr merged = util3d::transformAndMergePointClouds(clouds, threads);
        ASSERT_TRUE(merged.get() != nullptr);
        EXPECT_EQ(merged->height, 1u);
        EXPECT_EQ(merged->width, expected.size());
        expectSamePoints(expected, *merged);
    }
}

TEST(Util3dTransformsTest, TransformAndMergeSkipsNullCloudsAndHandlesEmptyInput)
{
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr empty = util3d::transformAndMergePointClouds(
            std::vector<std::pair<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr, Transform> >());
    ASSERT_TRUE(empty.get() != nullptr);
    EXPECT_TRUE(empty->empty());

    std::mt19937 rng(3);
    std::vector<std::pair<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr, Transform> > clouds;
    clouds.push_back(std::make_pair(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(), Transform::getIdentity()));
    clouds.push_back(std::make_pair(randomCloud(rng, 10), Transform(1, 2, 3, 0, 0, 0)));
    clouds.push_back(std::make_pair(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(), Transform::getIdentity()));
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr merged = util3d::transformAndMergePointClouds(clouds);
    ASSERT_EQ(merged->size(), 10u);
    expectSamePoints(*util3d::transformPointCloud(clouds[1].first, clouds[1].second), *merged);
}

TEST(Util3dTransformsTest, TransformAndMergeDensenessAndNaNPoints)
{
    std::mt19937 rng(5);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr dense = randomCloud(rng, 4);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr sparse = randomCloud(rng, 4);
    sparse->points[2].x = sparse->points[2].y = sparse->points[2].z = std::numeric_limits<float>::quiet_NaN();
    sparse->is_dense = false;

    std::vector<std::pair<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr, Transform> > clouds;
    clouds.push_back(std::make_pair(dense, Transform(0, 0, 1, 0, 0, 0)));
    EXPECT_TRUE(util3d::transformAndMergePointClouds(clouds)->is_dense);
    clouds.push_back(std::make_pair(sparse, Transform(1, 0, 0, 0, 0, 1.0f)));
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr merged = util3d::transformAndMergePointClouds(clouds);
    EXPECT_FALSE(merged->is_dense);
    ASSERT_EQ(merged->size(), 8u);
    EXPECT_TRUE(std::isnan(merged->points[4 + 2].x));
    EXPECT_FALSE(std::isnan(merged->points[4 + 1].x));
}
