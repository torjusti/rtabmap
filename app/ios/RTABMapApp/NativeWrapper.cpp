//
//  PCLWrapper.cpp
//  ThreeDScanner
//
//  Created by Steven Roach on 2/9/18.
//  Copyright © 2018 Steven Roach. All rights reserved.
//

#include "NativeWrapper.hpp"
#include "RTABMapApp.h"
#include <rtabmap/core/DBDriverSqlite3.h>

inline RTABMapApp *native(const void *object) {
  return (RTABMapApp *)object;
}

const void * createNativeApplication()
{
    RTABMapApp *app = new RTABMapApp();
    return (void *)app;
}

void setupCallbacksNative(const void *object, void * classPtr,
                          void(*progressCallback)(void*, int, int),
                          void(*initCallback)(void *, int, const char*),
                          void(*statsUpdatedCallback)(void *,
                                                   int, int, int, int,
                                                   float,
                                                   int, int, int, int, int ,int,
                                                   float,
                                                   int,
                                                   float,
                                                   int,
                                                   float, float, float, float,
                                                   int, int,
                                                   float, float, float, float, float, float),
                          void(*cameraInfoEventCallback)(void *, int, const char*, const char*))
{
    if(object)
    {
        native(object)->setupSwiftCallbacks(classPtr, progressCallback, initCallback, statsUpdatedCallback, cameraInfoEventCallback);
    }
    else
    {
        UERROR("object is null!");
    }
}

void destroyNativeApplication(const void *object)
{
    if(object)
    {
        delete native(object);
    }
    else
    {
        UERROR("object is null!");
    }
}

void setScreenRotationNative(const void *object, int displayRotation)
{
    if(object)
    {
        return native(object)->setScreenRotation(displayRotation, 0);
    }
    else
    {
        UERROR("object is null!");
    }
}

int openDatabaseNative(const void *object, const char * databasePath, bool databaseInMemory, bool optimize, bool clearDatabase)
{
    if(object)
    {
        return native(object)->openDatabase(databasePath, databaseInMemory, optimize, clearDatabase);
    }
    else
    {
        UERROR("object is null!");
        return -1;
    }
}

void saveNative(const void *object, const char * databasePath)
{
    if(object)
    {
        return native(object)->save(databasePath);
    }
    else
    {
        UERROR("object is null!");
    }
}

bool recoverNative(const void *object, const char * from, const char * to)
{
    if(object)
    {
        return native(object)->recover(from, to);
    }
    else
    {
        UERROR("object is null!");
    }
    return false;
}

void cancelProcessingNative(const void *object)
{
    if(object)
    {
        native(object)->cancelProcessing();
    }
    else
    {
        UERROR("object is null!");
    }
}

int postProcessingNative(const void *object, int approach)
{
    if(object)
    {
        return native(object)->postProcessing(approach);
    }
    else
    {
        UERROR("object is null!");
    }
    return -1;
}

bool exportMeshNative(
            const void *object,
            float cloudVoxelSize,
            bool regenerateCloud,
            bool meshing,
            int textureSize,
            int textureCount,
            int normalK,
            bool optimized,
            float optimizedVoxelSize,
            int optimizedDepth,
            int optimizedMaxPolygons,
            float optimizedColorRadius,
            bool optimizedCleanWhitePolygons,
            int optimizedMinClusterSize,
            float optimizedMaxTextureDistance,
            int optimizedMinTextureClusterSize,
            int textureVertexColorPolicy,
            bool blockRendering)
{
    if(object)
    {
        return native(object)->exportMesh(cloudVoxelSize, regenerateCloud, meshing, textureSize, textureCount, normalK, optimized, optimizedVoxelSize, optimizedDepth, optimizedMaxPolygons, optimizedColorRadius, optimizedCleanWhitePolygons, optimizedMinClusterSize, optimizedMaxTextureDistance, optimizedMinTextureClusterSize, textureVertexColorPolicy, blockRendering);
    }
    else
    {
        UERROR("object is null!");
    }
    return false;
}

bool postExportationNative(const void *object, bool visualize)
{
    if(object)
    {
        return native(object)->postExportation(visualize);
    }
    else
    {
        UERROR("object is null!");
    }
    return false;
}

bool writeExportedMeshNative(const void *object, const char * directory, const char * name)
{
    if(object)
    {
        return native(object)->writeExportedMesh(directory, name);
    }
    else
    {
        UERROR("object is null!");
    }
    return false;
}

void initGlContentNative(const void *object) {
    if(object)
    {
        native(object)->InitializeGLContent();
    }
    else
    {
        UERROR("object is null!");
    }
}

void setupGraphicNative(const void *object, int width, int height) {
    if(object)
    {
        native(object)->SetViewPort(width, height);
    }
    else
    {
        UERROR("object is null!");
    }
}

void onTouchEventNative(const void *object, int touch_count, int event, float x0, float y0, float x1,
    float y1) {
    if(object)
    {
        using namespace tango_gl;
        GestureCamera::TouchEvent touch_event =
          static_cast<GestureCamera::TouchEvent>(event);
        native(object)->OnTouchEvent(touch_count, touch_event, x0, y0, x1, y1);
    }
    else
    {
        UERROR("object is null!");
    }
}

void setPausedMappingNative(const void *object, bool paused)
{
    if(object)
    {
        return native(object)->setPausedMapping(paused);
    }
    else
    {
        UERROR("object is null!");
    }
}

int renderNative(const void *object) {
    if(object)
    {
        return native(object)->Render();
    }
    else
    {
        UERROR("object is null!");
        return -1;
    }
}

bool startCameraNative(const void *object) {
    if(object)
    {
        return native(object)->startCamera();
    }
    else
    {
        UERROR("object is null!");
        return false;
    }
}

void stopCameraNative(const void *object) {
    if(object)
    {
        native(object)->stopCamera();
    }
    else
    {
        UERROR("object is null!");
    }
}

void setCameraNative(const void *object, int type) {
    if(object)
    {
        native(object)->SetCameraType(tango_gl::GestureCamera::CameraType(type));
    }
    else
    {
        UERROR("object is null!");
    }
}

void postOdometryEventNative(const void *object,
                       float x, float y, float z,
                       float qx, float qy, float qz,
                       float fx, float fy, float cx, float cy,
                       int width, int height,
                       double stamp,
                       const void * yPlane, int yPlaneBytesPerRow,
                       const void * vPlane, int vPlaneBytesPerRow,
                       const void * depth, int depthBytesPerRow,
                       const void * depthMedium, int depthMediumBytesPerRow,
                       const void * depthLow, int depthLowBytesPerRow,
                       const void * confidence, int confidenceBytesPerRow,
                       int orientation)
{
    if(object)
    {
        rtabmap::Transform pose;
        if(!(x==0.0f && y==0.0f && z==0.0f && qx==0.0f && qy==0.0f && qz==0.0f))
        {
            // Create pose from position and quaternion
            Eigen::Matrix3f R;
            R = Eigen::Quaternionf(qx, qy, qz).toRotationMatrix();
            pose = rtabmap::Transform(
                x, y, z,
                R(0,0), R(0,1), R(0,2),
                R(1,0), R(1,1), R(1,2),
                R(2,0), R(2,1), R(2,2));
        }
        
        cv::Mat rgbMat;
        if(yPlane != 0 && vPlane != 0)
        {
            // Convert YUV to RGB
            cv::Mat yMat(height, width, CV_8UC1, (void*)yPlane, yPlaneBytesPerRow);
            cv::Mat uvMat(height/2, width/2, CV_8UC2, (void*)vPlane, vPlaneBytesPerRow);
            cv::cvtColor(yMat, rgbMat, cv::COLOR_YUV2BGR_NV21);
        }
        
        cv::Mat depthMat;
        if(depth != 0)
        {
            depthMat = cv::Mat(height, width, CV_32FC1, (void*)depth, depthBytesPerRow);
        }
        
        cv::Mat depthMediumMat;
        if(depthMedium != 0)
        {
            depthMediumMat = cv::Mat(height, width, CV_32FC1, (void*)depthMedium, depthMediumBytesPerRow);
        }
        
        cv::Mat depthLowMat;
        if(depthLow != 0)
        {
            depthLowMat = cv::Mat(height, width, CV_32FC1, (void*)depthLow, depthLowBytesPerRow);
        }
        
        cv::Mat confidenceMat;
        if(confidence != 0)
        {
            confidenceMat = cv::Mat(height, width, CV_8UC1, (void*)confidence, confidenceBytesPerRow);
        }
        
        rtabmap::CameraModel model(
            fx, fy, cx, cy,
            pose,
            0,
            cv::Size(width, height));
        
        rtabmap::SensorData data(
            rgbMat,
            depthMat,
            model,
            0,
            stamp);
            
        // Add the additional depth maps and confidence
        if(!depthMediumMat.empty())
        {
            data.setDepthMedium(depthMediumMat);
        }
        if(!depthLowMat.empty())
        {
            data.setDepthLow(depthLowMat);
        }
        if(!confidenceMat.empty())
        {
            data.setConfidenceMap(confidenceMat);
        }
        
        native(object)->postOdometryEvent(data);
    }
    else
    {
        UERROR("object is null!");
        return;
    }
}

ImageNative getPreviewImageNative(const char * databasePath)
{
    ImageNative imageNative;
    imageNative.data = 0;
    imageNative.objectPtr = 0;
    rtabmap::DBDriverSqlite3 driver;
    if(driver.openConnection(databasePath))
    {
        cv::Mat image = driver.loadPreviewImage();
        if(image.empty())
        {
            return imageNative;
        }
        cv::Mat * imagePtr = new cv::Mat();
        // We should add alpha channel
        cv::cvtColor(image, *imagePtr, cv::COLOR_BGR2BGRA);
        std::vector<cv::Mat> channels;
        cv::split(*imagePtr, channels);
        channels.back() = cv::Scalar(255);
        cv::merge(channels, *imagePtr);
        imageNative.objectPtr = imagePtr;
        imageNative.data = imagePtr->data;
        imageNative.width = imagePtr->cols;
        imageNative.height = imagePtr->rows;
        imageNative.channels = imagePtr->channels();
    }
    return imageNative;
}

void releasePreviewImageNative(ImageNative image)
{
    if(image.objectPtr)
    {
        delete (cv::Mat*)image.objectPtr;
    }
}


// Parameters
void setOnlineBlendingNative(const void *object, bool enabled)
{
    if(object)
        native(object)->setOnlineBlending(enabled);
    else
        UERROR("object is null!");
}
void setMapCloudShownNative(const void *object, bool shown)
{
    if(object)
        native(object)->setMapCloudShown(shown);
    else
        UERROR("object is null!");
}
void setOdomCloudShownNative(const void *object, bool shown)
{
    if(object)
        native(object)->setOdomCloudShown(shown);
    else
        UERROR("object is null!");
}
void setMeshRenderingNative(const void *object, bool enabled, bool withTexture)
{
    if(object)
        native(object)->setMeshRendering(enabled, withTexture);
    else
        UERROR("object is null!");
}
void setPointSizeNative(const void *object, float value)
{
    if(object)
        native(object)->setPointSize(value);
    else
        UERROR("object is null!");
}
void setFOVNative(const void *object, float angle)
{
    if(object)
        native(object)->setFOV(angle);
    else
        UERROR("object is null!");
}
void setOrthoCropFactorNative(const void *object, float value)
{
    if(object)
        native(object)->setOrthoCropFactor(value);
    else
        UERROR("object is null!");
}
void setGridRotationNative(const void *object, float value)
{
    if(object)
        native(object)->setGridRotation(value);
    else
        UERROR("object is null!");
}
void setLightingNative(const void *object, bool enabled)
{
    if(object)
        native(object)->setLighting(enabled);
    else
        UERROR("object is null!");
}
void setBackfaceCullingNative(const void *object, bool enabled)
{
    if(object)
        native(object)->setBackfaceCulling(enabled);
    else
        UERROR("object is null!");
}
void setWireframeNative(const void *object, bool enabled)
{
    if(object)
        native(object)->setWireframe(enabled);
    else
        UERROR("object is null!");
}
void setTextureColorSeamsHiddenNative(const void *object, bool hidden)
{
    if(object)
        native(object)->setTextureColorSeamsHidden(hidden);
    else
        UERROR("object is null!");
}
void setLocalizationModeNative(const void *object, bool enabled)
{
    if(object)
        native(object)->setLocalizationMode(enabled);
    else
        UERROR("object is null!");
}
void setDataRecorderModeNative(const void *object, bool enabled)
{
    if(object)
        native(object)->setDataRecorderMode(enabled);
    else
        UERROR("object is null!");
}
void setTrajectoryModeNative(const void *object, bool enabled)
{
    if(object)
        native(object)->setTrajectoryMode(enabled);
    else
        UERROR("object is null!");
}
void setGraphOptimizationNative(const void *object, bool enabled)
{
    if(object)
        native(object)->setGraphOptimization(enabled);
    else
        UERROR("object is null!");
}
void setNodesFilteringNative(const void *object, bool enabled)
{
    if(object)
        native(object)->setNodesFiltering(enabled);
    else
        UERROR("object is null!");
}
void setGraphVisibleNative(const void *object, bool visible)
{
    if(object)
        native(object)->setGraphVisible(visible);
    else
        UERROR("object is null!");
}
void setGridVisibleNative(const void *object, bool visible)
{
    if(object)
        native(object)->setGridVisible(visible);
    else
        UERROR("object is null!");
}
void setFullResolutionNative(const void *object, bool enabled)
{
    if(object)
        native(object)->setFullResolution(enabled);
    else
        UERROR("object is null!");
}
void setSmoothingNative(const void *object, bool enabled)
{
    if(object)
        native(object)->setSmoothing(enabled);
    else
        UERROR("object is null!");
}
void setAppendModeNative(const void *object, bool enabled)
{
    if(object)
        native(object)->setAppendMode(enabled);
    else
        UERROR("object is null!");
}
void setUpstreamRelocalizationAccThrNative(const void * object, float value)
{
    if(object)
        native(object)->setUpstreamRelocalizationAccThr(value);
    else
        UERROR("object is null!");
}
void setMaxCloudDepthNative(const void *object, float value)
{
    if(object)
        native(object)->setMaxCloudDepth(value);
    else
        UERROR("object is null!");
}
void setMinCloudDepthNative(const void *object, float value)
{
    if(object)
        native(object)->setMinCloudDepth(value);
    else
        UERROR("object is null!");
}
void setCloudDensityLevelNative(const void *object, int value)
{
    if(object)
        native(object)->setCloudDensityLevel(value);
    else
        UERROR("object is null!");
}
void setMeshAngleToleranceNative(const void *object, float value)
{
    if(object)
        native(object)->setMeshAngleTolerance(value);
    else
        UERROR("object is null!");
}
void setMeshDecimationFactorNative(const void *object, float value)
{
    if(object)
        native(object)->setMeshDecimationFactor(value);
    else
        UERROR("object is null!");
}
void setMeshTriangleSizeNative(const void *object, int value)
{
    if(object)
        native(object)->setMeshTriangleSize(value);
    else
        UERROR("object is null!");
}
void setClusterRatioNative(const void *object, float value)
{
    if(object)
        native(object)->setClusterRatio(value);
    else
        UERROR("object is null!");
}
void setMaxGainRadiusNative(const void *object, float value)
{
    if(object)
        native(object)->setMaxGainRadius(value);
    else
        UERROR("object is null!");
}
void setRenderingTextureDecimationNative(const void *object, int value)
{
    if(object)
        native(object)->setRenderingTextureDecimation(value);
    else
        UERROR("object is null!");
}
void setBackgroundColorNative(const void *object, float gray)
{
    if(object)
        native(object)->setBackgroundColor(gray);
    else
        UERROR("object is null!");
}
void setDepthConfidenceNative(const void *object, int value)
{
    if(object)
        native(object)->setDepthConfidence(value);
    else
        UERROR("object is null!");
}

void setExportPointCloudFormatNative(const void *object, const char * format)
{
    if(object)
        native(object)->setExportPointCloudFormat(format);
    else
        UERROR("object is null!");
}

int setMappingParameterNative(const void *object, const char * key, const char * value)
{
    if(object)
        return native(object)->setMappingParameter(key, value);
    else
        UERROR("object is null!");
    return -1;
}

void setGPSNative(const void *object, double stamp, double longitude, double latitude, double altitude, double accuracy, double bearing)
{
    rtabmap::GPS gps(stamp, longitude, latitude, altitude, accuracy, bearing);
    if(object)
        return native(object)->setGPS(gps);
    else
        UERROR("object is null!");
}

void addEnvSensorNative(const void *object, int type, float value)
{
    if(object)
        return native(object)->addEnvSensor(type, value);
    else
        UERROR("object is null!");
}

void removeMeasureNative(const void *object)
{
    if(object)
        return native(object)->removeMeasure();
    else
        UERROR("object is null!");
}
void addMeasureNative(const void *object)
{
    if(object)
        return native(object)->addMeasureButtonClicked();
    else
        UERROR("object is null!");
}
void teleportNative(const void *object)
{
    if(object)
        return native(object)->teleportButtonClicked();
    else
        UERROR("object is null!");
}
void setMeasuringModeNative(const void *object, int mode)
{
    if(object)
        return native(object)->setMeasuringMode(mode);
    else
        UERROR("object is null!");
}
void setMetricSystemNative(const void *object, bool enabled)
{
    if(object)
        return native(object)->setMetricSystem(enabled);
    else
        UERROR("object is null!");
}
void setMeasuringTextSizeNative(const void *object, float size)
{
    if(object)
        return native(object)->setMeasuringTextSize(size);
    else
        UERROR("object is null!");
}
void clearMeasuresNative(const void *object)
{
    if(object)
        return native(object)->clearMeasures();
    else
        UERROR("object is null!");
}
