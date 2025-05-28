//
//  RTABMap.swift
//  RTABMapApp
//
//  Created by Mathieu Labbe on 2020-12-31.
//

import Foundation
import ARKit

class RTABMap {
    var native_rtabmap: UnsafeMutableRawPointer
    
    struct Observation {
        weak var observer: RTABMapObserver?
    }
    private var observations = [ObjectIdentifier : Observation]()
    
    func addObserver(_ observer: RTABMapObserver) {
        let id = ObjectIdentifier(observer)
        observations[id] = Observation(observer: observer)
    }

    func removeObserver(_ observer: RTABMapObserver) {
        let id = ObjectIdentifier(observer)
        observations.removeValue(forKey: id)
    }
    
    init() {
        native_rtabmap = UnsafeMutableRawPointer(mutating: createNativeApplication())
    }
    
    func setupCallbacksWithCPP()
    {
        setupCallbacksNative(native_rtabmap,
             UnsafeMutableRawPointer(Unmanaged.passUnretained(self).toOpaque()),
             //progressCallback
             {(observer, count, max) -> Void in
                         // Extract pointer to `self` from void pointer:
                let mySelf = Unmanaged<RTABMap>.fromOpaque(observer!).takeUnretainedValue()
                         // Call instance method:
                         //mySelf.TestMethod();
                for (id, observation) in mySelf.observations {
                    // If the observer is no longer in memory, we
                    // can clean up the observation for its ID
                    guard let observer = observation.observer else {
                        mySelf.observations.removeValue(forKey: id)
                        continue
                    }

                    observer.progressUpdated(mySelf, count: Int(count), max: Int(max))
                }
             },
             //initCallback
             {(observer, status, msg) -> Void in
                         // Extract pointer to `self` from void pointer:
                let mySelf = Unmanaged<RTABMap>.fromOpaque(observer!).takeUnretainedValue()
                         // Call instance method:
                         //mySelf.TestMethod();
                for (id, observation) in mySelf.observations {
                    // If the observer is no longer in memory, we
                    // can clean up the observation for its ID
                    guard let observer = observation.observer else {
                        mySelf.observations.removeValue(forKey: id)
                        continue
                    }

                    let str = String(cString: msg!)
                    observer.initEventReceived(mySelf, status: Int(status), msg: str)
                }
             },
             //statsUpdatedCallback
             {(observer, nodes, words, points, polygons, updateTime, loopClosureId, highestHypId, databaseMemoryUsed, inliers, matches, featuresExtracted, hypothesis, nodesDrawn, fps, rejected, rehearsalValue, optimizationMaxError, optimizationMaxErrorRatio, distanceTravelled, fastMovement, landmarkDetected, x, y, z, roll, pitch, yaw) -> Void in
                         // Extract pointer to `self` from void pointer:
                let mySelf = Unmanaged<RTABMap>.fromOpaque(observer!).takeUnretainedValue()
                         // Call instance method:
                         //mySelf.TestMethod();
                for (id, observation) in mySelf.observations {
                    // If the observer is no longer in memory, we
                    // can clean up the observation for its ID
                    guard let observer = observation.observer else {
                        mySelf.observations.removeValue(forKey: id)
                        continue
                    }

                    observer.statsUpdated(mySelf, nodes: Int(nodes), words: Int(words), points: Int(points), polygons: Int(polygons), updateTime: updateTime, loopClosureId: Int(loopClosureId), highestHypId: Int(highestHypId), databaseMemoryUsed: Int(databaseMemoryUsed), inliers: Int(inliers), matches: Int(matches), featuresExtracted: Int(featuresExtracted), hypothesis: hypothesis, nodesDrawn: Int(nodesDrawn), fps: fps, rejected: Int(rejected), rehearsalValue: rehearsalValue, optimizationMaxError: optimizationMaxError, optimizationMaxErrorRatio: optimizationMaxErrorRatio, distanceTravelled: distanceTravelled, fastMovement: Int(fastMovement), landmarkDetected: Int(landmarkDetected), x: x, y: y, z: z, roll: roll, pitch: pitch, yaw: yaw)
                }
             },
             //cameraInfoEventCallback
             {(observer, type, key, value) -> Void in
                         // Extract pointer to `self` from void pointer:
                let mySelf = Unmanaged<RTABMap>.fromOpaque(observer!).takeUnretainedValue()
                         // Call instance method:
                         //mySelf.TestMethod();
                for (id, observation) in mySelf.observations {
                    // If the observer is no longer in memory, we
                    // can clean up the observation for its ID
                    guard let observer = observation.observer else {
                        mySelf.observations.removeValue(forKey: id)
                        continue
                    }
					
                    let strKey = String(cString: key!)
                    let strValue = String(cString: value!)
                    observer.cameraInfoEventReceived(mySelf, type: Int(type), key: strKey, value: strValue)
                }
             })
    }
    
    deinit {
        destroyNativeApplication(native_rtabmap)
    }
    
    func initGlContent() {
        initGlContentNative(native_rtabmap)
    }
    
    func setupGraphic(size: CGSize, orientation: UIInterfaceOrientation) {
        var rotation: Int32
        switch orientation.rawValue {
        case 4:
            rotation=2
        case 1:
            rotation=3
        case 2:
            rotation=1
        default:
            rotation=0
        }
        //NSLog("Orientation: ios %d android %d, view=%dx%d", orientation.rawValue, rotation, Int32(size.width), Int32(size.height))
        setScreenRotationNative(native_rtabmap, rotation)
        setupGraphicNative(native_rtabmap, Int32(size.width), Int32(size.height));
    }
    
    func openDatabase(databasePath:String, databaseInMemory:Bool, optimize:Bool, clearDatabase: Bool) -> Int {
        databasePath.utf8CString.withUnsafeBufferPointer { buffer -> Int in
            return Int(openDatabaseNative(native_rtabmap, buffer.baseAddress, databaseInMemory, optimize, clearDatabase))
        }
    }
    
    func save(databasePath:String) {
        databasePath.utf8CString.withUnsafeBufferPointer { buffer in
            saveNative(native_rtabmap, databasePath)
        }
    }
    
    func recover(from: String, to: String) -> Bool {
        from.utf8CString.withUnsafeBufferPointer { bufferFrom -> Bool in
            to.utf8CString.withUnsafeBufferPointer { bufferTo -> Bool in
                return recoverNative(native_rtabmap, bufferFrom.baseAddress, bufferTo.baseAddress)
            }
        }
    }
    
    func removeMeasure() {
        removeMeasureNative(native_rtabmap)
    }
    func addMeasureButtonClicked() {
        addMeasureNative(native_rtabmap)
    }
    func teleportButtonClicked() {
        teleportNative(native_rtabmap)
    }
    func setMeasuringMode(_ mode: Int) {
        setMeasuringModeNative(native_rtabmap, Int32(mode))
    }
    func setMetricSystem(_ enabled: Bool) {
        setMetricSystemNative(native_rtabmap, enabled)
    }
    func setMeasuringTextSize(_ size: Float32) {
        setMeasuringTextSizeNative(native_rtabmap, size)
    }
    func clearMeasures() {
        clearMeasuresNative(native_rtabmap)
    }
    func cancelProcessing() {
        cancelProcessingNative(native_rtabmap);
    }
    
    func postProcessing(approach: Int) -> Int {
        return Int(postProcessingNative(native_rtabmap, Int32(approach)))
    }
    
    func exportMesh(
        cloudVoxelSize: Float,
        regenerateCloud: Bool,
        meshing: Bool,
        textureSize: Int,
        textureCount: Int,
        normalK: Int,
        optimized: Bool,
        optimizedVoxelSize: Float,
        optimizedDepth: Int,
        optimizedMaxPolygons: Int,
        optimizedColorRadius: Float,
        optimizedCleanWhitePolygons: Bool,
        optimizedMinClusterSize: Int,
        optimizedMaxTextureDistance: Float,
        optimizedMinTextureClusterSize: Int,
        textureVertexColorPolicy: Int,
        blockRendering: Bool) -> Bool
    {
       return exportMeshNative(native_rtabmap,
                                    cloudVoxelSize,
                                    regenerateCloud,
                                    meshing,
                                    Int32(textureSize),
                                    Int32(textureCount),
                                    Int32(normalK),
                                    optimized,
                                    optimizedVoxelSize,
                                    Int32(optimizedDepth),
                                    Int32(optimizedMaxPolygons),
                                    optimizedColorRadius,
                                    optimizedCleanWhitePolygons,
                                    Int32(optimizedMinClusterSize),
                                    optimizedMaxTextureDistance,
                                    Int32(optimizedMinTextureClusterSize),
                            		Int32(textureVertexColorPolicy),
                                    blockRendering)
    }
    
    func postExportation(visualize: Bool) -> Bool
    {
        return postExportationNative(native_rtabmap, visualize)
    }
    
    func writeExportedMesh(directory: String, name: String) -> Bool
    {
        directory.utf8CString.withUnsafeBufferPointer { bufferDir in
            name.utf8CString.withUnsafeBufferPointer { bufferName in
                return writeExportedMeshNative(native_rtabmap, bufferDir.baseAddress, bufferName.baseAddress)
            }
        }
    }
    
    func onTouchEvent(touch_count: Int, event: Int, x0: Float, y0: Float, x1: Float, y1: Float) {
        onTouchEventNative(native_rtabmap, Int32(touch_count), Int32(event), x0, y0, x1, y1)
    }
    
    func setPausedMapping(paused: Bool) {
        setPausedMappingNative(native_rtabmap, paused)
    }
    
    func render() -> Int {
        return Int(renderNative(native_rtabmap))
    }
    
    func startCamera(imageOverlayInFirstPerson: Bool = true) -> Bool {
        return startCameraNative(native_rtabmap)
    }
    
    func stopCamera() {
        stopCameraNative(native_rtabmap)
    }
    
    func setCamera(type: Int) {
        setCameraNative(native_rtabmap, Int32(type))
    }

    func postOdometryEvent(frame: ARFrame, orientation: UIInterfaceOrientation, viewport: CGSize) {
        let pose = frame.camera.transform   // ViewMatrix
        
        // Convert camera transform to SIMD rotation matrix and position
        let rotation = simd_float3x3(
            SIMD3<Float>(pose[0,0], pose[0,1], pose[0,2]),
            SIMD3<Float>(pose[1,0], pose[1,1], pose[1,2]),
            SIMD3<Float>(pose[2,0], pose[2,1], pose[2,2])
        )
        let translation = SIMD3<Float>(pose[3,0], pose[3,1], pose[3,2])
        
        // Calculate position by applying inverse transform 
        // position = -(R^T * t) where R^T is transpose of rotation matrix
        let position = -(rotation.transpose * translation)
        
        var pixelBuffer: CVPixelBuffer?
        var cameraIntrinsics = frame.camera.intrinsics
        var depthBuffer: CVPixelBuffer?
        var depthMediumBuffer: CVPixelBuffer?
        var depthLowBuffer: CVPixelBuffer?
        var confidenceBuffer: CVPixelBuffer?
        
        // Get RGB image
        pixelBuffer = frame.capturedImage
        
        // Get depth image with different confidence levels
        if #available(iOS 14.0, *) {
            if frame.sceneDepth != nil {
                depthBuffer = frame.sceneDepth?.depthMap // High confidence
                confidenceBuffer = frame.sceneDepth?.confidenceMap
                
                // Create medium and low confidence maps
                if let confidenceMap = frame.sceneDepth?.confidenceMap {
                    let confidenceWidth = CVPixelBufferGetWidth(confidenceMap)
                    let confidenceHeight = CVPixelBufferGetHeight(confidenceMap)
                    
                    let depthMap = frame.sceneDepth!.depthMap
                    CVPixelBufferLockBaseAddress(depthMap, .readOnly)
                    CVPixelBufferLockBaseAddress(confidenceMap, .readOnly)
                    
                    // Create medium confidence depth map (confidence level 1)
                    let mediumDepthPixelFormat = kCVPixelFormatType_DepthFloat32
                    var mediumDepthPixelBuffer: CVPixelBuffer?
                    let mediumStatus = CVPixelBufferCreate(kCFAllocatorDefault,
                                                         confidenceWidth,
                                                         confidenceHeight,
                                                         mediumDepthPixelFormat,
                                                         nil,
                                                         &mediumDepthPixelBuffer)
                    
                    // Create low confidence depth map (confidence level 0)
                    let lowDepthPixelFormat = kCVPixelFormatType_DepthFloat32
                    var lowDepthPixelBuffer: CVPixelBuffer?
                    let lowStatus = CVPixelBufferCreate(kCFAllocatorDefault,
                                                      confidenceWidth,
                                                      confidenceHeight,
                                                      lowDepthPixelFormat,
                                                      nil,
                                                      &lowDepthPixelBuffer)
                    
                    if mediumStatus == kCVReturnSuccess && lowStatus == kCVReturnSuccess,
                       let mediumBuffer = mediumDepthPixelBuffer,
                       let lowBuffer = lowDepthPixelBuffer {
                        
                        CVPixelBufferLockBaseAddress(mediumBuffer, [])
                        CVPixelBufferLockBaseAddress(lowBuffer, [])
                        
                        let depthData = CVPixelBufferGetBaseAddress(depthMap)
                        let confidenceData = CVPixelBufferGetBaseAddress(confidenceMap)
                        let mediumData = CVPixelBufferGetBaseAddress(mediumBuffer)
                        let lowData = CVPixelBufferGetBaseAddress(lowBuffer)
                        
                        let depthBytesPerRow = CVPixelBufferGetBytesPerRow(depthMap)
                        let confidenceBytesPerRow = CVPixelBufferGetBytesPerRow(confidenceMap)
                        let mediumBytesPerRow = CVPixelBufferGetBytesPerRow(mediumBuffer)
                        let lowBytesPerRow = CVPixelBufferGetBytesPerRow(lowBuffer)
                        
                        for y in 0..<confidenceHeight {
                            let depthRow = depthData?.advanced(by: y * depthBytesPerRow)
                            let confidenceRow = confidenceData?.advanced(by: y * confidenceBytesPerRow)
                            let mediumRow = mediumData?.advanced(by: y * mediumBytesPerRow)
                            let lowRow = lowData?.advanced(by: y * lowBytesPerRow)
                            
                            for x in 0..<confidenceWidth {
                                let confidence = confidenceRow?.load(fromByteOffset: x, as: UInt8.self) ?? 0
                                let depth = depthRow?.load(fromByteOffset: x * 4, as: Float32.self) ?? 0
                                
                                if confidence == 1 {
                                    mediumRow?.storeBytes(of: depth, toByteOffset: x * 4, as: Float32.self)
                                }
                                else if confidence == 0 {
                                    lowRow?.storeBytes(of: depth, toByteOffset: x * 4, as: Float32.self)
                                }
                            }
                        }
                        
                        CVPixelBufferUnlockBaseAddress(mediumBuffer, [])
                        CVPixelBufferUnlockBaseAddress(lowBuffer, [])
                        
                        depthMediumBuffer = mediumBuffer
                        depthLowBuffer = lowBuffer
                    }
                    
                    CVPixelBufferUnlockBaseAddress(depthMap, .readOnly)
                    CVPixelBufferUnlockBaseAddress(confidenceMap, .readOnly)
                }
            }
        }
        
        if let pixelBuffer = pixelBuffer {
            let height = CVPixelBufferGetHeight(pixelBuffer)
            let width = CVPixelBufferGetWidth(pixelBuffer)
            var fx = cameraIntrinsics[0,0]
            var fy = cameraIntrinsics[1,1]
            var cx = cameraIntrinsics[2,0]
            var cy = cameraIntrinsics[2,1]
            
            var yPlane: UnsafeMutableRawPointer?
            var yPlaneBytesPerRow = 0
            var vPlane: UnsafeMutableRawPointer?
            var vPlaneBytesPerRow = 0
            
            // Lock Y Plane
            CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
            if CVPixelBufferGetPlaneCount(pixelBuffer) >= 2 {
                yPlane = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0)
                yPlaneBytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0)
                vPlane = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 1)
                vPlaneBytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 1)
            }
            
            var depth: UnsafeMutableRawPointer?
            var depthBytesPerRow = 0
            if let depthBuffer = depthBuffer {
                CVPixelBufferLockBaseAddress(depthBuffer, .readOnly)
                depth = CVPixelBufferGetBaseAddress(depthBuffer)
                depthBytesPerRow = CVPixelBufferGetBytesPerRow(depthBuffer)
            }
            
            var depthMedium: UnsafeMutableRawPointer?
            var depthMediumBytesPerRow = 0
            if let depthMediumBuffer = depthMediumBuffer {
                CVPixelBufferLockBaseAddress(depthMediumBuffer, .readOnly)
                depthMedium = CVPixelBufferGetBaseAddress(depthMediumBuffer)
                depthMediumBytesPerRow = CVPixelBufferGetBytesPerRow(depthMediumBuffer)
            }
            
            var depthLow: UnsafeMutableRawPointer?
            var depthLowBytesPerRow = 0
            if let depthLowBuffer = depthLowBuffer {
                CVPixelBufferLockBaseAddress(depthLowBuffer, .readOnly)
                depthLow = CVPixelBufferGetBaseAddress(depthLowBuffer)
                depthLowBytesPerRow = CVPixelBufferGetBytesPerRow(depthLowBuffer)
            }
            
            var confidence: UnsafeMutableRawPointer?
            var confidenceBytesPerRow = 0
            if let confidenceBuffer = confidenceBuffer {
                CVPixelBufferLockBaseAddress(confidenceBuffer, .readOnly)
                confidence = CVPixelBufferGetBaseAddress(confidenceBuffer)
                confidenceBytesPerRow = CVPixelBufferGetBytesPerRow(confidenceBuffer)
            }
            
            var rotation: Int32
            switch orientation.rawValue {
            case 4:
                rotation=2
            case 1:
                rotation=3
            case 2:
                rotation=1
            default:
                rotation=0
            }
            
            postOdometryEventNative(
                native_rtabmap,
                position.x, position.y, position.z,
                rotation.m00, rotation.m01, rotation.m02,
                rotation.m10, rotation.m11, rotation.m12,
                rotation.m20, rotation.m21, rotation.m22,
                fx, fy, cx, cy,
                Int32(width), Int32(height),
                frame.timestamp,
                yPlane, Int32(yPlaneBytesPerRow),
                vPlane, Int32(vPlaneBytesPerRow),
                depth, Int32(depthBytesPerRow),
                depthMedium, Int32(depthMediumBytesPerRow),
                depthLow, Int32(depthLowBytesPerRow),
                confidence, Int32(confidenceBytesPerRow),
                Int32(rotation))
            
            CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
            if let depthBuffer = depthBuffer {
                CVPixelBufferUnlockBaseAddress(depthBuffer, .readOnly)
            }
            if let depthMediumBuffer = depthMediumBuffer {
                CVPixelBufferUnlockBaseAddress(depthMediumBuffer, .readOnly)
            }
            if let depthLowBuffer = depthLowBuffer {
                CVPixelBufferUnlockBaseAddress(depthLowBuffer, .readOnly)
            }
            if let confidenceBuffer = confidenceBuffer {
                CVPixelBufferUnlockBaseAddress(confidenceBuffer, .readOnly)
            }
        }
    }
        
    // Parameters
    func setOnlineBlending(enabled: Bool) {
        setOnlineBlendingNative(native_rtabmap, enabled)
    }
    func setMapCloudShown(shown: Bool) {
        setMapCloudShownNative(native_rtabmap, shown)
    }
    func setOdomCloudShown(shown: Bool) {
        setOdomCloudShownNative(native_rtabmap, shown)
    }
    func setMeshRendering(enabled: Bool, withTexture: Bool) {
        setMeshRenderingNative(native_rtabmap, enabled, withTexture)
    }
    func setPointSize(value: Float) {
        setPointSizeNative(native_rtabmap, value)
    }
    func setFOV(angle: Float) {
        setFOVNative(native_rtabmap, angle)
    }
    func setOrthoCropFactor(value: Float) {
        setOrthoCropFactorNative(native_rtabmap, value)
    }
    func setGridRotation(value: Float) {
        setGridRotationNative(native_rtabmap, value)
    }
    func setLighting(enabled: Bool) {
        setLightingNative(native_rtabmap, enabled)
    }
    func setBackfaceCulling(enabled: Bool) {
        setBackfaceCullingNative(native_rtabmap, enabled)
    }
    func setWireframe(enabled: Bool) {
        setWireframeNative(native_rtabmap, enabled)
    }
    func setTextureColorSeamsHidden(hidden: Bool) {
        setTextureColorSeamsHiddenNative(native_rtabmap, hidden)
    }
    func setLocalizationMode(enabled: Bool) {
        setLocalizationModeNative(native_rtabmap, enabled)
    }
    func setDataRecorderMode(enabled: Bool) {
        setDataRecorderModeNative(native_rtabmap, enabled)
    }
    func setGraphOptimization(enabled: Bool) {
        setGraphOptimizationNative(native_rtabmap, enabled)
    }
    func setNodesFiltering(enabled: Bool) {
        setNodesFilteringNative(native_rtabmap, enabled)
    }
    func setGraphVisible(visible: Bool) {
        setGraphVisibleNative(native_rtabmap, visible)
    }
    func setGridVisible(visible: Bool) {
        setGridVisibleNative(native_rtabmap, visible)
    }
    func setFullResolution(enabled: Bool) {
        setFullResolutionNative(native_rtabmap, enabled)
    }
    func setSmoothing(enabled: Bool) {
        setSmoothingNative(native_rtabmap, enabled)
    }
    func setAppendMode(enabled: Bool) {
        setAppendModeNative(native_rtabmap, enabled)
    }
    func setUpstreamRelocalizationAccThr(value: Float) {
        setUpstreamRelocalizationAccThrNative(native_rtabmap, value)
    }
    func setMaxCloudDepth(value: Float) {
        setMaxCloudDepthNative(native_rtabmap, value)
    }
    func setMinCloudDepth(value: Float) {
        setMinCloudDepthNative(native_rtabmap, value)
    }
    func setCloudDensityLevel(value: Int) {
        setCloudDensityLevelNative(native_rtabmap, Int32(value))
    }
    func setMeshAngleTolerance(value: Float) {
        setMeshAngleToleranceNative(native_rtabmap, value)
    }
    func setMeshDecimationFactor(value: Float) {
        setMeshDecimationFactorNative(native_rtabmap, value)
    }
    func setMeshTriangleSize(value: Int) {
        setMeshTriangleSizeNative(native_rtabmap, Int32(value))
    }
    func setClusterRatio(value: Float) {
        setClusterRatioNative(native_rtabmap, value)
    }
    func setMaxGainRadius(value: Float) {
        setMaxGainRadiusNative(native_rtabmap, value)
    }
    func setRenderingTextureDecimation(value: Int) {
        setRenderingTextureDecimationNative(native_rtabmap, Int32(value))
    }
    func setBackgroundColor(gray: Float) {
        setBackgroundColorNative(native_rtabmap, gray)
    }
    func setDepthConfidence(value: Int) {
        setDepthConfidenceNative(native_rtabmap, Int32(value))
    }
    func setExportPointCloudFormat(format: String) {
        format.utf8CString.withUnsafeBufferPointer { bufferFormat in
            return setExportPointCloudFormatNative(native_rtabmap, bufferFormat.baseAddress)
        }
    }
    func setMappingParameter(key: String, value: String) {
        key.utf8CString.withUnsafeBufferPointer { bufferKey in
            value.utf8CString.withUnsafeBufferPointer { bufferValue in
                setMappingParameterNative(native_rtabmap, bufferKey.baseAddress, bufferValue.baseAddress)
            }
        }
    }
    func setGPS(location: CLLocation)
    {
        setGPSNative(native_rtabmap, location.timestamp.timeIntervalSince1970, location.coordinate.longitude, location.coordinate.latitude, location.altitude, location.horizontalAccuracy, location.course < 0.0 ? 0.0 : location.course)
    }
    func addEnvSensor(type: Int, value: Float)
    {
        addEnvSensorNative(native_rtabmap, Int32(type), value)
    }
    func setOrthoCropFactor(_ value: Float)
    {
        setOrthoCropFactorNative(native_rtabmap, value)
    }
    func setGridRotation(_ value: Float)
    {
        setGridRotationNative(native_rtabmap, value)
    }
}

func getPreviewImage(databasePath: String) -> UIImage?
{
    let imageOut = databasePath.utf8CString.withUnsafeBufferPointer { buffer -> UIImage? in
        var image = ImageNative()
        image = getPreviewImageNative(buffer.baseAddress)
        if let dataPtr = UnsafeRawPointer(image.data)
        {
            // copy data in swift
            let data = Data(bytes: dataPtr, count: Int(image.width*image.height*image.channels))
            // release memory
            releasePreviewImageNative(image)
            // Create bitmap image
            let bitmap = CIImage(bitmapData: data, bytesPerRow: Int(image.width*image.channels), size: CGSize(width: Int(image.width), height: Int(image.height)), format: CIFormat.BGRA8, colorSpace: nil)
            return UIImage(ciImage: bitmap)
        }
        return UIImage(named: "RTAB-Map1024")
    }
    return imageOut
}

protocol RTABMapObserver: class {
    func progressUpdated(_ rtabmap: RTABMap, count: Int, max: Int)
    func initEventReceived(_ rtabmap: RTABMap, status: Int, msg: String)
    func statsUpdated(_ rtabmap: RTABMap,
                      nodes: Int,
                      words: Int,
                      points: Int,
                      polygons: Int,
                      updateTime: Float,
                      loopClosureId: Int,
                      highestHypId: Int,
                      databaseMemoryUsed: Int,
                      inliers: Int,
                      matches: Int,
                      featuresExtracted: Int,
                      hypothesis: Float,
                      nodesDrawn: Int,
                      fps: Float,
                      rejected: Int,
                      rehearsalValue: Float,
                      optimizationMaxError: Float,
                      optimizationMaxErrorRatio: Float,
                      distanceTravelled: Float,
                      fastMovement: Int,
                      landmarkDetected: Int,
                      x: Float,
                      y: Float,
                      z: Float,
                      roll: Float,
                      pitch: Float,
                      yaw: Float)
    func cameraInfoEventReceived(_ rtabmap: RTABMap, type: Int, key: String, value: String)
}

extension String {

  func toPointer() -> UnsafePointer<UInt8>? {
    guard let data = self.data(using: String.Encoding.utf8) else { return nil }

    let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: data.count)
    let stream = OutputStream(toBuffer: buffer, capacity: data.count)

    stream.open()
    data.withUnsafeBytes({ (p: UnsafePointer<UInt8>) -> Void in
      stream.write(p, maxLength: data.count)
    })

    stream.close()

    return UnsafePointer<UInt8>(buffer)
  }
}

