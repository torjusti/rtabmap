#include <iostream>
#include "solvepnp_msac.h"

#include "rtabmap/utilite/UMath.h"
#include "rtabmap/utilite/ULogger.h"

using namespace cv;

namespace cv_custom
{

const float CHI2_95_UNSQUARED = 2.44765f;

cv::Matx22d computeWhiteningMatrix(
    const cv::Point3f& pt3d, const cv::Matx33d& R33, const cv::Matx31d& tvec31, 
    double fx, double fy, const cv::Matx33f& cov3D, float pixelVariance) 
{
    Matx31d XA(pt3d.x, pt3d.y, pt3d.z);
    Matx31d XB = R33 * XA + tvec31;
    double Z = std::max(XB(2), 1e-5); 

    Matx23d J_pi(fx/Z, 0.0, -fx*XB(0)/(Z*Z), 0.0, fy/Z, -fy*XB(1)/(Z*Z));
    Matx23d J_XA = J_pi * R33;
    Matx22d cov2D = J_XA * Matx33d(cov3D) * J_XA.t();
    
    cov2D(0,0) += pixelVariance; cov2D(1,1) += pixelVariance;

    double det = std::max(cov2D(0,0)*cov2D(1,1) - cov2D(0,1)*cov2D(1,0), 1e-12);
    return Matx22d(cov2D(1,1)/det, -cov2D(0,1)/det, -cov2D(1,0)/det, cov2D(0,0)/det);
}

class PnPMsacCallback : public cv3::PointSetRegistrator::Callback
{

public:

    PnPMsacCallback(Mat _cameraMatrix=Mat(3,3,CV_64F), Mat _distCoeffs=Mat(4,1,CV_64F), 
            const std::vector<Matx33f>& _cov3D=std::vector<Matx33f>(), float _pixelVariance=1.0f, int _flags=CV_ITERATIVE,
            bool _useExtrinsicGuess=false, Mat _rvec=Mat(), Mat _tvec=Mat() )
        : cameraMatrix(_cameraMatrix), distCoeffs(_distCoeffs), cov3D(_cov3D),
          pixelVariance(_pixelVariance), flags(_flags), useExtrinsicGuess(_useExtrinsicGuess),
          rvec(_rvec), tvec(_tvec) {}

    /* Pre: True */
    /* Post: compute _model with given points an return number of found models */
    int runKernel( InputArray _m1, InputArray _m2, OutputArray _model ) const override
    {
        Mat opoints = _m1.getMat(), ipoints = _m2.getMat();

        bool correspondence = solvePnP( _m1, _m2, cameraMatrix, distCoeffs,
                                            rvec, tvec, useExtrinsicGuess, flags );

        Mat _local_model;
        hconcat(rvec, tvec, _local_model);
        _local_model.copyTo(_model);

        return correspondence;
    }

    /* Pre: True */
    /* Post: fill _err with projection errors */
    void computeError( InputArray _m1, InputArray _m2, InputArray _model, OutputArray _err ) const override
    {

        Mat opoints = _m1.getMat(), ipoints = _m2.getMat(), model = _model.getMat();

        int i, count = opoints.checkVector(3);
        Mat _rvec = model.col(0);
        Mat _tvec = model.col(1);


        Mat projpoints(count, 2, CV_32FC1);
        projectPoints(opoints, _rvec, _tvec, cameraMatrix, distCoeffs, projpoints);

        Mat R;
        Rodrigues(_rvec, R);
        Matx33d R33; R.convertTo(R33, CV_64F);
        Matx31d tvec31(_tvec.at<double>(0), _tvec.at<double>(1), _tvec.at<double>(2));

        double fx = cameraMatrix.at<double>(0,0);
        double fy = cameraMatrix.at<double>(1,1);

        const Point3f* pt3d = opoints.ptr<Point3f>();
        const Point2f* pt2d = ipoints.ptr<Point2f>();
        const Point2f* pproj = projpoints.ptr<Point2f>();

        _err.create(count, 1, CV_32FC1);
        float* err = _err.getMat().ptr<float>();

        for ( i = 0; i < count; ++i)
        {
            Matx22d cov2D_inv;
            if(!cov3D.empty())
            {
                cov2D_inv = computeWhiteningMatrix(pt3d[i], R33, tvec31, fx, fy, cov3D[i], pixelVariance);
            }
            else
            {
                cov2D_inv = (pixelVariance > 0.0f) ? Matx22d(1.0/pixelVariance, 0.0, 0.0, 1.0/pixelVariance) : Matx22d(1.0, 0.0, 0.0, 1.0);
            }

            double ex = pt2d[i].x - pproj[i].x;
            double ey = pt2d[i].y - pproj[i].y;
            
            err[i] = static_cast<float>(ex * ex * cov2D_inv(0,0) + 2.0 * ex * ey * cov2D_inv(0,1) + ey * ey * cov2D_inv(1,1));
        }

    }


    Mat cameraMatrix;
    Mat distCoeffs;
    std::vector<Matx33f> cov3D;
    float pixelVariance;
    int flags;
    bool useExtrinsicGuess;
    Mat rvec;
    Mat tvec;
};

bool solvePnPMsac(InputArray _opoints, InputArray _ipoints,
                  InputArray _cameraMatrix, InputArray _distCoeffs,
                  const std::vector<cv::Matx33f>& covariances3A,
                  OutputArray _rvec, OutputArray _tvec, bool useExtrinsicGuess,
                  int iterationsCount, float confidence, float pixelVariance,
                  bool use_prosac_ordering, OutputArray _inliers, int flags)
{

    Mat opoints0 = _opoints.getMat(), ipoints0 = _ipoints.getMat();
    Mat opoints, ipoints;
    if( opoints0.depth() == CV_64F || !opoints0.isContinuous() )
        opoints0.convertTo(opoints, CV_32F);
    else
        opoints = opoints0;
    if( ipoints0.depth() == CV_64F || !ipoints0.isContinuous() )
        ipoints0.convertTo(ipoints, CV_32F);
    else
        ipoints = ipoints0;

    int npoints = std::max(opoints.checkVector(3, CV_32F), opoints.checkVector(3, CV_64F));
    CV_Assert( npoints >= 0 && npoints == std::max(ipoints.checkVector(2, CV_32F), ipoints.checkVector(2, CV_64F)) );

    CV_Assert(opoints.isContinuous());
    CV_Assert(opoints.depth() == CV_32F || opoints.depth() == CV_64F);
    CV_Assert((opoints.rows == 1 && opoints.channels() == 3) || opoints.cols*opoints.channels() == 3);
    CV_Assert(ipoints.isContinuous());
    CV_Assert(ipoints.depth() == CV_32F || ipoints.depth() == CV_64F);
    CV_Assert((ipoints.rows == 1 && ipoints.channels() == 2) || ipoints.cols*ipoints.channels() == 2);

    Mat rvec = useExtrinsicGuess ? _rvec.getMat() : Mat(3, 1, CV_64FC1);
    Mat tvec = useExtrinsicGuess ? _tvec.getMat() : Mat(3, 1, CV_64FC1);
    Mat cameraMatrix = _cameraMatrix.getMat(), distCoeffs = _distCoeffs.getMat();

    int model_points = 6;
    int ransac_kernel_method = CV_EPNP;

    if( npoints == 4 )
    {
        model_points = 4;
        ransac_kernel_method = CV_P3P;
    }

    Ptr<cv3::PointSetRegistrator::Callback> cb; // pointer to callback
    cb = Ptr<PnPMsacCallback>(new PnPMsacCallback( cameraMatrix, distCoeffs, covariances3A, pixelVariance, ransac_kernel_method, useExtrinsicGuess, rvec, tvec));

    double param1 = CHI2_95_UNSQUARED;                // msac uses hardcoded chi2 thresholds
    double param2 = confidence;                       // confidence
    int param3 = iterationsCount;                     // number maximum iterations

    Mat _local_model(3, 2, CV_64FC1);
    Mat _mask_local_inliers(1, opoints.rows, CV_8UC1);

    // call MSAC
    int result = createMSACPointSetRegistrator(cb, model_points,
        param1, param2, param3, use_prosac_ordering)->run(opoints, ipoints, _local_model, _mask_local_inliers);

    if( result <= 0 || _local_model.rows <= 0)
    {
    	rvec.copyTo(_rvec);    // output rotation vector
        tvec.copyTo(_tvec);    // output translation vector

        if( _inliers.needed() )
            _inliers.release();

        return false;
    }
    else
    {
    	_local_model.col(0).copyTo(_rvec);    // output rotation vector
    	_local_model.col(1).copyTo(_tvec);    // output translation vector
    }

    if(_inliers.needed())
    {
        Mat _local_inliers;
        for (int i = 0; i < npoints; ++i)
        {
            if((int)_mask_local_inliers.at<uchar>(i) != 0) // inliers mask
                _local_inliers.push_back(i);    // output inliers vector
        }
        _local_inliers.copyTo(_inliers);
    }
    return true;
}


class MSACPointSetRegistrator : public cv3::PointSetRegistrator
{
public:
    MSACPointSetRegistrator(const Ptr<cv3::PointSetRegistrator::Callback>& _cb=Ptr<cv3::PointSetRegistrator::Callback>(),
                              int _modelPoints=0, double _threshold=0, double _confidence=0.99, int _maxIters=1000, bool _prosacOrdering=false)
    : cb(_cb), modelPoints(_modelPoints), threshold(_threshold), confidence(_confidence), maxIters(_maxIters), prosacOrdering(_prosacOrdering)
    {
        checkPartialSubsets = false;
    }

    int evaluateModel( const Mat& m1, const Mat& m2, const Mat& model, Mat& err, Mat& mask, double thresh, float& cost ) const
    {
        cb->computeError( m1, m2, model, err );
        mask.create(err.size(), CV_8U);

        CV_Assert( err.isContinuous() && err.type() == CV_32F && mask.isContinuous() && mask.type() == CV_8U);
        const float* errptr = err.ptr<float>();
        uchar* maskptr = mask.ptr<uchar>();
        float t = (float)(thresh*thresh);
        int i, n = (int)err.total(), nz = 0;
        cost = 0;
        for( i = 0; i < n; i++ )
        {
            float sq_err = errptr[i];
            int f = sq_err <= t;
            maskptr[i] = (uchar)f;
            nz += f;
            cost += f ? sq_err : t;
        }
        return nz;
    }

    bool getSubset( const Mat& m1, const Mat& m2,
                    Mat& ms1, Mat& ms2, RNG& rng,
                    int& n, double& t_n, int iter,
                    int maxAttempts=1000 ) const
    {
        cv::AutoBuffer<int> _idx(modelPoints);
        int* idx = _idx;
        int i = 0, j, k, iters = 0;
        int esz1 = (int)m1.elemSize(), esz2 = (int)m2.elemSize();
        int d1 = m1.channels() > 1 ? m1.channels() : m1.cols;
        int d2 = m2.channels() > 1 ? m2.channels() : m2.cols;
        int count = m1.checkVector(d1), count2 = m2.checkVector(d2);
        const int *m1ptr = m1.ptr<int>(), *m2ptr = m2.ptr<int>();

        ms1.create(modelPoints, 1, CV_MAKETYPE(m1.depth(), d1));
        ms2.create(modelPoints, 1, CV_MAKETYPE(m2.depth(), d2));

        int *ms1ptr = ms1.ptr<int>(), *ms2ptr = ms2.ptr<int>();

        CV_Assert( count >= modelPoints && count == count2 );
        CV_Assert( (esz1 % sizeof(int)) == 0 && (esz2 % sizeof(int)) == 0 );
        esz1 /= sizeof(int);
        esz2 /= sizeof(int);

        if (prosacOrdering && n < count && iter >= t_n)
        {
            n++;
            double delta_t = MAX(maxIters, 1) * static_cast<double>(modelPoints) / count;
            for (int k_t = 0; k_t < modelPoints - 1; ++k_t) {
                delta_t *= static_cast<double>(n - 1 - k_t) / (count - 1 - k_t);
            }
            t_n += delta_t;
        }

        for(; iters < maxAttempts; iters++)
        {
            for( i = 0; i < modelPoints && iters < maxAttempts; )
            {
                int idx_i = 0;
                for(;;)
                {
                    if (prosacOrdering && n < count) {
                        idx_i = idx[i] = (i == 0) ? n - 1 : rng.uniform(0, n - 1);
                    } else {
                        idx_i = idx[i] = rng.uniform(0, count);
                    }
                    
                    for( j = 0; j < i; j++ )
                        if( idx_i == idx[j] )
                            break;
                    if( j == i )
                        break;
                }
                for( k = 0; k < esz1; k++ )
                    ms1ptr[i*esz1 + k] = m1ptr[idx_i*esz1 + k];
                for( k = 0; k < esz2; k++ )
                    ms2ptr[i*esz2 + k] = m2ptr[idx_i*esz2 + k];
                if( checkPartialSubsets && !cb->checkSubset( ms1, ms2, i+1 ))
                {
                    // we may have selected some bad points;
                    // so, let's remove some of them randomly
                    i = rng.uniform(0, i+1);
                    iters++;
                    continue;
                }
                i++;
            }
            if( !checkPartialSubsets && i == modelPoints && !cb->checkSubset(ms1, ms2, i))
                continue;
            break;
        }

        return i == modelPoints && iters < maxAttempts;
    }

    bool run(InputArray _m1, InputArray _m2, OutputArray _model, OutputArray _mask) const override
    {
        bool result = false;
        Mat m1 = _m1.getMat(), m2 = _m2.getMat();
        Mat err, mask, model, bestModel, ms1, ms2;

        int iter, niters = MAX(maxIters, 1);
        int d1 = m1.channels() > 1 ? m1.channels() : m1.cols;
        int d2 = m2.channels() > 1 ? m2.channels() : m2.cols;
        int count = m1.checkVector(d1), count2 = m2.checkVector(d2), bestInliers = 0;
        float minCost = DBL_MAX;

        RNG rng((uint64)-1);

        CV_Assert( cb );
        CV_Assert( confidence > 0 && confidence < 1 );

        CV_Assert( count >= 0 && count2 == count );
        if( count < modelPoints )
            return false;

        Mat bestMask0, bestMask;

        if( _mask.needed() )
        {
            _mask.create(count, 1, CV_8U, -1, true);
            bestMask0 = bestMask = _mask.getMat();
            CV_Assert( (bestMask.cols == 1 || bestMask.rows == 1) && (int)bestMask.total() == count );
        }
        else
        {
            bestMask.create(count, 1, CV_8U);
            bestMask0 = bestMask;
        }

        if( count == modelPoints )
        {
            if( cb->runKernel(m1, m2, bestModel) <= 0 )
                return false;
            bestModel.copyTo(_model);
            bestMask.setTo(Scalar::all(1));
            return true;
        }

        int n = modelPoints;
        double t_n = 1.0;

        for( iter = 0; iter < niters; iter++ )
        {
            int i, currentInliers, nmodels;
            if( count > modelPoints )
            {
                bool found = getSubset( m1, m2, ms1, ms2, rng, n, t_n, iter, 10000 );
                if( !found )
                {
                    if( iter == 0 )
                        return false;
                    break;
                }
            }

            nmodels = cb->runKernel( ms1, ms2, model );
            if( nmodels <= 0 )
                continue;
            CV_Assert( model.rows % nmodels == 0 );
            Size modelSize(model.cols, model.rows/nmodels);

            for( i = 0; i < nmodels; i++ )
            {
                Mat model_i = model.rowRange( i*modelSize.height, (i+1)*modelSize.height );
                float currentCost = 0;
                currentInliers = evaluateModel( m1, m2, model_i, err, mask, threshold, currentCost );

                if( currentCost < minCost && currentInliers >= modelPoints )
                {
                    std::swap(mask, bestMask);
                    model_i.copyTo(bestModel);
                    minCost = currentCost;
                    bestInliers = currentInliers;
                    niters = cv3::RANSACUpdateNumIters( confidence, (double)(count - bestInliers)/count, modelPoints, niters );
                }
            }
        }

        if( minCost < DBL_MAX )
        {
            if( bestMask.data != bestMask0.data )
            {
                if( bestMask.size() == bestMask0.size() )
                    bestMask.copyTo(bestMask0);
                else
                    transpose(bestMask, bestMask0);
            }
            bestModel.copyTo(_model);
            result = true;
        }
        else
            _model.release();

        return result;
    }

    void setCallback(const Ptr<cv3::PointSetRegistrator::Callback>& _cb) override { cb = _cb; }

    Ptr<cv3::PointSetRegistrator::Callback> cb;
    int modelPoints;
    bool checkPartialSubsets;
    double threshold;
    double confidence;
    int maxIters;
    bool prosacOrdering;
};


Ptr<cv3::PointSetRegistrator> createMSACPointSetRegistrator(const Ptr<cv3::PointSetRegistrator::Callback>& _cb,
                                                         int _modelPoints, double _threshold,
                                                         double _confidence, int _maxIters, bool _prosacOrdering)
{
    return Ptr<cv3::PointSetRegistrator>(
        new MSACPointSetRegistrator(_cb, _modelPoints, _threshold, _confidence, _maxIters, _prosacOrdering));
}

// LM Optimizer Callback with Cholesky Weighting
class WeightedLMSolverCallback : public cv::LMSolver::Callback
{
public:
    WeightedLMSolverCallback(InputArray _opoints, InputArray _ipoints, const std::vector<Matx33f>& _cov3D, float pixelVariance,
                         Mat _K, Mat _D)
        : opoints(_opoints.getMat()), ipoints(_ipoints.getMat()), cov3D(_cov3D), pixelVariance(pixelVariance), K(_K), D(_D) {}

    bool compute(InputArray _param, OutputArray _err, OutputArray _J) const override
    {
        UASSERT(cov3D.empty() || (size_t)opoints.checkVector(3) == cov3D.size());

        Mat param = _param.getMat();
        Mat rvec = param.rowRange(0, 3);
        Mat tvec = param.rowRange(3, 6);

        Mat proj, jac_pose;
        projectPoints(opoints, rvec, tvec, K, D, proj, jac_pose);

        int N = opoints.checkVector(3);
        _err.create(N * 2, 1, CV_64F);
        _J.create(N * 2, 6, CV_64F);
        Mat err = _err.getMat(), J = _J.getMat();

        Mat R;
        Rodrigues(rvec, R);
        Matx33d R33; R.convertTo(R33, CV_64F);
        Matx31d tvec31(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));

        double fx = K.at<double>(0,0);
        double fy = K.at<double>(1,1);

        const Point3f* pt3d = opoints.ptr<Point3f>();
        const Point2f* pt2d = ipoints.ptr<Point2f>();
        const Point2f* pproj = proj.ptr<Point2f>();

        for(int i=0; i < N; i++) {
            Matx22d cov2D_inv;
            if(!cov3D.empty())
            {
                cov2D_inv = computeWhiteningMatrix(pt3d[i], R33, tvec31, fx, fy, cov3D[i], pixelVariance);
            }
            else
            {
                if(pixelVariance > 0.0f)
                {
                    cov2D_inv = Matx22d(1.0/pixelVariance, 0.0, 0.0, 1.0/pixelVariance);
                }
                else
                {
                    cov2D_inv = Matx22d(1.0, 0.0, 0.0, 1.0);
                }
            }

            double l00 = std::max(1e-6, std::sqrt(std::max(0.0, cov2D_inv(0,0))));
            double l10 = cov2D_inv(0,1) / l00;
            double l11 = std::sqrt(std::max(0.0, cov2D_inv(1,1) - l10*l10));
            Matx22d Lt(l00, l10, 0.0, l11); 

            Matx21d e(pt2d[i].x - pproj[i].x, pt2d[i].y - pproj[i].y);
            Matx21d we = Lt * e;
            err.at<double>(i*2, 0) = we(0);
            err.at<double>(i*2+1, 0) = we(1);

            Matx<double, 2, 6> J_p;
            for(int k=0; k<6; k++) {
                J_p(0, k) = jac_pose.at<double>(i*2, k);
                J_p(1, k) = jac_pose.at<double>(i*2+1, k);
            }
            Matx<double, 2, 6> wJ = Lt * J_p;
            for(int k=0; k<6; k++) {
                J.at<double>(i*2, k) = wJ(0, k);
                J.at<double>(i*2+1, k) = wJ(1, k);
            }
        }
        return true;
    }
    Mat opoints, ipoints;
    std::vector<Matx33f> cov3D;
    float pixelVariance;
    Mat K, D;
};

// Returns the linear (unsquared) Mahalanobis distance to match RANSAC's variance logic
std::vector<float> computeMahalanobisReprojErrors(
   const std::vector<cv::Point3f>& opoints,
   const std::vector<cv::Point2f>& ipoints,
   const cv::Mat & cameraMatrix,
   const cv::Mat & distCoeffs,
   const cv::Mat & rvec,
   const cv::Mat & tvec,
   const std::vector<cv::Matx33f> & cov3D,
   float pixelVariance,
   float unsquaredThreshold,
   std::vector<int> & inliers)
{
    int count = (int)opoints.size();
    std::vector<cv::Point2f> projpoints;
    projectPoints(opoints, rvec, tvec, cameraMatrix, distCoeffs, projpoints);

    Mat R;
    Rodrigues(rvec, R);
    Matx33d R33; R.convertTo(R33, CV_64F);
    Matx31d tvec31(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));

    double fx = cameraMatrix.at<double>(0,0);
    double fy = cameraMatrix.at<double>(1,1);

    inliers.resize(count, 0);
    std::vector<float> err(count);
    int oi = 0;

    // Squared threshold for actual numeric check
    double squaredThreshold = (double)unsquaredThreshold * (double)unsquaredThreshold;

    for (int i = 0; i < count; ++i)
    {
        Matx22d cov2D_inv;
        if(!cov3D.empty())
        {
            cov2D_inv = computeWhiteningMatrix(opoints[i], R33, tvec31, fx, fy, cov3D[i], pixelVariance);
        }
        else
        {
            if(pixelVariance > 0.0f)
            {
                cov2D_inv = Matx22d(1.0/pixelVariance, 0.0, 0.0, 1.0/pixelVariance);
            }
            else
            {
                cov2D_inv = Matx22d(1.0, 0.0, 0.0, 1.0);
            }
        }

        double ex = ipoints[i].x - projpoints[i].x;
        double ey = ipoints[i].y - projpoints[i].y;
        
        double sq_err = ex * ex * cov2D_inv(0,0) + 2.0 * ex * ey * cov2D_inv(0,1) + ey * ey * cov2D_inv(1,1);

        if(sq_err <= squaredThreshold)
        {
            inliers[oi] = i;
            // Return unsquared Mahalanobis distance to perfectly map RTABMap's linear variance logic
            err[oi++] = static_cast<float>(std::sqrt(sq_err)); 
        }
    }
    inliers.resize(oi);
    err.resize(oi);
    return err;
}

// Transforms a 6x6 pose covariance matrix of the form [rvec, tvec] (rvec in Rodrigues space)
// into format [x, y, z, roll, pitch, yaw] using closed-form analytical derivatives.
cv::Mat poseCovarianceRodriguesToRPY(const cv::Mat& rvec, 
                                        const cv::Mat& cov_rodriguez) 
{
    CV_Assert(cov_rodriguez.rows == 6 && cov_rodriguez.cols == 6);

    // Ensure double precision across all matrix operations
    cv::Mat cov64, rvec64;
    cov_rodriguez.convertTo(cov64, CV_64FC1);
    rvec.convertTo(rvec64, CV_64FC1);

    // Rodrigues Jacobian
    cv::Mat R, J_rodrigues;
    cv::Rodrigues(rvec64, R, J_rodrigues);

    if (J_rodrigues.rows == 3 && J_rodrigues.cols == 9) {
        J_rodrigues = J_rodrigues.t();
    }
    if (J_rodrigues.type() != CV_64FC1) {
        J_rodrigues.convertTo(J_rodrigues, CV_64FC1);
    }

    // RPY derivatives w.r.t rotation matrix R elements
    double r00 = R.at<double>(0, 0);
    double r10 = R.at<double>(1, 0);
    double r21 = R.at<double>(2, 1);
    double r22 = R.at<double>(2, 2);

    double roll_denom = r21 * r21 + r22 * r22;
    if (roll_denom < 1e-12) roll_denom = 1e-12;

    double pitch_denom2 = r00 * r00 + r10 * r10;
    if (pitch_denom2 < 1e-12) pitch_denom2 = 1e-12;
    double pitch_denom = std::sqrt(pitch_denom2);

    cv::Mat J_RPY_R = cv::Mat::zeros(3, 9, CV_64FC1);

    // Roll derivatives
    J_RPY_R.at<double>(0, 7) =  r22 / roll_denom; 
    J_RPY_R.at<double>(0, 8) = -r21 / roll_denom; 

    // Pitch derivative (SO(3) manifold simplification)
    J_RPY_R.at<double>(1, 6) = -1.0 / pitch_denom;  

    // Yaw derivatives
    J_RPY_R.at<double>(2, 0) = -r10 / pitch_denom2; 
    J_RPY_R.at<double>(2, 3) =  r00 / pitch_denom2; 

    // Chain rule
    cv::Mat J_rpy_rvec = J_RPY_R * J_rodrigues; 

    // Slice sub-blocks using OpenCV layout [rvec, tvec]
    cv::Mat cov_rvec      = cov64(cv::Range(0,3), cv::Range(0,3));
    cv::Mat cov_tvec      = cov64(cv::Range(3,6), cv::Range(3,6));
    cv::Mat cov_rvec_tvec = cov64(cv::Range(0,3), cv::Range(3,6));

    // Propagate rotation & cross-covariance terms
    cv::Mat cov_rpy      = J_rpy_rvec * cov_rvec * J_rpy_rvec.t();
    cv::Mat cov_tvec_rpy = cov_rvec_tvec.t() * J_rpy_rvec.t();

    // Reassemble into RTAB-Map layout: [x, y, z, roll, pitch, yaw]
    cv::Mat outCovariance6x6 = cv::Mat::zeros(6, 6, CV_64FC1);
    
    cov_tvec.copyTo(outCovariance6x6(cv::Range(0,3), cv::Range(0,3)));
    cov_rpy.copyTo(outCovariance6x6(cv::Range(3,6), cv::Range(3,6)));

    cov_tvec_rpy.copyTo(outCovariance6x6(cv::Range(0,3), cv::Range(3,6)));
    cv::Mat(cov_tvec_rpy.t()).copyTo(outCovariance6x6(cv::Range(3,6), cv::Range(0,3)));

    return outCovariance6x6;
}

// Extra RTABMap-like refinement loop for MSAC, including covariance propagation
void solvePnPMsac(const std::vector<cv::Point3f> & objectPoints,
                  const std::vector<cv::Point2f> & imagePoints,
                  const cv::Mat & cameraMatrix,
                  const cv::Mat & distCoeffs,
                  const std::vector<cv::Matx33f> & covariances3A,
                  cv::Mat & rvec, cv::Mat & tvec,
                  bool useExtrinsicGuess, int iterationsCount,
                  float reprojectionError, int minInliersCount,
                  float confidence, float pixelVariance,
                  std::vector<int> & inliers, int flags,
                  int refineIterations, float refineSigma,
                  bool use_prosac_ordering, cv::Mat & outPoseCovariance)
{
    if(minInliersCount < 4)
    {
        minInliersCount = 4;
    }

    UDEBUG("MSAC input points=%d useExtrinsicGuess=%d iterations=%d minInliers=%d flags=%d refineIterations=%d refineSigma=%f",
           (int)objectPoints.size(), useExtrinsicGuess, iterationsCount, minInliersCount, flags, refineIterations, refineSigma);

    // 1. Call the Core OpenCV-style function
    cv_custom::solvePnPMsac(
            objectPoints, imagePoints, cameraMatrix, distCoeffs, covariances3A,
            rvec, tvec, useExtrinsicGuess, iterationsCount, confidence,
            pixelVariance, use_prosac_ordering, inliers, flags);

    float inlierThreshold = CHI2_95_UNSQUARED;

    // 2. Exact mimic of the Refinement loop
    if((int)inliers.size() >= minInliersCount && refineIterations > 0)
    {
        float error_threshold = inlierThreshold;
        int refine_iterations = 0;
        bool inlier_changed = false, oscillating = false;
        std::vector<int> new_inliers, prev_inliers = inliers;
        std::vector<size_t> inliers_sizes;
        
        cv::Mat new_model_rvec = rvec.clone();
        cv::Mat new_model_tvec = tvec.clone();

        do
        {
            // Get inliers from the current model
            std::vector<cv::Point3f> opoints_inliers(prev_inliers.size());
            std::vector<cv::Point2f> ipoints_inliers(prev_inliers.size());
            std::vector<cv::Matx33f> cov_inliers(prev_inliers.size());
            for(unsigned int i = 0; i < prev_inliers.size(); ++i)
            {
                opoints_inliers[i] = objectPoints[prev_inliers[i]];
                ipoints_inliers[i] = imagePoints[prev_inliers[i]];
                cov_inliers[i] = covariances3A[prev_inliers[i]];
            }

            UDEBUG("inliers=%d refine_iterations=%d, rvec=%f,%f,%f tvec=%f,%f,%f", (int)prev_inliers.size(), refine_iterations,
                   *new_model_rvec.ptr<double>(0), *new_model_rvec.ptr<double>(1), *new_model_rvec.ptr<double>(2),
                   *new_model_tvec.ptr<double>(0), *new_model_tvec.ptr<double>(1), *new_model_tvec.ptr<double>(2));

            // Optimize the model coefficients
            cv::Mat param(6, 1, CV_64F);
            new_model_rvec.copyTo(param.rowRange(0, 3));
            new_model_tvec.copyTo(param.rowRange(3, 6));

            cv::Ptr<cv::LMSolver::Callback> lm_cb = cv::makePtr<WeightedLMSolverCallback>(
                opoints_inliers, ipoints_inliers, cov_inliers, pixelVariance, cameraMatrix, distCoeffs
            );

            cv::Ptr<cv::LMSolver> solver = cv::LMSolver::create(lm_cb, 20); 
            solver->run(param);

            param.rowRange(0, 3).copyTo(new_model_rvec);
            param.rowRange(3, 6).copyTo(new_model_tvec);
            
            inliers_sizes.push_back(prev_inliers.size());

            UDEBUG("rvec=%f,%f,%f tvec=%f,%f,%f",
                   *new_model_rvec.ptr<double>(0), *new_model_rvec.ptr<double>(1), *new_model_rvec.ptr<double>(2),
                   *new_model_tvec.ptr<double>(0), *new_model_tvec.ptr<double>(1), *new_model_tvec.ptr<double>(2));

            // Select the new inliers based on the optimized coefficients and new threshold
            std::vector<float> err = computeMahalanobisReprojErrors(
                objectPoints, imagePoints, cameraMatrix, distCoeffs, 
                new_model_rvec, new_model_tvec, covariances3A, pixelVariance, error_threshold, new_inliers
            );

            UDEBUG("MSAC refineModel: Number of inliers found (before/after): %d/%d, with an error threshold of %f.",
                   (int)prev_inliers.size (), (int)new_inliers.size (), error_threshold);

            if ((int)new_inliers.size() < minInliersCount)
            {
                ++refine_iterations;
                if (refine_iterations >= refineIterations)
                {
                    break;
                }
                continue;
            }

            // Estimate the variance and the new threshold
            float m = uMean(err.data(), err.size());
            float variance = uVariance(err.data(), err.size());
            error_threshold = std::min(inlierThreshold, refineSigma * float(sqrt(variance)));

            UDEBUG ("MSAC refineModel: New estimated error threshold: %f (variance=%f mean=%f) on iteration %d out of %d.",
                  error_threshold, variance, m, refine_iterations, refineIterations);
            
            inlier_changed = false;
            std::swap (prev_inliers, new_inliers);

            // If the number of inliers changed, then we are still optimizing
            if (new_inliers.size () != prev_inliers.size ())
            {
                // Check if the number of inliers is oscillating in between two values
                if ((int)inliers_sizes.size () >= minInliersCount)
                {
                    if (inliers_sizes[inliers_sizes.size () - 1] == inliers_sizes[inliers_sizes.size () - 3] &&
                        inliers_sizes[inliers_sizes.size () - 2] == inliers_sizes[inliers_sizes.size () - 4])
                    {
                        oscillating = true;
                        break;
                    }
                }
                inlier_changed = true;
                continue;
            }

            // Check the values of the inlier set
            for (size_t i = 0; i < prev_inliers.size (); ++i)
            {
                // If the value of the inliers changed, then we are still optimizing
                if (prev_inliers[i] != new_inliers[i])
                {
                    inlier_changed = true;
                    break;
                }
            }
        }
        while (inlier_changed && ++refine_iterations < refineIterations);

        // If the new set of inliers is empty, we didn't do a good job refining
        if ((int)prev_inliers.size() < minInliersCount)
        {
            UWARN ("MSAC refineModel: Refinement failed: got very low inliers (%d)!", (int)prev_inliers.size());
        }

        if (oscillating)
        {
            UDEBUG("MSAC refineModel: Detected oscillations in the model refinement.");
        }

        std::swap (inliers, new_inliers);
        rvec = new_model_rvec;
        tvec = new_model_tvec;
    }

    // 3. Covariance Extraction
    if ((int)inliers.size() >= minInliersCount)
    {
        std::vector<cv::Point3f> final_opoints;
        std::vector<cv::Point2f> final_ipoints;
        std::vector<cv::Matx33f> final_covs;
        for (int idx : inliers) {
            final_opoints.push_back(objectPoints[idx]);
            final_ipoints.push_back(imagePoints[idx]);
            final_covs.push_back(covariances3A[idx]);
        }

        cv::Ptr<cv::LMSolver::Callback> final_cb = cv::makePtr<WeightedLMSolverCallback>(
            final_opoints, final_ipoints, final_covs, pixelVariance, cameraMatrix, distCoeffs
        );

        cv::Mat final_err, final_J;
        cv::Mat param(6, 1, CV_64F);
        rvec.copyTo(param.rowRange(0, 3));
        tvec.copyTo(param.rowRange(3, 6));

        final_cb->compute(param, final_err, final_J);
        cv::Mat H = final_J.t() * final_J;
        cv::Mat cov_LM;
        cv::invert(H, cov_LM, cv::DECOMP_SVD);
        poseCovarianceRodriguesToRPY(rvec, cov_LM).copyTo(outPoseCovariance);
    }

}

} // namespace cv_custom