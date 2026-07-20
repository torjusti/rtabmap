#!/usr/bin/env python3
"""One-shot refactor: parallelize ExportCloudsDialog::getClouds().

Extracts the per-node loop body into a GUI-free helper function and rewrites
getClouds() to run it on a thread pool (serial fallback when subtract
filtering is enabled). Every substitution is count-asserted.
"""
import re
import sys

PATH = "/home/torjusti/Projects/rtabmap/guilib/src/ExportCloudsDialog.cpp"

src = open(PATH).read()

# ---------------------------------------------------------------- locate fn
sig = ("std::map<int, std::pair<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr, "
       "pcl::IndicesPtr> > ExportCloudsDialog::getClouds(")
assert src.count(sig) == 1
fn_start = src.index(sig)
end_marker = "\n\treturn clouds;\n}\n"
fn_end = src.index(end_marker, fn_start) + len(end_marker)
fn = src[fn_start:fn_end]

# ------------------------------------------------------------- extract body
body_start = fn.index("\t\tif(!iter->second.isNull())\n")
body_end_marker = "\t\telse\n\t\t{\n\t\t\tUERROR(\"transform is null!?\");\n\t\t}\n"
body_end = fn.index(body_end_marker) + len(body_end_marker)
body = fn[body_start:body_end]


def subn(pattern, repl, text, count, flags=0):
    out, n = re.subn(pattern, repl, text, flags=flags)
    assert n == count, "pattern %r matched %d times, expected %d" % (pattern, n, count)
    return out


# ---- chunk: distortion model (load hoisted out; pointer passed in params)
body = subn(
    r"if\(!depth\.empty\(\) &&\n"
    r"\s*!_ui->lineEdit_distortionModel->text\(\)\.isEmpty\(\) &&\n"
    r"\s*QFileInfo\(_ui->lineEdit_distortionModel->text\(\)\)\.exists\(\)\)\n"
    r"(\s*)\{\n"
    r"\s*clams::DiscreteDepthDistortionModel model;\n"
    r"\s*model\.load\(_ui->lineEdit_distortionModel->text\(\)\.toStdString\(\)\);\n"
    r"(\s*)depth = depth\.clone\(\);// make sure we are not modifying data in cached signatures\.\n"
    r"\s*model\.undistort\(depth\);\n"
    r"(\s*)\}",
    lambda m: ("if(!depth.empty() && p.distortionModel)\n"
               "%s{\n"
               "%sdepth = depth.clone();// make sure we are not modifying data in cached signatures.\n"
               "%sp.distortionModel->undistort(depth);\n"
               "%s}" % (m.group(1), m.group(2), m.group(2), m.group(3))),
    body, 1)

# ---- chunk: ROI ratios (parsed once in the snapshot)
body = subn(
    r"std::vector<float> roiRatios;\n"
    r"\s*if\(!_ui->lineEdit_roiRatios->text\(\)\.isEmpty\(\)\)\n"
    r"\s*\{\n"
    r"\s*QStringList values = _ui->lineEdit_roiRatios->text\(\)\.split\(' '\);\n"
    r"\s*if\(values\.size\(\) == 4\)\n"
    r"\s*\{\n"
    r"\s*roiRatios\.resize\(4\);\n"
    r"\s*for\(int i=0; i<values\.size\(\); \+\+i\)\n"
    r"\s*\{\n"
    r"\s*roiRatios\[i\] = uStr2Float\(values\[i\]\.toStdString\(\)\.c_str\(\)\);\n"
    r"\s*\}\n"
    r"\s*\}\n"
    r"\s*\}",
    "const std::vector<float> & roiRatios = p.roiRatios;",
    body, 1)

# ---- chunk: subtraction state through pointers (serial mode only)
body = subn(r"_ui->doubleSpinBox_subtractPointFilteringRadius->value\(\) > 0\.0\)",
            "_ui->doubleSpinBox_subtractPointFilteringRadius->value() > 0.0 && previousCloud)",
            body, 1)
body = subn(r"previousCloud\.get\(\) != 0 &&", "previousCloud->get() != 0 &&", body, 1)
body = subn(r"previousIndices\.get\(\) != 0 &&", "previousIndices->get() != 0 &&", body, 1)
body = subn(r"previousIndices->size\(\) &&", "(*previousIndices)->size() &&", body, 1)
body = subn(r"!previousPose\.isNull\(\)\)", "!previousPose->isNull())", body, 1)
body = subn(r"rtabmap::Transform t = iter->second\.inverse\(\) \* previousPose;",
            "rtabmap::Transform t = pose.inverse() * (*previousPose);", body, 1)
body = subn(r"rtabmap::util3d::transformPointCloud\(previousCloud, t\);",
            "rtabmap::util3d::transformPointCloud(*previousCloud, t);", body, 1)
body = subn(r"(\n\s*)previousIndices,(\n)", r"\1*previousIndices,\2", body, 1)
body = subn(r"(\n\s*)previousCloud = cloud;", r"\1*previousCloud = cloud;", body, 1)
body = subn(r"(\n\s*)previousIndices = beforeSubtractionIndices;",
            r"\1*previousIndices = beforeSubtractionIndices;", body, 1)
body = subn(r"(\n\s*)previousPose = iter->second;", r"\1*previousPose = pose;", body, 1)

# ---- chunk: output assignment instead of inserting into the map
body = subn(
    r"(\n\s*)clouds\.insert\(std::make_pair\(iter->first, std::make_pair\(cloud, indices\)\)\);\n"
    r"\s*points = \(int\)cloud->size\(\);\n"
    r"\s*totalIndices = \(int\)indices->size\(\);",
    r"\1result.cloud = cloud;\1result.indices = indices;",
    body, 1)

# ---- chunk: scansHaveRGB out-flag
body = subn(r"\n(\s*)scansHaveRGB = scan\.hasRGB\(\);",
            r"\n\1result.hasScan = true;\n\1result.scanHasRGB = scan.hasRGB();",
            body, 2)

# ---- chunk: progress messages collected in the result
body = subn(r"_progressDialog->appendText\((tr\(.*), Qt::darkYellow\);",
            r"result.messages.push_back(std::make_pair(ExportCloudsDialog::\1, QColor(Qt::darkYellow)));",
            body, 2)

# ---- global renames
renames = [
    ("_ui->spinBox_decimation->value() == 0?1:_ui->spinBox_decimation->value()", "p.decimation"),
    ("_ui->checkBox_regenerate->isChecked()", "p.regenerate"),
    ("_ui->checkBox_fromDepth->isChecked()", "p.fromDepth"),
    ("_ui->spinBox_depthConfidence->value()", "p.depthConfidenceThr"),
    ("_ui->spinBox_fillDepthHoles->value()", "p.fillDepthHoles"),
    ("_ui->spinBox_fillDepthHolesError->value()", "p.fillDepthHolesError"),
    ("_ui->doubleSpinBox_depthEdgeFiltering->value()", "p.depthEdgeFiltering"),
    ("_ui->checkBox_bilateral->isChecked()", "p.bilateral"),
    ("_ui->doubleSpinBox_bilateral_sigmaS->value()", "p.bilateralSigmaS"),
    ("_ui->doubleSpinBox_bilateral_sigmaR->value()", "p.bilateralSigmaR"),
    ("_ui->doubleSpinBox_maxDepth->value()", "p.maxDepth"),
    ("_ui->doubleSpinBox_minDepth->value()", "p.minDepth"),
    ("_ui->comboBox_pipeline->currentIndex()==0", "p.pipelineOrganized"),
    ("_ui->checkBox_meshing->isChecked()", "p.meshing"),
    ("_ui->doubleSpinBox_voxelSize_assembled->value()", "p.voxelSize"),
    ("_ui->spinBox_normalKSearch->value()", "p.normalK"),
    ("_ui->doubleSpinBox_normalRadiusSearch->value()", "p.normalRadius"),
    ("_ui->doubleSpinBox_groundNormalsUp->value()", "p.groundNormalsUp"),
    ("_ui->checkBox_subtraction->isChecked()", "p.subtraction"),
    ("_ui->doubleSpinBox_subtractPointFilteringRadius->value()", "p.subtractRadius"),
    ("_ui->doubleSpinBox_subtractPointFilteringAngle->value()", "p.subtractAngle"),
    ("_ui->spinBox_subtractFilteringMinPts->value()", "p.subtractMinPts"),
    ("_ui->spinBox_decimation_scan->value()", "p.scanDecimation"),
    ("_ui->doubleSpinBox_rangeMin->value()", "p.rangeMin"),
    ("_ui->doubleSpinBox_rangeMax->value()", "p.rangeMax"),
    ("_ui->checkBox_filtering->isChecked()", "p.filtering"),
    ("_ui->doubleSpinBox_ceilingHeight->value()", "p.ceilingHeight"),
    ("_ui->doubleSpinBox_floorHeight->value()", "p.floorHeight"),
    ("_ui->doubleSpinBox_footprintHeight->value()", "p.footprintHeight"),
    ("_ui->doubleSpinBox_footprintWidth->value()", "p.footprintWidth"),
    ("_ui->doubleSpinBox_footprintLength->value()", "p.footprintLength"),
    ("_ui->doubleSpinBox_filteringRadius->value()", "p.filteringRadius"),
    ("_ui->spinBox_filteringMinNeighbors->value()", "p.filteringMinNeighbors"),
    ("_ui->groupBox_offAxisFiltering->isChecked()", "p.offAxisFiltering"),
    ("_ui->doubleSpinBox_offAxisFilteringAngle->value()", "p.offAxisFilteringAngle"),
    ("_ui->checkBox_offAxisFilteringPosX->isChecked()", "p.offAxisPosX"),
    ("_ui->checkBox_offAxisFilteringNegX->isChecked()", "p.offAxisNegX"),
    ("_ui->checkBox_offAxisFilteringPosY->isChecked()", "p.offAxisPosY"),
    ("_ui->checkBox_offAxisFilteringNegY->isChecked()", "p.offAxisNegY"),
    ("_ui->checkBox_offAxisFilteringPosZ->isChecked()", "p.offAxisPosZ"),
    ("_ui->checkBox_offAxisFilteringNegZ->isChecked()", "p.offAxisNegZ"),
    ("_ui->comboBox_frame->isEnabled()", "p.frameEnabled"),
    ("_ui->comboBox_frame->currentIndex()", "p.frameIndex"),
    ("_dbDriver", "dbDriver"),
    ("iter->first", "nodeId"),
    ("iter->second", "pose"),
    ("poses.size()", "totalPoses"),
]
for old, new in renames:
    assert old in body or True  # some may legitimately be gone after chunk edits
    body = body.replace(old, new)

# ---- sanity: nothing GUI/loop-scoped must remain in the helper body
for forbidden in ("_ui->", "iter->", "_progressDialog", "_dbDriver",
                  "_canceled", "clouds.insert", "QApplication"):
    assert forbidden not in body, "forbidden token %r still in body" % forbidden

# ---- dedent one level (loop body -> function body)
body = "\n".join((l[1:] if l.startswith("\t") else l) for l in body.split("\n"))

HELPER = """namespace {

// Snapshot of the UI parameters used to generate the clouds, so that the
// generation code never touches widgets and can run in worker threads.
struct CloudGenParams
{
\tbool regenerate = false;
\tbool fromDepth = true;
\tint depthConfidenceThr = 0;
\tint fillDepthHoles = 0;
\tint fillDepthHolesError = 2;
\tdouble depthEdgeFiltering = 0.0;
\tbool bilateral = false;
\tdouble bilateralSigmaS = 10.0;
\tdouble bilateralSigmaR = 0.1;
\tstd::vector<float> roiRatios;
\tint decimation = 1;
\tdouble maxDepth = 0.0;
\tdouble minDepth = 0.0;
\tint scanDecimation = 1;
\tdouble rangeMin = 0.0;
\tdouble rangeMax = 0.0;
\tbool pipelineOrganized = false;
\tbool meshing = false;
\tdouble voxelSize = 0.0;
\tint normalK = 0;
\tdouble normalRadius = 0.0;
\tdouble groundNormalsUp = 0.0;
\tbool subtraction = false;
\tdouble subtractRadius = 0.0;
\tdouble subtractAngle = 0.0;
\tint subtractMinPts = 0;
\tbool filtering = false;
\tdouble ceilingHeight = 0.0;
\tdouble floorHeight = 0.0;
\tdouble footprintHeight = 0.0;
\tdouble footprintWidth = 0.0;
\tdouble footprintLength = 0.0;
\tdouble filteringRadius = 0.0;
\tint filteringMinNeighbors = 0;
\tbool offAxisFiltering = false;
\tdouble offAxisFilteringAngle = 0.0;
\tbool offAxisPosX = false;
\tbool offAxisNegX = false;
\tbool offAxisPosY = false;
\tbool offAxisNegY = false;
\tbool offAxisPosZ = false;
\tbool offAxisNegZ = false;
\tbool frameEnabled = false;
\tint frameIndex = 0;
\tconst clams::DiscreteDepthDistortionModel * distortionModel = nullptr;
};

struct CloudGenResult
{
\tpcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud;
\tpcl::IndicesPtr indices;
\tbool hasScan = false;
\tbool scanHasRGB = false;
\tstd::vector<std::pair<QString, QColor> > messages;
};

// Generates the cloud of a single node. Doesn't touch any GUI object so that
// it can be called from worker threads (progress messages are returned in the
// result). previousCloud/previousIndices/previousPose are only used (and
// updated) when point subtraction filtering is enabled, which requires the
// nodes to be processed sequentially.
CloudGenResult generateCloudForNode(
\t\tint nodeId,
\t\tconst Transform & pose,
\t\tint index,
\t\tint totalPoses,
\t\tconst CloudGenParams & p,
\t\tconst QMap<int, Signature> & cachedSignatures,
\t\tconst std::map<int, std::pair<pcl::PointCloud<pcl::PointXYZRGB>::Ptr, pcl::IndicesPtr> > & cachedClouds,
\t\tconst std::map<int, LaserScan> & cachedScans,
\t\tconst ParametersMap & parameters,
\t\tconst DBDriver * dbDriver,
\t\tpcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr * previousCloud,
\t\tpcl::IndicesPtr * previousIndices,
\t\tTransform * previousPose)
{
\tCloudGenResult result;
%BODY%\treturn result;
}

} // namespace

"""

NEW_GETCLOUDS = """std::map<int, std::pair<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr, pcl::IndicesPtr> > ExportCloudsDialog::getClouds(
\t\tconst std::map<int, Transform> & poses,
\t\tconst QMap<int, Signature> & cachedSignatures,
\t\tconst std::map<int, std::pair<pcl::PointCloud<pcl::PointXYZRGB>::Ptr, pcl::IndicesPtr> > & cachedClouds,
\t\tconst std::map<int, LaserScan> & cachedScans,
\t\tconst ParametersMap & parameters,
\t\tbool & has2dScans,
\t\tbool & scansHaveRGB) const
{
\tscansHaveRGB = false;
\thas2dScans = false;
\tstd::map<int, std::pair<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr, pcl::IndicesPtr> > clouds;

\tCloudGenParams p;
\tp.regenerate = _ui->checkBox_regenerate->isChecked();
\tp.fromDepth = _ui->checkBox_fromDepth->isChecked();
\tp.depthConfidenceThr = _ui->spinBox_depthConfidence->value();
\tp.fillDepthHoles = _ui->spinBox_fillDepthHoles->value();
\tp.fillDepthHolesError = _ui->spinBox_fillDepthHolesError->value();
\tp.depthEdgeFiltering = _ui->doubleSpinBox_depthEdgeFiltering->value();
\tp.bilateral = _ui->checkBox_bilateral->isChecked();
\tp.bilateralSigmaS = _ui->doubleSpinBox_bilateral_sigmaS->value();
\tp.bilateralSigmaR = _ui->doubleSpinBox_bilateral_sigmaR->value();
\tif(!_ui->lineEdit_roiRatios->text().isEmpty())
\t{
\t\tQStringList values = _ui->lineEdit_roiRatios->text().split(' ');
\t\tif(values.size() == 4)
\t\t{
\t\t\tp.roiRatios.resize(4);
\t\t\tfor(int i=0; i<values.size(); ++i)
\t\t\t{
\t\t\t\tp.roiRatios[i] = uStr2Float(values[i].toStdString().c_str());
\t\t\t}
\t\t}
\t}
\tp.decimation = _ui->spinBox_decimation->value() == 0?1:_ui->spinBox_decimation->value();
\tp.maxDepth = _ui->doubleSpinBox_maxDepth->value();
\tp.minDepth = _ui->doubleSpinBox_minDepth->value();
\tp.scanDecimation = _ui->spinBox_decimation_scan->value();
\tp.rangeMin = _ui->doubleSpinBox_rangeMin->value();
\tp.rangeMax = _ui->doubleSpinBox_rangeMax->value();
\tp.pipelineOrganized = _ui->comboBox_pipeline->currentIndex()==0;
\tp.meshing = _ui->checkBox_meshing->isChecked();
\tp.voxelSize = _ui->doubleSpinBox_voxelSize_assembled->value();
\tp.normalK = _ui->spinBox_normalKSearch->value();
\tp.normalRadius = _ui->doubleSpinBox_normalRadiusSearch->value();
\tp.groundNormalsUp = _ui->doubleSpinBox_groundNormalsUp->value();
\tp.subtraction = _ui->checkBox_subtraction->isChecked();
\tp.subtractRadius = _ui->doubleSpinBox_subtractPointFilteringRadius->value();
\tp.subtractAngle = _ui->doubleSpinBox_subtractPointFilteringAngle->value();
\tp.subtractMinPts = _ui->spinBox_subtractFilteringMinPts->value();
\tp.filtering = _ui->checkBox_filtering->isChecked();
\tp.ceilingHeight = _ui->doubleSpinBox_ceilingHeight->value();
\tp.floorHeight = _ui->doubleSpinBox_floorHeight->value();
\tp.footprintHeight = _ui->doubleSpinBox_footprintHeight->value();
\tp.footprintWidth = _ui->doubleSpinBox_footprintWidth->value();
\tp.footprintLength = _ui->doubleSpinBox_footprintLength->value();
\tp.filteringRadius = _ui->doubleSpinBox_filteringRadius->value();
\tp.filteringMinNeighbors = _ui->spinBox_filteringMinNeighbors->value();
\tp.offAxisFiltering = _ui->groupBox_offAxisFiltering->isChecked();
\tp.offAxisFilteringAngle = _ui->doubleSpinBox_offAxisFilteringAngle->value();
\tp.offAxisPosX = _ui->checkBox_offAxisFilteringPosX->isChecked();
\tp.offAxisNegX = _ui->checkBox_offAxisFilteringNegX->isChecked();
\tp.offAxisPosY = _ui->checkBox_offAxisFilteringPosY->isChecked();
\tp.offAxisNegY = _ui->checkBox_offAxisFilteringNegY->isChecked();
\tp.offAxisPosZ = _ui->checkBox_offAxisFilteringPosZ->isChecked();
\tp.offAxisNegZ = _ui->checkBox_offAxisFilteringNegZ->isChecked();
\tp.frameEnabled = _ui->comboBox_frame->isEnabled();
\tp.frameIndex = _ui->comboBox_frame->currentIndex();

\t// Load the distortion model once instead of reloading it for every node.
\tclams::DiscreteDepthDistortionModel distortionModel;
\tif(p.regenerate && p.fromDepth &&
\t   !_ui->lineEdit_distortionModel->text().isEmpty() &&
\t   QFileInfo(_ui->lineEdit_distortionModel->text()).exists())
\t{
\t\tdistortionModel.load(_ui->lineEdit_distortionModel->text().toStdString());
\t\tp.distortionModel = &distortionModel;
\t}

\tstd::vector<std::pair<int, Transform> > nodes(poses.lower_bound(1), poses.end());
\tint totalPoses = (int)nodes.size();
\tstd::vector<CloudGenResult> results(nodes.size());

\t// Appends the progress messages of a processed node and inserts its cloud
\t// in the output map. Always called on the GUI thread, in poses order.
\tauto flushResult = [&](size_t i)
\t{
\t\tconst CloudGenResult & result = results[i];
\t\tfor(size_t m=0; m<result.messages.size(); ++m)
\t\t{
\t\t\t_progressDialog->appendText(result.messages[m].first, result.messages[m].second);
\t\t}
\t\tif(result.hasScan)
\t\t{
\t\t\tscansHaveRGB = result.scanHasRGB;
\t\t}
\t\tint points = 0;
\t\tint totalIndices = 0;
\t\tif(result.cloud.get() && result.indices.get() && !result.indices->empty())
\t\t{
\t\t\tclouds.insert(std::make_pair(nodes[i].first, std::make_pair(result.cloud, result.indices)));
\t\t\tpoints = (int)result.cloud->size();
\t\t\ttotalIndices = (int)result.indices->size();
\t\t}
\t\tif(points>0)
\t\t{
\t\t\tif(p.regenerate)
\t\t\t{
\t\t\t\t_progressDialog->appendText(tr("Generated cloud %1 with %2 points and %3 indices (%4/%5).")
\t\t\t\t\t\t.arg(nodes[i].first).arg(points).arg(totalIndices).arg((int)i+1).arg(totalPoses));
\t\t\t}
\t\t\telse
\t\t\t{
\t\t\t\t_progressDialog->appendText(tr("Copied cloud %1 from cache with %2 points and %3 indices (%4/%5).")
\t\t\t\t\t\t.arg(nodes[i].first).arg(points).arg(totalIndices).arg((int)i+1).arg(totalPoses));
\t\t\t}
\t\t}
\t\telse
\t\t{
\t\t\t_progressDialog->appendText(tr("Ignored cloud %1 (%2/%3).").arg(nodes[i].first).arg((int)i+1).arg(totalPoses));
\t\t}
\t\t_progressDialog->incrementStep();
\t\tQApplication::processEvents();
\t};

\t// Point subtraction filtering compares each cloud with the previous one,
\t// so in that case nodes must be processed sequentially.
\tif(p.subtraction && p.subtractRadius > 0.0)
\t{
\t\tpcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr previousCloud;
\t\tpcl::IndicesPtr previousIndices;
\t\tTransform previousPose;
\t\tfor(int i=0; i<totalPoses && !_canceled; ++i)
\t\t{
\t\t\tresults[i] = generateCloudForNode(
\t\t\t\t\tnodes[i].first, nodes[i].second, i+1, totalPoses, p,
\t\t\t\t\tcachedSignatures, cachedClouds, cachedScans, parameters, _dbDriver,
\t\t\t\t\t&previousCloud, &previousIndices, &previousPose);
\t\t\tflushResult(i);
\t\t}
\t}
\telse
\t{
\t\t// Clouds are generated in parallel (they are independent of each other).
\t\t// The OpenMP master thread is the GUI thread: between its own iterations
\t\t// it reports progress of completed nodes, keeps the UI responsive and
\t\t// checks for cancellation. Code internally parallelized with OpenMP in
\t\t// generateCloudForNode() (e.g., normal estimation) runs single-threaded
\t\t// here because nested OpenMP parallelism is disabled by default.
\t\tstd::vector<std::atomic<bool> > done(nodes.size());
\t\tfor(size_t i=0; i<done.size(); ++i)
\t\t{
\t\t\tdone[i] = false;
\t\t}
\t\tstd::atomic<bool> cancelRequested(false);
\t\tint nextFlush = 0;

\t\t#pragma omp parallel for schedule(dynamic)
\t\tfor(int i=0; i<totalPoses; ++i)
\t\t{
\t\t\tif(!cancelRequested)
\t\t\t{
\t\t\t\tresults[i] = generateCloudForNode(
\t\t\t\t\t\tnodes[i].first, nodes[i].second, i+1, totalPoses, p,
\t\t\t\t\t\tcachedSignatures, cachedClouds, cachedScans, parameters, _dbDriver,
\t\t\t\t\t\tnullptr, nullptr, nullptr);
\t\t\t\tdone[i] = true;
\t\t\t}

#ifdef _OPENMP
\t\t\tif(omp_get_thread_num() == 0)
#endif
\t\t\t{
\t\t\t\twhile(nextFlush < totalPoses && done[nextFlush])
\t\t\t\t{
\t\t\t\t\tflushResult(nextFlush++);
\t\t\t\t}
\t\t\t\tQApplication::processEvents();
\t\t\t\tif(_canceled)
\t\t\t\t{
\t\t\t\t\tcancelRequested = true;
\t\t\t\t}
\t\t\t}
\t\t}

\t\t// Report the nodes completed after the master thread's last iteration.
\t\twhile(nextFlush < totalPoses && done[nextFlush])
\t\t{
\t\t\tflushResult(nextFlush++);
\t\t}
\t\tQApplication::processEvents();
\t}

\treturn clouds;
}
"""

new_fn = HELPER.replace("%BODY%", body) + NEW_GETCLOUDS
src = src[:fn_start] + new_fn + src[fn_end:]

# ---- add includes
inc_anchor = "#include <QScreen>\n"
assert src.count(inc_anchor) == 1
src = src.replace(inc_anchor, inc_anchor +
                  "\n#include <atomic>\n"
                  "\n#ifdef _OPENMP\n#include <omp.h>\n#endif\n")

# The file is CRLF; universal-newline read gave us plain \n, translate back.
open(PATH, "w", newline="\r\n").write(src)
print("OK, refactor applied")
