#!/usr/bin/env python3
"""One-shot refactor: parallelize the per-node loop of rtabmap-export.

Splits the loop body of tools/Export/main.cpp into:
  - loadNode(): db read + decompression + cloud creation/filtering (thread-safe)
  - flushNode(): accumulation into shared state, sequential in poses order
and drives them by chunks with an OpenMP parallel for.

Every anchor/rename is count-asserted; original code is transplanted verbatim.
"""
import sys

PATH = "/home/torjusti/Projects/rtabmap/tools/Export/main.cpp"

src = open(PATH).read()  # universal newlines: CRLF -> \n


def index_once(hay, needle):
    assert hay.count(needle) == 1, "anchor not unique or missing: %r" % needle[:80]
    return hay.index(needle)


def rename(text, old, new, expected=None):
    n = text.count(old)
    if expected is not None:
        assert n == expected, "%r found %d times, expected %d" % (old, n, expected)
    else:
        assert n > 0, "%r not found" % old
    return text.replace(old, new)


# ------------------------------------------------------------- locate region
LOOP_START = ("\tint processedNodes = 0;\n"
              "\tint lastPercent = 0;\n"
              "\tfor(std::map<int, Transform>::iterator iter=optimizedPoses.begin(); iter!=optimizedPoses.end(); ++iter)\n"
              "\t{\n")
SENTINEL = ("\tif(exportCloud || exportMesh)\n"
            "\t{\n"
            "\t\tprintf(\"Create and assemble the clouds... done")
start = index_once(src, LOOP_START)
end = index_once(src, SENTINEL)
region = src[start:end]

# ------------------------------------------------------------------- slices
A0 = "\t\tTransform p, gt;\n"
B0 = "\t\tif(exportCloud || exportMesh || exportImages)\n"
C0 = "\t\t\t\tif(cloudFromScan)\n"
C1 = "\t\t\t\t\tcameraDepths.insert(std::make_pair(iter->first, depth));\n\t\t\t\t}\n"
D0 = "\t\tif(models.empty())\n"
E0 = "\t\tif(weight != -1 && (export2DMap || exportOctomap)) {\n"
E1 = "\t\tif(optimizedPoses.size() >= 500)\n"

a = region[index_once(region, A0):index_once(region, B0)]
b = region[index_once(region, B0):index_once(region, C0)]
c = region[index_once(region, C0):index_once(region, C1) + len(C1)]
d = region[index_once(region, D0):index_once(region, E0)]
e = region[index_once(region, E0):index_once(region, E1)]

# --------------------------------------------------------- slice A -> worker
a = rename(a, "\t\tTransform p, gt;\n",
              "\t\tTransform p;\n\t\tTransform & gt = out.gt;\n", 1)
a = rename(a, "\t\tGPS gps;\n", "\t\tGPS & gps = out.gps;\n", 1)
a = rename(a, "\t\tint weight = -1;\n", "\t\tint & weight = out.weight;\n", 1)
a = rename(a, "\t\tdouble stamp = 0.0;\n", "\t\tdouble & stamp = out.stamp;\n", 1)
a = rename(a, "\t\tSensorData data;\n", "\t\tSensorData & data = out.data;\n", 1)
a = rename(a, "\t\tstd::vector<CameraModel> models;\n",
              "\t\tstd::vector<CameraModel> & models = out.models;\n", 1)
a = rename(a, "\t\tstd::vector<StereoCameraModel> stereoModels;\n",
              "\t\tstd::vector<StereoCameraModel> & stereoModels = out.stereoModels;\n", 1)
a = rename(a, "iter->first", "nodeId", 3)  # getNodeInfo, getNodeData, getCalibration

# --------------------------------------------------------- slice B -> worker
b = rename(b, "\t\t\tcv::Mat depth;\n", "\t\t\tcv::Mat & depth = out.depth;\n", 1)
b = rename(b, "\t\t\tpcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;\n",
              "\t\t\tpcl::PointCloud<pcl::PointXYZRGB>::Ptr & cloud = out.cloud;\n", 1)
b = rename(b, "\t\t\tpcl::PointCloud<pcl::PointXYZI>::Ptr cloudI;\n",
              "\t\t\tpcl::PointCloud<pcl::PointXYZI>::Ptr & cloudI = out.cloudI;\n", 1)
b = rename(b, "\t\t\t\t++imagesExported;\n", "\t\t\t\tout.imageExported = true;\n", 1)
b = rename(b, "iter->first", "nodeId")
b = rename(b, "iter->second", "pose")
# close if(exportCloud || exportMesh) and if(exportCloud || exportMesh || exportImages)
b += "\t\t\t}\n\t\t}\n"

# ---------------------------------------------------------- slice C -> flush
c = rename(c, "\t\t\t\t\trawViewpoints.insert(*iter);\n",
              "\t\t\t\t\trawViewpoints.insert(std::make_pair(nodeId, pose));\n", 1)
c = rename(c, "iter->first", "nodeId")
c = rename(c, "iter->second", "pose")
# dedent one level: was nested in two ifs, now in one
c = "\n".join((l[1:] if l.startswith("\t") else l) for l in c.split("\n"))
c = "\t\tif(exportCloud || exportMesh)\n\t\t{\n" + c + "\t\t}\n"

# ---------------------------------------------------------- slice D -> flush
d = rename(d, "iter->first", "nodeId")
d = rename(d, "iter->second", "pose")

# ---------------------------------------------------------- slice E -> flush
e = rename(e, ("\t\t\tcv::Mat ground;\n"
               "\t\t\tcv::Mat obstacles;\n"
               "\t\t\tcv::Mat empty;\n"
               "\t\t\tdata.uncompressDataConst(0, 0, 0, 0, &ground, &obstacles, &empty);\n"),
              ("\t\t\tcv::Mat & ground = node.groundCells;\n"
               "\t\t\tcv::Mat & obstacles = node.obstacleCells;\n"
               "\t\t\tcv::Mat & empty = node.emptyCells;\n"), 1)
e = rename(e, "\t\t\t\taddedPosesToMap.insert(*iter);\n",
              "\t\t\t\taddedPosesToMap.insert(std::make_pair(nodeId, pose));\n", 1)
e = rename(e, "iter->first", "nodeId")

for name, text in (("a", a), ("b", b), ("c", c), ("d", d), ("e", e)):
    assert "iter" not in text, "leftover iter in slice %s" % name

# ------------------------------------------------------------- new region
NEW = """\tint processedNodes = 0;
\tint lastPercent = 0;

\t// Landmarks don't have node data, just add them to the pose lists.
\tstd::vector<std::pair<int, Transform> > nodes;
\tnodes.reserve(optimizedPoses.size());
\tfor(std::map<int, Transform>::iterator iter=optimizedPoses.begin(); iter!=optimizedPoses.end(); ++iter)
\t{
\t\tif(iter->first<0)
\t\t{
\t\t\trobotPoses.insert(*iter);
\t\t\trobotStamps.insert(std::make_pair(iter->first, 0));

\t\t\tlandmarkPoses.insert(*iter);
\t\t\tlandmarkStamps.insert(std::make_pair(iter->first, 0));
\t\t}
\t\telse
\t\t{
\t\t\tnodes.push_back(*iter);
\t\t}
\t}

\t// Per-node data loaded and preprocessed by loadNode(), which is thread-safe
\t// (db access is mutex-protected). Accumulation into shared state is done
\t// sequentially in poses order by flushNode() to keep the exact same
\t// output as when everything was done in a single sequential loop.
\tstruct NodeExportData
\t{
\t\tint weight = -1;
\t\tdouble stamp = 0.0;
\t\tTransform gt;
\t\tGPS gps;
\t\tSensorData data;
\t\tstd::vector<CameraModel> models;
\t\tstd::vector<StereoCameraModel> stereoModels;
\t\tcv::Mat depth;
\t\tpcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
\t\tpcl::PointCloud<pcl::PointXYZI>::Ptr cloudI;
\t\tbool imageExported = false;
\t\tcv::Mat groundCells, obstacleCells, emptyCells;
\t};

\tauto loadNode = [&](int nodeId, const Transform & pose, NodeExportData & out)
\t{
%A%%B%
\t\tif(out.weight != -1 && (export2DMap || exportOctomap))
\t\t{
\t\t\tout.data.uncompressDataConst(0, 0, 0, 0, &out.groundCells, &out.obstacleCells, &out.emptyCells);
\t\t}
\t};

\tauto flushNode = [&](int nodeId, const Transform & pose, NodeExportData & node)
\t{
\t\tSensorData & data = node.data;
\t\tstd::vector<CameraModel> & models = node.models;
\t\tstd::vector<StereoCameraModel> & stereoModels = node.stereoModels;
\t\tpcl::PointCloud<pcl::PointXYZRGB>::Ptr & cloud = node.cloud;
\t\tpcl::PointCloud<pcl::PointXYZI>::Ptr & cloudI = node.cloudI;
\t\tcv::Mat & depth = node.depth;
\t\tdouble stamp = node.stamp;
\t\tint weight = node.weight;
\t\tGPS & gps = node.gps;
\t\tTransform & gt = node.gt;

\t\tif(node.imageExported)
\t\t{
\t\t\t++imagesExported;
\t\t}

%C%
%D%%E%\t};

\t// Preprocess the nodes in parallel by chunks (chunking bounds the memory
\t// used by pending preprocessed clouds), then accumulate them in order.
\tconst size_t chunkSize = 128;
\tstd::vector<NodeExportData> chunkData;
\tfor(size_t chunkStart=0; chunkStart<nodes.size(); chunkStart+=chunkSize)
\t{
\t\tsize_t chunkNodes = std::min(chunkSize, nodes.size()-chunkStart);
\t\tchunkData.assign(chunkNodes, NodeExportData());

\t\t#pragma omp parallel for schedule(dynamic)
\t\tfor(int i=0; i<(int)chunkNodes; ++i)
\t\t{
\t\t\tloadNode(nodes[chunkStart+i].first, nodes[chunkStart+i].second, chunkData[i]);
\t\t}

\t\tfor(size_t i=0; i<chunkNodes; ++i)
\t\t{
\t\t\tflushNode(nodes[chunkStart+i].first, nodes[chunkStart+i].second, chunkData[i]);

\t\t\tif(optimizedPoses.size() >= 500)
\t\t\t{
\t\t\t\t++processedNodes;
\t\t\t\tint percent = processedNodes*100/(int)optimizedPoses.size();
\t\t\t\tif(percent != lastPercent)
\t\t\t\t{
\t\t\t\t\tprintf("Processed %d/%d (%d%%) nodes...\\n",
\t\t\t\t\t\tprocessedNodes,
\t\t\t\t\t\t(int)optimizedPoses.size(),
\t\t\t\t\t\tpercent);
\t\t\t\t\tlastPercent = percent;
\t\t\t\t}
\t\t\t}
\t\t}
\t}
"""

NEW = (NEW.replace("%A%", a).replace("%B%", b)
          .replace("%C%", c).replace("%D%", d).replace("%E%", e))

src = src[:start] + NEW + src[end:]

# <algorithm> for std::min (likely transitively included, but be explicit)
inc = "#include <stdio.h>\n"
assert src.count(inc) == 1
src = src.replace(inc, inc + "#include <algorithm>\n")

open(PATH, "w", newline="\r\n").write(src)
print("OK, refactor applied")
