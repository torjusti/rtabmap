/*
Copyright (c) 2010-2016, Mathieu Labbe - IntRoLab - Universite de Sherbrooke
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

#ifndef GUILIB_SRC_NODECLOUDCACHE_H_
#define GUILIB_SRC_NODECLOUDCACHE_H_

#include <QString>
#include <QDateTime>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <rtabmap/core/Transform.h>
#include <map>
#include <set>

namespace rtabmap {

/**
 * Per-node clouds of a database, kept on disk so that viewing the 3D map again does
 * not read every node's images from the database and project them again.
 *
 * One file per database and set of cloud settings, in the user's cache directory (so
 * on the local disk, whatever the disk of the database). Each cloud is in the frame
 * of its node, so it does not depend on the poses: it can be placed with whatever
 * graph is optimized when the file is read. A file can hold any subset of the nodes
 * and grows as more nodes are saved.
 *
 * Layout (little endian):
 *   header   "RTBNCLD1", u32 version, u32 flags (1: normals, 2: intensity colors),
 *            u64 offset of the index, u32 node count, u32 key length, i64 creation time
 *   key      the database path and a digest of the cloud settings; a file whose key
 *            differs from the one asked for is ignored
 *   records  i32 node id, u32 point count, 12 x f32 local transform (3x4, row major),
 *            then per point: 3 x f32 xyz, 4 x u8 rgba, [3 x f32 normal]
 *   index    per record: i32 node id, u32 point count, u64 offset of the record
 */
class NodeCloudCache
{
public:
	struct Entry
	{
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud; // in the frame of the node
		Transform localTransform; // sensor in the robot frame, used as viewpoint of the normals
	};

	struct Summary
	{
		Summary() : valid(false), withNormals(false), useIntensity(false), bytes(0) {}
		bool valid;
		QDateTime created;
		bool withNormals;
		bool useIntensity;
		qint64 bytes;
		std::set<int> ids;
	};

	// The key of a database and the digest of its cloud settings
	static QString makeKey(const QString & databasePath, const QString & settingsDigest);
	static QString filePath(const QString & key);

	// Header and index of the file of this key (invalid if there is none, or it cannot be trusted)
	static Summary summary(const QString & key);

	// Reads the wanted nodes that the file holds
	static bool load(const QString & key, const std::set<int> & wanted, std::map<int, Entry> & entries, QString * error = 0);

	// Adds the entries to the file of this key (replacing nodes already in it)
	static bool save(const QString & key, bool withNormals, bool useIntensity, const std::map<int, Entry> & entries, QString * error = 0);

	static bool remove(const QString & key);
};

}

#endif /* GUILIB_SRC_NODECLOUDCACHE_H_ */
