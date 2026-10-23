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

#include "NodeCloudCache.h"

#include <QFile>
#include <QFileInfo>
#include <QDir>
#include <QRegExp>
#include <QStandardPaths>
#include <QCryptographicHash>
#include <QByteArray>
#include <QtGlobal>
#include <algorithm>
#include <cstring>
#include <vector>

namespace rtabmap {

namespace {

const char kMagic[8] = {'R','T','B','N','C','L','D','1'};
const quint32 kVersion = 1;
const quint32 kFlagNormals = 1;
const quint32 kFlagIntensity = 2;
const qint64 kHeaderSize = 40; // fixed part, the key follows
const qint64 kRecordHeaderSize = 8 + 12*4;
const qint64 kIndexItemSize = 16;

struct Header
{
	Header() : flags(0), indexOffset(0), nodeCount(0), keyLength(0), created(0) {}
	quint32 flags;
	quint64 indexOffset;
	quint32 nodeCount;
	quint32 keyLength;
	qint64 created;
};

struct IndexItem
{
	qint32 id;
	quint32 points;
	quint64 offset;
};

template<typename T> void put(QByteArray & buffer, T value)
{
	buffer.append(reinterpret_cast<const char *>(&value), sizeof(T));
}

template<typename T> T get(const char * & data)
{
	T value;
	memcpy(&value, data, sizeof(T));
	data += sizeof(T);
	return value;
}

qint64 pointSize(bool withNormals)
{
	return 16 + (withNormals ? 12 : 0);
}

QString cacheDirectory()
{
	return QStandardPaths::writableLocation(QStandardPaths::GenericCacheLocation) + "/rtabmap/nodeclouds";
}

bool readHeader(QFile & file, const QString & key, Header & header)
{
	QByteArray fixed = file.read(kHeaderSize);
	if(fixed.size() != kHeaderSize || memcmp(fixed.constData(), kMagic, 8) != 0)
	{
		return false;
	}
	const char * data = fixed.constData() + 8;
	if(get<quint32>(data) != kVersion)
	{
		return false;
	}
	header.flags = get<quint32>(data);
	header.indexOffset = get<quint64>(data);
	header.nodeCount = get<quint32>(data);
	header.keyLength = get<quint32>(data);
	header.created = get<qint64>(data);
	if(header.indexOffset == 0) // never finalized
	{
		return false;
	}
	const QByteArray expected = key.toUtf8();
	if((qint64)header.keyLength != expected.size())
	{
		return false;
	}
	return file.read(header.keyLength) == expected;
}

bool readIndex(QFile & file, const Header & header, std::vector<IndexItem> & index)
{
	if(!file.seek((qint64)header.indexOffset))
	{
		return false;
	}
	QByteArray raw = file.read((qint64)header.nodeCount * kIndexItemSize);
	if(raw.size() != (qint64)header.nodeCount * kIndexItemSize)
	{
		return false;
	}
	const char * data = raw.constData();
	index.resize(header.nodeCount);
	for(quint32 i=0; i<header.nodeCount; ++i)
	{
		index[i].id = get<qint32>(data);
		index[i].points = get<quint32>(data);
		index[i].offset = get<quint64>(data);
	}
	return true;
}

QByteArray serialize(int id, const NodeCloudCache::Entry & entry, bool withNormals)
{
	const pcl::PointCloud<pcl::PointXYZRGBNormal> & cloud = *entry.cloud;
	QByteArray buffer;
	buffer.reserve((int)(kRecordHeaderSize + cloud.size() * pointSize(withNormals)));
	put<qint32>(buffer, id);
	put<quint32>(buffer, (quint32)cloud.size());
	Transform t = entry.localTransform.isNull() ? Transform::getIdentity() : entry.localTransform;
	for(int i=0; i<12; ++i)
	{
		put<float>(buffer, t.data()[i]);
	}
	for(size_t i=0; i<cloud.size(); ++i)
	{
		const pcl::PointXYZRGBNormal & p = cloud[i];
		put<float>(buffer, p.x);
		put<float>(buffer, p.y);
		put<float>(buffer, p.z);
		put<quint8>(buffer, p.r);
		put<quint8>(buffer, p.g);
		put<quint8>(buffer, p.b);
		put<quint8>(buffer, p.a);
		if(withNormals)
		{
			put<float>(buffer, p.normal_x);
			put<float>(buffer, p.normal_y);
			put<float>(buffer, p.normal_z);
		}
	}
	return buffer;
}

bool deserialize(const QByteArray & raw, bool withNormals, int & id, NodeCloudCache::Entry & entry)
{
	if(raw.size() < kRecordHeaderSize)
	{
		return false;
	}
	const char * data = raw.constData();
	id = get<qint32>(data);
	const quint32 points = get<quint32>(data);
	if(raw.size() != kRecordHeaderSize + (qint64)points * pointSize(withNormals))
	{
		return false;
	}
	float t[12];
	for(int i=0; i<12; ++i)
	{
		t[i] = get<float>(data);
	}
	entry.localTransform = Transform(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8], t[9], t[10], t[11]);
	entry.cloud.reset(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
	entry.cloud->resize(points);
	for(quint32 i=0; i<points; ++i)
	{
		pcl::PointXYZRGBNormal & p = (*entry.cloud)[i];
		p.x = get<float>(data);
		p.y = get<float>(data);
		p.z = get<float>(data);
		p.r = get<quint8>(data);
		p.g = get<quint8>(data);
		p.b = get<quint8>(data);
		p.a = get<quint8>(data);
		if(withNormals)
		{
			p.normal_x = get<float>(data);
			p.normal_y = get<float>(data);
			p.normal_z = get<float>(data);
		}
		else
		{
			p.normal_x = p.normal_y = p.normal_z = 0.0f;
		}
		p.curvature = 0.0f;
	}
	return true;
}

void setError(QString * error, const QString & message)
{
	if(error)
	{
		*error = message;
	}
}

}

QString NodeCloudCache::makeKey(const QString & databasePath, const QString & settingsDigest)
{
	return databasePath + "\n" + settingsDigest;
}

QString NodeCloudCache::filePath(const QString & key)
{
	const QString database = key.section('\n', 0, 0);
	QString name = QFileInfo(database).completeBaseName();
	name.replace(QRegExp("[^A-Za-z0-9_.-]"), "_");
	const QString hash = QString(QCryptographicHash::hash(key.toUtf8(), QCryptographicHash::Sha1).toHex()).left(12);
	return cacheDirectory() + "/" + name + "_" + hash + ".rtabnc";
}

NodeCloudCache::Summary NodeCloudCache::summary(const QString & key)
{
	Summary out;
#if Q_BYTE_ORDER != Q_LITTLE_ENDIAN
	return out;
#endif
	QFile file(filePath(key));
	Header header;
	std::vector<IndexItem> index;
	if(!file.open(QIODevice::ReadOnly) || !readHeader(file, key, header) || !readIndex(file, header, index))
	{
		return out;
	}
	out.valid = true;
	out.created = QDateTime::fromSecsSinceEpoch(header.created);
	out.withNormals = header.flags & kFlagNormals;
	out.useIntensity = header.flags & kFlagIntensity;
	out.bytes = file.size();
	for(size_t i=0; i<index.size(); ++i)
	{
		out.ids.insert(index[i].id);
	}
	return out;
}

bool NodeCloudCache::load(const QString & key, const std::set<int> & wanted, std::map<int, Entry> & entries, QString * error)
{
	QFile file(filePath(key));
	Header header;
	std::vector<IndexItem> index;
	if(!file.open(QIODevice::ReadOnly) || !readHeader(file, key, header) || !readIndex(file, header, index))
	{
		setError(error, QObject::tr("The saved clouds cannot be read."));
		return false;
	}
	const bool withNormals = header.flags & kFlagNormals;

	// in file order, so the disk is read forward
	std::vector<IndexItem> toRead;
	for(size_t i=0; i<index.size(); ++i)
	{
		if(wanted.find(index[i].id) != wanted.end())
		{
			toRead.push_back(index[i]);
		}
	}
	std::sort(toRead.begin(), toRead.end(), [](const IndexItem & a, const IndexItem & b) {return a.offset < b.offset;});

	for(size_t i=0; i<toRead.size(); ++i)
	{
		const qint64 size = kRecordHeaderSize + (qint64)toRead[i].points * pointSize(withNormals);
		if(!file.seek((qint64)toRead[i].offset))
		{
			setError(error, QObject::tr("The saved clouds are truncated."));
			return false;
		}
		QByteArray raw = file.read(size);
		int id = 0;
		Entry entry;
		if(raw.size() != size || !deserialize(raw, withNormals, id, entry) || id != toRead[i].id)
		{
			setError(error, QObject::tr("The saved clouds are damaged (node %1).").arg(toRead[i].id));
			return false;
		}
		entries[id] = entry;
	}
	return true;
}

bool NodeCloudCache::save(const QString & key, bool withNormals, bool useIntensity, const std::map<int, Entry> & entries, QString * error)
{
#if Q_BYTE_ORDER != Q_LITTLE_ENDIAN
	setError(error, QObject::tr("Saving per-node clouds is only supported on little endian machines."));
	return false;
#endif
	const quint32 flags = (withNormals ? kFlagNormals : 0) | (useIntensity ? kFlagIntensity : 0);
	const QString path = filePath(key);
	if(!QDir().mkpath(QFileInfo(path).absolutePath()))
	{
		setError(error, QObject::tr("Cannot create \"%1\".").arg(QFileInfo(path).absolutePath()));
		return false;
	}

	// Nodes already in the file are carried over, unless they are saved again
	QFile previous(path);
	Header previousHeader;
	std::vector<IndexItem> previousIndex;
	bool carryOver = false;
	if(previous.open(QIODevice::ReadOnly) && readHeader(previous, key, previousHeader) && readIndex(previous, previousHeader, previousIndex))
	{
		carryOver = previousHeader.flags == flags;
	}

	const QString tmpPath = path + ".tmp";
	QFile out(tmpPath);
	if(!out.open(QIODevice::WriteOnly | QIODevice::Truncate))
	{
		setError(error, QObject::tr("Cannot write \"%1\".").arg(tmpPath));
		return false;
	}
	const QByteArray keyBytes = key.toUtf8();
	QByteArray header;
	header.append(kMagic, 8);
	put<quint32>(header, kVersion);
	put<quint32>(header, flags);
	put<quint64>(header, 0); // index offset, set at the end
	put<quint32>(header, 0); // node count, set at the end
	put<quint32>(header, (quint32)keyBytes.size());
	put<qint64>(header, (qint64)QDateTime::currentSecsSinceEpoch());
	header.append(keyBytes);
	bool ok = out.write(header) == header.size();

	std::vector<IndexItem> index;
	for(std::map<int, Entry>::const_iterator iter=entries.begin(); ok && iter!=entries.end(); ++iter)
	{
		if(!iter->second.cloud)
		{
			continue;
		}
		IndexItem item;
		item.id = iter->first;
		item.points = (quint32)iter->second.cloud->size();
		item.offset = (quint64)out.pos();
		const QByteArray record = serialize(iter->first, iter->second, withNormals);
		ok = out.write(record) == record.size();
		index.push_back(item);
	}

	if(ok && carryOver)
	{
		std::sort(previousIndex.begin(), previousIndex.end(), [](const IndexItem & a, const IndexItem & b) {return a.offset < b.offset;});
		for(size_t i=0; ok && i<previousIndex.size(); ++i)
		{
			if(entries.find(previousIndex[i].id) != entries.end())
			{
				continue;
			}
			const qint64 size = kRecordHeaderSize + (qint64)previousIndex[i].points * pointSize(withNormals);
			IndexItem item = previousIndex[i];
			item.offset = (quint64)out.pos();
			ok = previous.seek((qint64)previousIndex[i].offset);
			QByteArray record = ok ? previous.read(size) : QByteArray();
			ok = ok && record.size() == size && out.write(record) == record.size();
			index.push_back(item);
		}
	}
	previous.close();

	if(ok)
	{
		const quint64 indexOffset = (quint64)out.pos();
		QByteArray raw;
		for(size_t i=0; i<index.size(); ++i)
		{
			put<qint32>(raw, index[i].id);
			put<quint32>(raw, index[i].points);
			put<quint64>(raw, index[i].offset);
		}
		ok = out.write(raw) == raw.size();
		QByteArray fields;
		put<quint64>(fields, indexOffset);
		put<quint32>(fields, (quint32)index.size());
		ok = ok && out.seek(16) && out.write(fields) == fields.size();
	}
	ok = ok && out.flush();
	out.close();

	if(ok)
	{
		QFile::remove(path);
		ok = QFile::rename(tmpPath, path);
	}
	if(!ok)
	{
		QFile::remove(tmpPath);
		setError(error, QObject::tr("Cannot write \"%1\" (disk full?).").arg(path));
	}
	return ok;
}

bool NodeCloudCache::remove(const QString & key)
{
	return QFile::remove(filePath(key));
}

}
