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

#include "SqliteReadAheadVfs.h"

#include <sqlite3.h>

#include <cstdlib>
#include <cstring>
#include <list>
#include <vector>
#include <algorithm>
#include <mutex>

#if defined(__linux__)
#include <sys/vfs.h>
#endif

namespace rtabmap {

#if defined(__linux__)

namespace {

// Same as the filesystem magic numbers of <linux/magic.h>
const long kMagic9p = 0x01021997;
const long kMagicSmb = 0x517B;
const long kMagicCifs = 0xFF534D42;
const long kMagicSmb2 = 0xFE534D42;
const long kMagicNfs = 0x6969;
const long kMagicFuse = 0x65735546;

const int kDefaultWindow = 1024*1024;
const size_t kMaxCachedBytes = 64*1024*1024;

thread_local int g_nextWindow = 0;

struct Window
{
	sqlite3_int64 start;
	std::vector<char> data;
};

struct State
{
	sqlite3_int64 window;
	sqlite3_int64 lastEnd;
	size_t bytes;
	std::list<Window> windows; // most recently used first
	unsigned char fileChangeCounter[16];
	bool haveFileChangeCounter;

	State(sqlite3_int64 w) : window(w), lastEnd(-1), bytes(0), haveFileChangeCounter(false)
	{
		memset(fileChangeCounter, 0, sizeof(fileChangeCounter));
	}
	void clear()
	{
		windows.clear();
		bytes = 0;
		lastEnd = -1;
	}
};

struct File
{
	sqlite3_file base; // must be first
	State * state;     // 0 when the file is passed through untouched
	// the real file follows, at an offset given by realFile()
};

sqlite3_vfs * g_orig = 0;
sqlite3_vfs g_vfs;

inline size_t round8(size_t n) {return (n+7)&~(size_t)7;}
inline sqlite3_file * realFile(sqlite3_file * f) {return (sqlite3_file*)(((char*)f) + round8(sizeof(File)));}

int raClose(sqlite3_file * f)
{
	File * p = (File*)f;
	sqlite3_file * real = realFile(f);
	delete p->state;
	p->state = 0;
	int rc = real->pMethods ? real->pMethods->xClose(real) : SQLITE_OK;
	return rc;
}

int raRead(sqlite3_file * f, void * buf, int amt, sqlite3_int64 ofst)
{
	File * p = (File*)f;
	sqlite3_file * real = realFile(f);
	State * st = p->state;
	if(!st)
	{
		return real->pMethods->xRead(real, buf, amt, ofst);
	}

	// SQLite checks "has another connection changed the file?" by reading the 16
	// bytes of the file change counter, at offset 24 of the header. Never serve
	// that from a window, and drop the windows if it moved.
	if(ofst <= 24 && ofst + amt >= 40)
	{
		int rc = real->pMethods->xRead(real, buf, amt, ofst);
		if(rc == SQLITE_OK)
		{
			const unsigned char * counter = (const unsigned char *)buf + (24 - ofst);
			if(!st->haveFileChangeCounter || memcmp(counter, st->fileChangeCounter, 16) != 0)
			{
				st->clear();
				memcpy(st->fileChangeCounter, counter, 16);
				st->haveFileChangeCounter = true;
			}
		}
		return rc;
	}

	for(std::list<Window>::iterator iter=st->windows.begin(); iter!=st->windows.end(); ++iter)
	{
		if(ofst >= iter->start && ofst + amt <= iter->start + (sqlite3_int64)iter->data.size())
		{
			memcpy(buf, iter->data.data() + (ofst - iter->start), amt);
			st->lastEnd = ofst + amt;
			if(iter != st->windows.begin())
			{
				st->windows.splice(st->windows.begin(), st->windows, iter);
			}
			return SQLITE_OK;
		}
	}

	const bool sequential = ofst == st->lastEnd;
	st->lastEnd = ofst + amt;
	if(!sequential || amt >= st->window)
	{
		return real->pMethods->xRead(real, buf, amt, ofst);
	}

	// The previous read ended where this one starts: read a whole window.
	sqlite3_int64 fileSize = 0;
	if(real->pMethods->xFileSize(real, &fileSize) != SQLITE_OK || fileSize - ofst < amt)
	{
		return real->pMethods->xRead(real, buf, amt, ofst);
	}
	const sqlite3_int64 len = std::min(st->window, fileSize - ofst);
	Window w;
	w.start = ofst;
	w.data.resize((size_t)len);
	if(real->pMethods->xRead(real, w.data.data(), (int)len, ofst) != SQLITE_OK)
	{
		return real->pMethods->xRead(real, buf, amt, ofst);
	}
	memcpy(buf, w.data.data(), amt);
	st->bytes += w.data.size();
	st->windows.push_front(std::move(w));
	while(st->bytes > kMaxCachedBytes && st->windows.size() > 1)
	{
		st->bytes -= st->windows.back().data.size();
		st->windows.pop_back();
	}
	return SQLITE_OK;
}

int raWrite(sqlite3_file * f, const void * buf, int amt, sqlite3_int64 ofst)
{
	File * p = (File*)f;
	if(p->state)
	{
		p->state->clear();
	}
	sqlite3_file * real = realFile(f);
	return real->pMethods->xWrite(real, buf, amt, ofst);
}

int raTruncate(sqlite3_file * f, sqlite3_int64 size)
{
	File * p = (File*)f;
	if(p->state)
	{
		p->state->clear();
	}
	sqlite3_file * real = realFile(f);
	return real->pMethods->xTruncate(real, size);
}

int raSync(sqlite3_file * f, int flags) {sqlite3_file * r = realFile(f); return r->pMethods->xSync(r, flags);}
int raFileSize(sqlite3_file * f, sqlite3_int64 * size) {sqlite3_file * r = realFile(f); return r->pMethods->xFileSize(r, size);}
int raLock(sqlite3_file * f, int lock) {sqlite3_file * r = realFile(f); return r->pMethods->xLock(r, lock);}
int raUnlock(sqlite3_file * f, int lock) {sqlite3_file * r = realFile(f); return r->pMethods->xUnlock(r, lock);}
int raCheckReservedLock(sqlite3_file * f, int * out) {sqlite3_file * r = realFile(f); return r->pMethods->xCheckReservedLock(r, out);}
int raFileControl(sqlite3_file * f, int op, void * arg) {sqlite3_file * r = realFile(f); return r->pMethods->xFileControl(r, op, arg);}
int raSectorSize(sqlite3_file * f) {sqlite3_file * r = realFile(f); return r->pMethods->xSectorSize(r);}
int raDeviceCharacteristics(sqlite3_file * f) {sqlite3_file * r = realFile(f); return r->pMethods->xDeviceCharacteristics(r);}
int raShmMap(sqlite3_file * f, int page, int pageSize, int extend, void volatile ** out)
{
	sqlite3_file * r = realFile(f);
	return r->pMethods->xShmMap(r, page, pageSize, extend, out);
}
int raShmLock(sqlite3_file * f, int offset, int n, int flags) {sqlite3_file * r = realFile(f); return r->pMethods->xShmLock(r, offset, n, flags);}
void raShmBarrier(sqlite3_file * f) {sqlite3_file * r = realFile(f); r->pMethods->xShmBarrier(r);}
int raShmUnmap(sqlite3_file * f, int deleteFlag) {sqlite3_file * r = realFile(f); return r->pMethods->xShmUnmap(r, deleteFlag);}

// No xFetch/xUnfetch (version 2 only): memory mapping is left off so every read goes through xRead.
const sqlite3_io_methods g_methods = {
	2,
	raClose, raRead, raWrite, raTruncate, raSync, raFileSize,
	raLock, raUnlock, raCheckReservedLock, raFileControl,
	raSectorSize, raDeviceCharacteristics,
	raShmMap, raShmLock, raShmBarrier, raShmUnmap,
	0, 0
};

int raOpen(sqlite3_vfs *, const char * name, sqlite3_file * f, int flags, int * outFlags)
{
	File * p = (File*)f;
	p->state = 0;
	sqlite3_file * real = realFile(f);
	int rc = g_orig->xOpen(g_orig, name, real, flags, outFlags);
	if(rc != SQLITE_OK)
	{
		return rc;
	}
	if((flags & SQLITE_OPEN_MAIN_DB) && g_nextWindow > 0)
	{
		p->state = new State(g_nextWindow);
	}
	p->base.pMethods = &g_methods;
	return SQLITE_OK;
}

std::once_flag g_registered;
bool g_available = false;

void registerVfs()
{
	g_orig = sqlite3_vfs_find(0);
	if(!g_orig)
	{
		return;
	}
	g_vfs = *g_orig;
	g_vfs.pNext = 0;
	g_vfs.zName = "rtabmap-readahead";
	g_vfs.szOsFile = (int)(round8(sizeof(File)) + g_orig->szOsFile);
	g_vfs.xOpen = raOpen;
	// pAppData is kept: the default VFS finds its locking style through it
	g_available = sqlite3_vfs_register(&g_vfs, 0) == SQLITE_OK;
}

}

int sqliteReadAheadWindowFor(const std::string & path)
{
	const char * env = std::getenv("RTABMAP_SQLITE_READAHEAD_KB");
	if(env && *env)
	{
		return std::max(0, std::atoi(env)) * 1024;
	}
	struct statfs fs;
	if(path.empty() || statfs(path.c_str(), &fs) != 0)
	{
		return 0;
	}
	const long type = (long)fs.f_type;
	if(type == kMagic9p || type == kMagicSmb || type == kMagicCifs || type == kMagicSmb2 ||
	   type == kMagicNfs || type == kMagicFuse)
	{
		return kDefaultWindow;
	}
	return 0;
}

const char * sqliteReadAheadVfsName()
{
	std::call_once(g_registered, registerVfs);
	return g_available ? g_vfs.zName : 0;
}

void sqliteReadAheadSetNextWindow(int bytes)
{
	g_nextWindow = bytes;
}

#else

int sqliteReadAheadWindowFor(const std::string &) {return 0;}
const char * sqliteReadAheadVfsName() {return 0;}
void sqliteReadAheadSetNextWindow(int) {}

#endif

}
