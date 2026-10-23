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

#ifndef CORELIB_SRC_SQLITEREADAHEADVFS_H_
#define CORELIB_SRC_SQLITEREADAHEADVFS_H_

#include <string>

namespace rtabmap {

/**
 * SQLite reads a database one page (4 KB) at a time. A node's images and scans
 * are stored in overflow pages, so loading one node is a few hundred 4 KB
 * reads in a row. On a filesystem with a high cost per request (WSL reaching a
 * Windows drive through 9p, CIFS, NFS, FUSE) that is about 7 times slower than
 * reading the same bytes with one large request.
 *
 * This VFS wraps the default one. When a read starts where the previous one
 * ended, it fetches a whole window at once and serves the next reads from it.
 * Isolated reads (B-tree lookups) go straight through.
 */

/** Window, in bytes, to use for the database at "path"; 0 means "do not use the VFS".
 * Automatic: 1 MB when the file is on a network-like filesystem, 0 otherwise.
 * The environment variable RTABMAP_SQLITE_READAHEAD_KB overrides it (0 disables). */
int sqliteReadAheadWindowFor(const std::string & path);

/** Name of the VFS to give to sqlite3_open_v2() (registered on first call), or 0 if unavailable on this platform. */
const char * sqliteReadAheadVfsName();

/** Window used by the next database opened by this thread with the VFS above. */
void sqliteReadAheadSetNextWindow(int bytes);

}

#endif /* CORELIB_SRC_SQLITEREADAHEADVFS_H_ */
