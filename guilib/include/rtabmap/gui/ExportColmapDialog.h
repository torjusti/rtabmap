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

#ifndef EXPORTCOLMAPDIALOG_H_
#define EXPORTCOLMAPDIALOG_H_

#include "rtabmap/gui/rtabmap_gui_export.h"

#include <QDialog>
#include <QSettings>
#include <rtabmap/core/Parameters.h>
#include <rtabmap/core/Link.h>
#include <rtabmap/core/Transform.h>
#include <rtabmap/core/Signature.h>
#include <map>

class Ui_ExportColmapDialog;

namespace rtabmap {

class DBDriver;

/**
 * @class ExportColmapDialog
 * @brief Exports an optimized map as a COLMAP sparse reconstruction.
 *
 * Writes cameras.txt, images.txt and points3D.txt, plus optionally RGB images.
 * Replaces the former Bundler (*.out) export.
 */
class RTABMAP_GUI_EXPORT ExportColmapDialog : public QDialog
{
	Q_OBJECT

public:
	ExportColmapDialog(QWidget * parent = 0);
	virtual ~ExportColmapDialog();

	void saveSettings(QSettings & settings, const QString & group = "") const;
	void loadSettings(QSettings & settings, const QString & group = "");

	void setWorkingDirectory(const QString & path);

	void exportColmap(
			std::map<int, Transform> & poses,
			const std::multimap<int, Link> & links,
			const std::map<int, Signature> & signatures,
			DBDriver * dbDriver,
			const ParametersMap & parameters);

Q_SIGNALS:
	void configChanged();

private Q_SLOTS:
	void getPath();
	void restoreDefaults();
	void updateVisibility();

private:
	Ui_ExportColmapDialog * _ui;
};

}

#endif /* EXPORTCOLMAPDIALOG_H_ */
