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

#include "rtabmap/gui/ExportColmapDialog.h"
#include "ui_exportColmapDialog.h"
#include "rtabmap/gui/ProgressDialog.h"
#include <rtabmap/core/Optimizer.h>
#include <rtabmap/core/Parameters.h>
#include <rtabmap/core/DBDriver.h>
#include <rtabmap/core/Colmap.h>
#include <rtabmap/core/ProgressState.h>
#include <QFileDialog>
#include <QPushButton>
#include <QMessageBox>
#include <QApplication>
#include <QDir>
#include <string>
#include <algorithm>
#include <algorithm>
#include <cstdlib>

namespace rtabmap {

namespace {

Optimizer::Type sbaIndexToType(int index)
{
	return static_cast<Optimizer::Type>(index);
}

bool sbaIndexUsesPixelVariance(int index)
{
	return sbaIndexToType(index) != Optimizer::kTypeCVSBA;
}

class ColmapProgressState : public ProgressState
{
public:
	ColmapProgressState(ProgressDialog * dialog) :
		dialog_(dialog)
	{
		QObject::connect(dialog_, &ProgressDialog::canceled, [this]() { setCanceled(true); });
	}
	virtual bool callback(const std::string & msg) const
	{
		if(!msg.empty())
		{
			int n = 0, m = 0;
			const std::string::size_type slash = msg.rfind('/');
			std::string label = msg;
			if(slash != std::string::npos && slash > 0)
			{
				std::string::size_type start = msg.rfind(' ', slash);
				start = (start == std::string::npos) ? 0 : start + 1;
				n = atoi(msg.c_str() + start);
				m = atoi(msg.c_str() + slash + 1);
				if(start > 0)
				{
					label = msg.substr(0, start - (msg[start - 1] == ' ' ? 1 : 0));
				}
			}
			if(label != lastLabel_)
			{
				dialog_->appendText(QString::fromStdString(msg));
				lastLabel_ = label;
			}
			if(m > 0 && n >= 0)
			{
				dialog_->setMaximumSteps(m + 1);
				dialog_->setValue(std::min(n, m));
			}
			else
			{
				dialog_->incrementStep();
			}
		}
		QApplication::processEvents();
		return !isCanceled();
	}
private:
	ProgressDialog * dialog_;
	mutable std::string lastLabel_;
};

}  // namespace

ExportColmapDialog::ExportColmapDialog(QWidget * parent) :
	QDialog(parent)
{
	_ui = new Ui_ExportColmapDialog();
	_ui->setupUi(this);

	connect(_ui->toolButton_path, SIGNAL(clicked()), this, SLOT(getPath()));

	restoreDefaults();
	connect(_ui->buttonBox->button(QDialogButtonBox::RestoreDefaults), SIGNAL(clicked()), this, SLOT(restoreDefaults()));

	connect(_ui->doubleSpinBox_laplacianVariance, SIGNAL(valueChanged(double)), this, SIGNAL(configChanged()));
	connect(_ui->doubleSpinBox_linearSpeed, SIGNAL(valueChanged(double)), this, SIGNAL(configChanged()));
	connect(_ui->doubleSpinBox_angularSpeed, SIGNAL(valueChanged(double)), this, SIGNAL(configChanged()));
	connect(_ui->groupBox_export_points, SIGNAL(clicked(bool)), this, SIGNAL(configChanged()));
	connect(_ui->sba_iterations, SIGNAL(valueChanged(int)), this, SIGNAL(configChanged()));
	connect(_ui->comboBox_sbaType, SIGNAL(currentIndexChanged(int)), this, SIGNAL(configChanged()));
	connect(_ui->comboBox_sbaType, SIGNAL(currentIndexChanged(int)), this, SLOT(updateVisibility()));
	connect(_ui->comboBox_cameraModel, SIGNAL(currentIndexChanged(int)), this, SIGNAL(configChanged()));
	connect(_ui->sba_rematchFeatures, SIGNAL(stateChanged(int)), this, SIGNAL(configChanged()));
	connect(_ui->checkBox_exportImages, SIGNAL(stateChanged(int)), this, SIGNAL(configChanged()));
	connect(_ui->checkBox_allKeypoints, SIGNAL(stateChanged(int)), this, SIGNAL(configChanged()));

	int firstAvailable = -1;
	for(int i = 0; i < _ui->comboBox_sbaType->count(); ++i)
	{
		const Optimizer::Type type = sbaIndexToType(i);
		const bool usable = type != Optimizer::kTypeTORO && Optimizer::isAvailable(type);
		if(usable)
		{
			if(firstAvailable < 0) firstAvailable = i;
		}
		else
		{
			_ui->comboBox_sbaType->setItemData(i, 0, Qt::UserRole - 1);
		}
	}
	if(firstAvailable < 0)
	{
		_ui->groupBox_export_points->setEnabled(false);
		_ui->groupBox_export_points->setChecked(false);
	}
	else
	{
		_ui->comboBox_sbaType->setCurrentIndex(firstAvailable);
	}

	_ui->lineEdit_path->setText(QDir::currentPath());

	updateVisibility();
}

ExportColmapDialog::~ExportColmapDialog()
{
	delete _ui;
}

void ExportColmapDialog::saveSettings(QSettings & settings, const QString & group) const
{
	if(!group.isEmpty())
	{
		settings.beginGroup(group);
	}
	settings.setValue("maxLinearSpeed", _ui->doubleSpinBox_linearSpeed->value());
	settings.setValue("maxAngularSpeed", _ui->doubleSpinBox_angularSpeed->value());
	settings.setValue("laplacianThr", _ui->doubleSpinBox_laplacianVariance->value());
	settings.setValue("exportPoints", _ui->groupBox_export_points->isChecked());
	settings.setValue("exportImages", _ui->checkBox_exportImages->isChecked());
	settings.setValue("allKeypoints", _ui->checkBox_allKeypoints->isChecked());
	settings.setValue("cameraModel", _ui->comboBox_cameraModel->currentIndex());
	settings.setValue("sba_iterations", _ui->sba_iterations->value());
	settings.setValue("sba_type", static_cast<int>(sbaIndexToType(_ui->comboBox_sbaType->currentIndex())));
	settings.setValue("sba_variance", _ui->sba_variance->value());
	settings.setValue("sba_rematch_features", _ui->sba_rematchFeatures->isChecked());
	if(!group.isEmpty())
	{
		settings.endGroup();
	}
}

void ExportColmapDialog::loadSettings(QSettings & settings, const QString & group)
{
	if(!group.isEmpty())
	{
		settings.beginGroup(group);
	}
	_ui->doubleSpinBox_linearSpeed->setValue(settings.value("maxLinearSpeed", _ui->doubleSpinBox_linearSpeed->value()).toDouble());
	_ui->doubleSpinBox_angularSpeed->setValue(settings.value("maxAngularSpeed", _ui->doubleSpinBox_angularSpeed->value()).toDouble());
	_ui->doubleSpinBox_laplacianVariance->setValue(settings.value("laplacianThr", _ui->doubleSpinBox_laplacianVariance->value()).toDouble());
	_ui->groupBox_export_points->setChecked(settings.value("exportPoints", _ui->groupBox_export_points->isChecked()).toBool());
	_ui->checkBox_exportImages->setChecked(settings.value("exportImages", _ui->checkBox_exportImages->isChecked()).toBool());
	_ui->checkBox_allKeypoints->setChecked(settings.value("allKeypoints", _ui->checkBox_allKeypoints->isChecked()).toBool());
	_ui->comboBox_cameraModel->setCurrentIndex(settings.value("cameraModel", _ui->comboBox_cameraModel->currentIndex()).toInt());
	_ui->sba_iterations->setValue(settings.value("sba_iterations", _ui->sba_iterations->value()).toInt());
	{
		int saved = settings.value("sba_type", static_cast<int>(Optimizer::kTypeG2O)).toInt();
		if(saved == static_cast<int>(Optimizer::kTypeTORO))
		{
			saved = static_cast<int>(Optimizer::kTypeG2O);
		}
		if(saved >= 0 && saved < _ui->comboBox_sbaType->count())
		{
			_ui->comboBox_sbaType->setCurrentIndex(saved);
		}
	}
	_ui->sba_variance->setValue(settings.value("sba_variance", _ui->sba_variance->value()).toDouble());
	_ui->sba_rematchFeatures->setChecked(settings.value("sba_rematch_features", _ui->sba_rematchFeatures->isChecked()).toBool());
	if(!group.isEmpty())
	{
		settings.endGroup();
	}
}

void ExportColmapDialog::setWorkingDirectory(const QString & path)
{
	_ui->lineEdit_path->setText((path.isEmpty()?QDir::currentPath():path) + "/colmap");
}

void ExportColmapDialog::restoreDefaults()
{
	_ui->doubleSpinBox_linearSpeed->setValue(0);
	_ui->doubleSpinBox_angularSpeed->setValue(0);
	_ui->doubleSpinBox_laplacianVariance->setValue(0);
	_ui->groupBox_export_points->setChecked(true);
	_ui->checkBox_exportImages->setChecked(true);
	_ui->checkBox_allKeypoints->setChecked(true);
	_ui->comboBox_cameraModel->setCurrentIndex(kColmapOpenCV);
	_ui->sba_iterations->setValue(20);

	int defaultIndex = static_cast<int>(Optimizer::kTypeG2O);
	for(int i = 1; i < _ui->comboBox_sbaType->count(); ++i)
	{
		if(Optimizer::isAvailable(sbaIndexToType(i)))
		{
			defaultIndex = i;
			break;
		}
	}
	_ui->comboBox_sbaType->setCurrentIndex(defaultIndex);

	_ui->sba_variance->setValue(1.0);
	_ui->sba_rematchFeatures->setChecked(true);
}

void ExportColmapDialog::updateVisibility()
{
	const bool usesPixelVariance = sbaIndexUsesPixelVariance(_ui->comboBox_sbaType->currentIndex());
	_ui->sba_variance->setVisible(usesPixelVariance);
	_ui->label_variance->setVisible(usesPixelVariance);
}

void ExportColmapDialog::getPath()
{
	QString path = QFileDialog::getExistingDirectory(this, tr("Exporting COLMAP model..."), _ui->lineEdit_path->text());
	if(!path.isEmpty())
	{
		_ui->lineEdit_path->setText(path);
	}
}

void ExportColmapDialog::exportColmap(
		std::map<int, Transform> & poses,
		const std::multimap<int, Link> & links,
		const std::map<int, Signature> & signatures,
		DBDriver * dbDriver,
		const ParametersMap & parameters)
{
	if(this->exec() != QDialog::Accepted)
	{
		return;
	}
	const QString path = _ui->lineEdit_path->text();
	if(path.isEmpty())
	{
		return;
	}

	ColmapExportOptions options;
	options.cameraModel = _ui->comboBox_cameraModel->currentIndex();
	options.exportPoints = _ui->groupBox_export_points->isEnabled() && _ui->groupBox_export_points->isChecked();
	options.exportImages = _ui->checkBox_exportImages->isChecked();
	options.allKeypoints = _ui->checkBox_allKeypoints->isChecked();
	options.sbaType = sbaIndexToType(_ui->comboBox_sbaType->currentIndex());
	options.sbaIterations = _ui->sba_iterations->value();
	options.sbaVariance = _ui->sba_variance->value();
	options.sbaRematchFeatures = _ui->sba_rematchFeatures->isChecked();
	options.maxLinearVelocity = _ui->doubleSpinBox_linearSpeed->value();
	options.maxAngularVelocity = _ui->doubleSpinBox_angularSpeed->value();
	options.laplacianThr = _ui->doubleSpinBox_laplacianVariance->value();

	ProgressDialog progressDialog(this);
	progressDialog.setWindowTitle(tr("Exporting COLMAP model..."));
	progressDialog.setMinimumWidth(720);
	progressDialog.setMaximumSteps(1);
	progressDialog.setCancelButtonVisible(true);
	progressDialog.setAutoClose(false, 0);
	progressDialog.appendText(tr("Exporting COLMAP model to \"%1\"...").arg(path));
	progressDialog.show();
	QApplication::processEvents();

	ColmapProgressState state(&progressDialog);
	const ColmapExportResult result = rtabmap::exportColmap(
			poses, links, signatures, dbDriver, parameters, options, path.toStdString(), &state);

	if(state.isCanceled())
	{
		progressDialog.appendText(tr("Canceled."));
		progressDialog.setValue(progressDialog.maximumSteps());
		return;
	}

	progressDialog.setValue(progressDialog.maximumSteps());

	if(!result.ok)
	{
		QMessageBox::warning(this, tr("Exporting COLMAP..."), QString::fromStdString(result.error));
		return;
	}

	QString warning;
	if(result.droppedPoints)
	{
		warning += tr("\n%1 points were dropped for having fewer than 2 remaining views.").arg(result.droppedPoints);
	}
	if(result.outliersRemoved)
	{
		warning += tr("\n%1 bundle-adjustment outlier observations were removed.").arg(result.outliersRemoved);
	}
	if(result.posesTotal > result.camerasExported)
	{
		warning += tr("\n%1/%2 cameras were excluded for fast motion, blur, or missing calibration.")
				.arg(result.posesTotal - result.camerasExported).arg(result.posesTotal);
	}

	QMessageBox::information(this,
			tr("Exporting COLMAP model..."),
			tr("%1 cameras and %2 points exported to \"%3\".%4")
			.arg(result.camerasExported)
			.arg(result.pointsExported)
			.arg(path)
			.arg(warning));
}

}
