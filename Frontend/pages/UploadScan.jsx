import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './UploadScan.css';

const UploadScan = () => {
  const [dragActive, setDragActive] = useState(false);
  const [patientData, setPatientData] = useState({
    patientId: '',
    name: '',
    age: '',
    gender: ''
  });

  const [scanFiles, setScanFiles] = useState({
    axial: null,
    sagittal: null,
    coronal: null
  });

  const navigate = useNavigate();

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e, type) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setScanFiles(prev => ({
        ...prev,
        [type]: e.dataTransfer.files
      }));
    }
  };

  const handleFileInput = (e, type) => {
    if (e.target.files) {
      setScanFiles(prev => ({
        ...prev,
        [type]: e.target.files
      }));
    }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setPatientData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleStartAnalysis = () => {
    console.log('Starting analysis with:', patientData, scanFiles);
    navigate('/patient/123');
  };

  return (
    <div className="upload-container">
      <header className="upload-header">
        <h1>AI-Powered Radiology Reporting for Precision Diagnostics</h1>
        <p>Leverage cutting-edge AI to automate and enhance the accuracy of medical image analysis.</p>
      </header>

      {/* === UPDATED SECTION STARTS HERE === */}
      <section className="upload-section">
        <h2>Upload DICOM Scans</h2>
        <p>Please upload all three orientations for optimal analysis</p>

       <div className="scan-group horizontal">


          {/* Axial */}
          <div className="scan-block">
            <h4>Axial</h4>
            <div
              className={`upload-area ${dragActive ? 'drag-active' : ''}`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={(e) => handleDrop(e, 'axial')}
            >
             
              <input
                type="file"
                id="file-axial"
                multiple
                accept=".dcm,.dicom"
                style={{ display: 'none' }}
                onChange={(e) => handleFileInput(e, 'axial')}
              />
              <button onClick={() => document.getElementById('file-axial').click()}>
                Browse Axial
              </button>
            </div>
            {scanFiles.axial && <p className="file-info">{scanFiles.axial.length} file(s) selected</p>}
          </div>

          {/* Sagittal */}
          <div className="scan-block">
            <h4>Sagittal</h4>
            <div
              className={`upload-area ${dragActive ? 'drag-active' : ''}`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={(e) => handleDrop(e, 'sagittal')}
            >
              
              <input
                type="file"
                id="file-sagittal"
                multiple
                accept=".dcm,.dicom"
                style={{ display: 'none' }}
                onChange={(e) => handleFileInput(e, 'sagittal')}
              />
              <button onClick={() => document.getElementById('file-sagittal').click()}>
                Browse Sagittal
              </button>
            </div>
            {scanFiles.sagittal && <p className="file-info">{scanFiles.sagittal.length} file(s) selected</p>}
          </div>

          {/* Coronal */}
          <div className="scan-block">
            <h4>Coronal</h4>
            <div
              className={`upload-area ${dragActive ? 'drag-active' : ''}`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={(e) => handleDrop(e, 'coronal')}
            >
             
              <input
                type="file"
                id="file-coronal"
                multiple
                accept=".dcm,.dicom"
                style={{ display: 'none' }}
                onChange={(e) => handleFileInput(e, 'coronal')}
              />
              <button onClick={() => document.getElementById('file-coronal').click()}>
                Browse Coronal
              </button>
            </div>
            {scanFiles.coronal && <p className="file-info">{scanFiles.coronal.length} file(s) selected</p>}
          </div>

        </div>
      </section>
      {/* === UPDATED SECTION ENDS HERE === */}

      <section className="patient-details">
        <h3>Patient Details</h3>
        <div className="patient-form">
          <div className="form-row">
            <div className="form-group">
              <label>Patient ID</label>
              <input
                type="text"
                name="patientId"
                placeholder="Unique Patient Identifier"
                value={patientData.patientId}
                onChange={handleInputChange}
              />
            </div>
            <div className="form-group">
              <label>Name</label>
              <input
                type="text"
                name="name"
                placeholder="Full Name"
                value={patientData.name}
                onChange={handleInputChange}
              />
            </div>
          </div>

          <div className="form-row">
            <div className="form-group">
              <label>Age</label>
              <input
                type="number"
                name="age"
                placeholder="Age in years"
                value={patientData.age}
                onChange={handleInputChange}
              />
            </div>
            <div className="form-group">
              <label>Gender</label>
              <select
                name="gender"
                value={patientData.gender}
                onChange={handleInputChange}
              >
                <option value="">Select Gender</option>
                <option value="male">Male</option>
                <option value="female">Female</option>
                <option value="other">Other</option>
              </select>
            </div>
          </div>
        </div>

        <button className="analysis-btn" onClick={handleStartAnalysis}>
          Start Analysis
        </button>
      </section>

      <footer className="upload-footer">
        <div>Company Resources Legal</div>
      </footer>
    </div>
  );
};

export default UploadScan;

