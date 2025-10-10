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

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      // Handle the dropped files
      console.log('Files dropped:', e.dataTransfer.files);
    }
  };

  const handleFileInput = (e) => {
    if (e.target.files) {
      console.log('Files selected:', e.target.files);
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
    // Mock analysis start
    console.log('Starting analysis with:', patientData);
    navigate('/patient/123'); // Navigate to patient details page
  };

  return (
    <div className="upload-container">
      <header className="upload-header">
        <h1>AI-Powered Radiology Reporting for Precision Diagnostics</h1>
        <p>Leverage cutting-edge AI to automate and enhance the accuracy of medical image analysis.</p>
      </header>

      <section className="upload-section">
        <h2>Upload DICOM Scan</h2>
        <div 
          className={`upload-area ${dragActive ? 'drag-active' : ''}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <div className="upload-content">
            <p>Drag & drop your DICOM files here, or</p>
            <input
              type="file"
              id="file-input"
              multiple
              accept=".dcm,.dicom"
              onChange={handleFileInput}
              style={{ display: 'none' }}
            />
            <button 
              className="browse-btn"
              onClick={() => document.getElementById('file-input').click()}
            >
              Browse Files
            </button>
          </div>
        </div>
      </section>

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
