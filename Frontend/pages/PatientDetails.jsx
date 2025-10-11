import React from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import './PatientDetails.css';

const PatientDetails = () => {
  const { id } = useParams();
  const navigate = useNavigate();

  // Mock patient data - in real app, fetch from backend
  const patient = {
    id: 'P-789012',
    name: 'Alice Johnson',
    age: 58,
    gender: 'Female'
  };

  const findings = [
    {
      condition: 'Ground Glass Opacity (GGO)',
      confidence: '88.2%',
      description: 'Food ground glass opacity in the left upper lobe, consistent with inflammatory changes. Requires follow-up to monitor resolution.'
    },
    {
      condition: 'Spinal Stenosis',
      description: 'Moderate lumbar spinal stenosis at L4-L5 level with associated neural foraminal narrowing. Impinging on the traversing nerve root.'
    },
    {
      condition: 'Cerebral Microbleeds',
      description: 'Multiple small cerebral microbleeds identified in the frontal and parietal white matter. Correlate clinically for risk factors.'
    }
  ];

  const handleGenerateReport = () => {
    navigate('/report/123');
  };

  return (
    <div className="patient-details-container">
      <header className="patient-header">
        <h1>Patient Details</h1>
        <button className="generate-report-btn" onClick={handleGenerateReport}>
          Generate Radiology Report
        </button>
      </header>

      <div className="patient-info">
        <div className="info-grid">
          <div className="info-item">
            <label>Patient ID:</label>
            <span>{patient.id}</span>
          </div>
          <div className="info-item">
            <label>Name:</label>
            <span>{patient.name}</span>
          </div>
          <div className="info-item">
            <label>Age:</label>
            <span>{patient.age}</span>
          </div>
          <div className="info-item">
            <label>Gender:</label>
            <span>{patient.gender}</span>
          </div>
        </div>
      </div>

      <section className="dicom-viewer-section">
        <h2>DICOM Viewer</h2>
        <div className="dicom-viewer">
          <div className="viewer-placeholder">
            <p>DICOM Image Viewer would be integrated here</p>
            <p>Using libraries like Cornerstone.js</p>
          </div>
        </div>
      </section>

      <section className="ai-findings">
        <h2>AI Analysis Findings</h2>
        <div className="findings-list">
          {findings.map((finding, index) => (
            <div key={index} className="finding-item">
              <h4>{finding.condition}</h4>
              {finding.confidence && (
                <span className="confidence">Confidence: {finding.confidence}</span>
              )}
              <p>{finding.description}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="diagram-section">
        <h3>Analysis Diagram</h3>
        <div className="diagram-placeholder">
          <p>AI-generated heatmaps and diagrams would appear here</p>
          <ul>
            <li>Primary: 0.1 mm</li>
            <li>Primary: 0.3 mm</li>
            <li>Primary: 0.4 mm</li>
          </ul>
        </div>
      </section>
    </div>
  );
};

export default PatientDetails;