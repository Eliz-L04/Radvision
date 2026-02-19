import React from "react";
import { useParams } from "react-router-dom";
import "./RadiologyReport.css";

const RadiologyReport = () => {
  const { id } = useParams();

  return (
    <div className="rad-report-container">

      {/* HEADER */}
      <div className="rad-report-header">
        <div>
          <div className="case-id">CASE #{id}</div>
          <h1>Draft Radiology Report</h1>
          <p>
            Reviewing diagnostic imaging for patient <strong>RadVision AI - Knee MRI Study</strong>.
            Ensure all clinical impressions are finalized before PACS transmission.
          </p>
        </div>

        <div className="header-actions">
          <button className="outline-btn">Save Draft</button>
          <button className="outline-btn">Export PDF</button>
          <button className="outline-btn">Send to PACS</button>
          <button className="primary-btn">Finalize Report</button>
        </div>
      </div>

      {/* MAIN CONTENT GRID */}
      <div className="rad-grid">

        {/* LEFT COLUMN */}
        <div className="left-panel">

          {/* Findings */}
          <div className="card">
            <div className="card-header">
              <h2>Findings</h2>
              <span className="status analyzing">Analyzing</span>
            </div>
            <div className="card-content">
              <p>
                MRI of the right knee demonstrates a tear of the medial meniscus 
                posterior horn with mild joint effusion. Early cartilage thinning 
                is noted in the medial compartment. The anterior cruciate ligament (ACL) 
                and posterior cruciate ligament (PCL) appear intact with normal signal intensity.
                No acute fracture or bone marrow edema identified.
              </p>
              <div className="autosave">Auto-saved 2 minutes ago</div>
            </div>
          </div>

          {/* Impressions */}
          <div className="card">
            <div className="card-header">
              <h2>Impressions</h2>
              <span className="status pending">Pending Verification</span>
            </div>
            <div className="card-content">
              <p>
                Medial meniscus posterior horn tear with mild joint effusion. 
                Early degenerative cartilage changes in the medial compartment.
                Cruciate ligaments intact. Clinical correlation recommended.
              </p>
              <div className="autosave">Auto-saved 2 minutes ago</div>
            </div>
          </div>

        </div>

        {/* RIGHT COLUMN */}
        <div className="right-panel">

          {/* Radiologist Notes */}
          <div className="side-card">
            <h3>Radiologist Notes</h3>
            <p>
              Patient presents with chronic knee pain and limited range of motion.
              Findings correlate with clinical suspicion of medial meniscus injury.
              Consider orthopedic consultation.
            </p>
          </div>

          {/* Study Info */}
          <div className="side-card">
            <h3>Study Information</h3>
            <div className="info-row">
              <span>Modality</span>
              <strong>MRI</strong>
            </div>
            <div className="info-row">
              <span>Region</span>
              <strong>Knee Joint</strong>
            </div>
            <div className="info-row">
              <span>Field Strength</span>
              <strong>1.5T</strong>
            </div>
            <div className="info-row">
              <span>Slice Thickness</span>
              <strong>3 mm</strong>
            </div>
          </div>

          {/* Patient Info */}
          <div className="side-card">
            <h3>Patient Information</h3>
            <div className="info-row">
              <span>Name</span>
              <strong>Alice Johnson</strong>
            </div>
            <div className="info-row">
              <span>Age</span>
              <strong>45 Years</strong>
            </div>
            <div className="info-row">
              <span>Gender</span>
              <strong>Female</strong>
            </div>
          </div>

        </div>
      </div>

    </div>
  );
};

export default RadiologyReport;
