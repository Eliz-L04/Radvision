import React from "react";
import { useNavigate, useParams } from "react-router-dom";
import "./PatientDetails.css";

import PersonIcon from "@mui/icons-material/Person";
import BadgeIcon from "@mui/icons-material/Badge";
import CalendarTodayIcon from "@mui/icons-material/CalendarToday";
import FemaleIcon from "@mui/icons-material/Female";
import VisibilityIcon from "@mui/icons-material/Visibility";
import HubIcon from "@mui/icons-material/Hub";
import LocalFireDepartmentIcon from "@mui/icons-material/LocalFireDepartment";
import SaveIcon from "@mui/icons-material/Save";

const PatientDetails = () => {
  const navigate = useNavigate();
  const { id } = useParams();

  return (
    <div className="pd-container">

      {/* HEADER */}
      <div className="pd-top">
        <div>
          <h1>Patient Diagnostic Review</h1>
          <p>Case ID: RAD-9928-AX-001 • Study Date: Oct 24, 2024</p>
        </div>

        <div className="top-actions">
          <button className="green-outline-btn">
            <SaveIcon fontSize="small" />
            Save Progress
          </button>

          <button 
            className="green-btn"
            onClick={() => navigate(`/report/${id}`)}
          >
            Generate Report
          </button>
        </div>
      </div>

      {/* PATIENT INFO */}
      <div className="pd-info-row">
        <div className="info-card">
          <PersonIcon />
          <div>
            <span>Full Name</span>
            <h4>Alice Johnson</h4>
          </div>
        </div>

        <div className="info-card">
          <BadgeIcon />
          <div>
            <span>Patient ID</span>
            <h4>P-789012</h4>
          </div>
        </div>

        <div className="info-card">
          <CalendarTodayIcon />
          <div>
            <span>Age / DOB</span>
            <h4>58Y (Mar 12, 1966)</h4>
          </div>
        </div>

        <div className="info-card">
          <FemaleIcon />
          <div>
            <span>Gender</span>
            <h4>Female</h4>
          </div>
        </div>
      </div>

      {/* MAIN GRID */}
      <div className="pd-main">

        {/* LEFT SIDE */}
        <div className="left-panel">

          {/* MPR */}
          <div className="section-card">
            <div className="section-header">
              <VisibilityIcon />
              <h3>Multi-Planar Reconstruction (MPR)</h3>
            </div>

            <div className="dicom-grid">
              <div className="dicom-box">Axial View</div>
              <div className="dicom-box">Coronal View</div>
              <div className="dicom-box">Sagittal View</div>
            </div>

            <div className="dicom-footer">
              WL 40 • WW 400 • Zoom 1.2x • Interp: Bilinear
            </div>
          </div>

          {/* HEATMAP SECTION */}
          <div className="section-card">
            <div className="section-header">
              <LocalFireDepartmentIcon />
              <h3>Heatmap Distribution Analysis</h3>
            </div>

            <div className="heatmap-container">
              <div className="heatmap-chart">
                Contrast Density Delta
              </div>

              <div className="heatmap-table">
                <div className="heat-row critical">
                  <span>L-Upper Lobe</span>
                  <span>0.92</span>
                  <span>Critical</span>
                </div>
                <div className="heat-row">
                  <span>R-Lower Lobe</span>
                  <span>0.45</span>
                  <span>Normal</span>
                </div>
                <div className="heat-row">
                  <span>Mediastinum</span>
                  <span>0.68</span>
                  <span>Observe</span>
                </div>
                <div className="heat-row">
                  <span>Pleural Cavity</span>
                  <span>0.21</span>
                  <span>Normal</span>
                </div>
              </div>
            </div>
          </div>

        </div>

        {/* RIGHT SIDE */}
        <div className="right-panel">

          {/* TDA SHAPE ANALYSIS */}
          <div className="section-card">
            <div className="section-header">
              <HubIcon />
              <h3>TDA Shape Analysis</h3>
            </div>

            <div className="tda-graph">
              Persistence Diagram
            </div>

            <div className="tda-metrics">
              <div>
                <span>Global Shape Entropy</span>
                <strong>4.82 bits</strong>
              </div>

              <div>
                <span>Connectivity Index</span>
                <strong>0.64 β</strong>
              </div>

              <div>
                <span>Geometric Divergence</span>
                <strong>12.5%</strong>
              </div>
            </div>

            <div className="tda-warning">
              Observation Required (Stage 2)
              <p>Structural anomalies detected in H1 dimension.</p>
            </div>

          </div>

        </div>

      </div>
    </div>
  );
};

export default PatientDetails;
