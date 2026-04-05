import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import "./UploadScan.css";

import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import PersonIcon from "@mui/icons-material/Person";
import DescriptionIcon from "@mui/icons-material/Description";
import SecurityIcon from "@mui/icons-material/Security";

const UploadScan = () => {
  const [dragActive, setDragActive] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [patientData, setPatientData] = useState({
    patientId: "",
    name: "",
    age: "",
    gender: ""
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
      setScanFiles((prev) => ({
        ...prev,
        [type]: e.dataTransfer.files
      }));
    }
  };

  const handleFileInput = (e, type) => {
    if (e.target.files) {
      setScanFiles((prev) => ({
        ...prev,
        [type]: e.target.files
      }));
    }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setPatientData((prev) => ({
      ...prev,
      [name]: value
    }));
  };

  const handleStartAnalysis = async () => {
    setError("");

    if (
      !patientData.patientId ||
      !patientData.name ||
      !patientData.age ||
      !patientData.gender
    ) {
      setError("Please fill in all patient details");
      return;
    }

    if (!scanFiles.axial && !scanFiles.sagittal && !scanFiles.coronal) {
      setError("Please upload at least one scan");
      return;
    }

    const formData = new FormData();
    formData.append("patientId", patientData.patientId);
    formData.append("name", patientData.name);
    formData.append("age", patientData.age);
    formData.append("gender", patientData.gender);

    if (scanFiles.axial) formData.append("axial", scanFiles.axial[0]);
    if (scanFiles.sagittal) formData.append("sagittal", scanFiles.sagittal[0]);
    if (scanFiles.coronal) formData.append("coronal", scanFiles.coronal[0]);

    try {
      setLoading(true);

      const res = await fetch("http://127.0.0.1:5000/upload-scan", {
        method: "POST",
        body: formData
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.error || "Upload failed");
      }

      navigate(`/patient/${data.patientId}`, {
        state: { patient: patientData, files: data.files }
      });

    } catch (err) {
      console.error(err);
      setError(err.message || "Cannot connect to server");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="upload-page">

      {/* ERROR MESSAGE */}
      {error && (
        <div
          style={{
            backgroundColor: "#fee",
            color: "#c00",
            padding: "12px",
            borderRadius: "6px",
            margin: "20px",
            textAlign: "center"
          }}
        >
          {error}
        </div>
      )}

      {/* HERO */}
      <div className="upload-hero">
        <div className="hero-left">
          <h1>
            AI-Powered Radiology Reporting
            <br />
            for <span>Precision Diagnostics</span>
          </h1>

          <p>
            Automate medical image analysis with advanced AI.
            Upload multi-planar DICOM or NumPy scans to generate
            structured diagnostic reports in seconds.
          </p>
        </div>

        <div className="hero-right">
          <img src="/images/patient-dashboard.jpg" alt="Radiology Lab" />
        </div>
      </div>

      {/* UPLOAD CARD */}
      <div className="upload-card">
        <div className="card-title">
          <CloudUploadIcon /> Upload DICOM / NPY Scans
        </div>

        <div className="scan-grid">
          {["axial", "sagittal", "coronal"].map((type) => (
            <div key={type} className="scan-box">
              <div className="scan-label">
                <DescriptionIcon fontSize="small" />{" "}
                {type.charAt(0).toUpperCase() + type.slice(1)}
              </div>

              <div
                className={`upload-area ${dragActive ? "drag-active" : ""}`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={(e) => handleDrop(e, type)}
              >
                <input
                  type="file"
                  id={`file-${type}`}
                  multiple
                  accept=".dcm,.dicom,.npy"
                  hidden
                  onChange={(e) => handleFileInput(e, type)}
                />

                <button
                  onClick={() =>
                    document.getElementById(`file-${type}`).click()
                  }
                >
                  Browse {type}
                </button>
              </div>

              {scanFiles[type] && (
                <div className="file-info">
                  {scanFiles[type].length} file(s) selected
                </div>
              )}
            </div>
          ))}
        </div>

        {/* PATIENT DETAILS */}
        <div className="patient-title">
          <PersonIcon /> Patient Details
        </div>

        <div className="patient-grid">
          <input
            name="patientId"
            placeholder="Patient ID"
            value={patientData.patientId}
            onChange={handleInputChange}
          />

          <input
            name="name"
            placeholder="Full Name"
            value={patientData.name}
            onChange={handleInputChange}
          />

          <input
            name="age"
            type="number"
            placeholder="Age"
            value={patientData.age}
            onChange={handleInputChange}
          />

          <select
            name="gender"
            value={patientData.gender}
            onChange={handleInputChange}
          >
            <option value="">Gender</option>
            <option value="male">Male</option>
            <option value="female">Female</option>
            <option value="other">Other</option>
          </select>
        </div>

        <button
          className="start-btn"
          onClick={handleStartAnalysis}
          disabled={loading}
        >
          {loading ? "Uploading..." : "Start Automated Analysis →"}
        </button>
      </div>

      <div className="upload-footer">
        <SecurityIcon fontSize="small" /> FDA Cleared • ISO 27001 • 99.4% Accuracy
      </div>
    </div>
  );
};

export default UploadScan;