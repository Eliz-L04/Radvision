import React, { useState, useEffect, useCallback } from "react";
import { useNavigate, useParams, useLocation } from "react-router-dom";
import "./PatientDetails.css";

import MRISlider from "./MRISlider";
import PersonIcon from "@mui/icons-material/Person";
import BadgeIcon from "@mui/icons-material/Badge";
import CalendarTodayIcon from "@mui/icons-material/CalendarToday";
import FemaleIcon from "@mui/icons-material/Female";
import VisibilityIcon from "@mui/icons-material/Visibility";
import HubIcon from "@mui/icons-material/Hub";
import LocalFireDepartmentIcon from "@mui/icons-material/LocalFireDepartment";
import SaveIcon from "@mui/icons-material/Save";
import InfoOutlinedIcon from "@mui/icons-material/InfoOutlined";

const API_BASE = "http://127.0.0.1:5000";

const PatientDetails = () => {
  const navigate = useNavigate();
  const { id } = useParams();
  const location = useLocation();

  const { patient } = location.state || {};

  // TDA state
  const [tdaMetrics, setTdaMetrics] = useState(null);
  const [tdaLoading, setTdaLoading] = useState(false);
  const [tdaError, setTdaError] = useState("");
  const [selectedPlane, setSelectedPlane] = useState("");
  const [availablePlanes, setAvailablePlanes] = useState([]);
  const [diagramKey, setDiagramKey] = useState(0);
  const [showExplanation, setShowExplanation] = useState(false);

  // Grad-CAM state
  const [gcamInfo, setGcamInfo] = useState(null);
  const [gcamLoading, setGcamLoading] = useState(false);
  const [gcamError, setGcamError] = useState("");
  const [gcamPlane, setGcamPlane] = useState("");
  const [gcamAvailablePlanes, setGcamAvailablePlanes] = useState([]);
  const [gcamKey, setGcamKey] = useState(0);
  const [showGcamExplanation, setShowGcamExplanation] = useState(false);

  // Fetch TDA analysis
  const fetchTDA = useCallback((plane) => {
    if (!patient) return;

    setTdaLoading(true);
    setTdaError("");

    const url = plane
      ? `${API_BASE}/tda-analysis/${patient.patientId}?plane=${plane}`
      : `${API_BASE}/tda-analysis/${patient.patientId}`;

    fetch(url)
      .then((res) => {
        if (!res.ok) return res.json().then((d) => { throw new Error(d.error || "TDA failed"); });
        return res.json();
      })
      .then((data) => {
        setTdaMetrics(data);
        setAvailablePlanes(data.available_planes || []);
        if (!selectedPlane && data.plane) {
          setSelectedPlane(data.plane);
        }
        setDiagramKey((prev) => prev + 1);
        setTdaLoading(false);
      })
      .catch((err) => {
        console.error("TDA error:", err);
        setTdaError(err.message);
        setTdaLoading(false);
      });
  }, [patient, selectedPlane]);

  // Fetch Grad-CAM info
  const fetchGcamInfo = useCallback((plane) => {
    if (!patient) return;
    setGcamLoading(true);
    setGcamError("");

    const url = plane
      ? `${API_BASE}/grad-cam-info/${patient.patientId}?plane=${plane}`
      : `${API_BASE}/grad-cam-info/${patient.patientId}`;

    fetch(url)
      .then((res) => res.json())
      .then((data) => {
        if (data.error && !data.available_planes) throw new Error(data.error);
        setGcamInfo(data);
        setGcamAvailablePlanes(data.available_planes || []);
        if (!gcamPlane && data.plane) setGcamPlane(data.plane);
        setGcamKey((prev) => prev + 1);
        setGcamLoading(false);
      })
      .catch((err) => {
        setGcamError(err.message);
        setGcamLoading(false);
      });
  }, [patient, gcamPlane]);

  // Initial fetch: only load TDA (lighter) — Grad-CAM deferred until user selects plane
  useEffect(() => {
    if (patient) {
      fetchTDA(null);
    }
  }, [patient]);

  // When user changes TDA plane
  const handlePlaneChange = (e) => {
    const plane = e.target.value;
    setSelectedPlane(plane);
    fetchTDA(plane);
  };

  if (!patient) {
    return (
      <div style={{ padding: "40px", textAlign: "center" }}>
        <h2>No patient data found</h2>
        <p>Please upload scans first.</p>
        <button onClick={() => navigate("/")}>Go Back to Upload</button>
      </div>
    );
  }

  // Report generation state
  const [reportLoading, setReportLoading] = useState(false);
  const [reportError, setReportError] = useState("");

  const handleGenerateReport = async () => {
    setReportLoading(true);
    setReportError("");

    try {
      const res = await fetch(`${API_BASE}/generate-report/${patient.patientId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: patient.name,
          age: patient.age,
          gender: patient.gender,
        }),
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.error || "Report generation failed");
      }

      // Navigate to report page with AI-generated data
      navigate(`/report/${id}`, {
        state: { patient, report: data },
      });
    } catch (err) {
      console.error("Report generation error:", err);
      setReportError(err.message || "Failed to generate report");
    } finally {
      setReportLoading(false);
    }
  };

  const [saving, setSaving] = useState(false);

  const handleSaveProgress = () => {
    setSaving(true);
    fetch(`${API_BASE}/save-progress`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        patientId: patient.patientId,
        name: patient.name,
        age: patient.age,
        gender: patient.gender,
        scanType: "Knee MRI",
        status: "In Progress",
        tdaMetrics: tdaMetrics,
      }),
    })
      .then((res) => res.json())
      .then((data) => {
        setSaving(false);
        alert(`✅ ${data.message || "Progress saved!"}`);
      })
      .catch((err) => {
        setSaving(false);
        alert(`❌ Failed to save: ${err.message}`);
      });
  };

  const diagramUrl = selectedPlane
    ? `${API_BASE}/tda-diagram/${patient.patientId}?plane=${selectedPlane}&k=${diagramKey}`
    : `${API_BASE}/tda-diagram/${patient.patientId}?k=${diagramKey}`;

  return (
    <div className="pd-container">

      {/* HEADER */}
      <div className="pd-top">
        <div>
          <h1>Patient Diagnostic Review</h1>
          <p>Case ID: {patient.patientId} • Study Date: {new Date().toDateString()}</p>
        </div>
        <div className="top-actions">
          <button
            className="green-outline-btn"
            onClick={handleSaveProgress}
            disabled={saving}
          >
            <SaveIcon fontSize="small" />
            {saving ? "Saving..." : "Save Progress"}
          </button>
          <button
            className="green-btn"
            onClick={handleGenerateReport}
            disabled={reportLoading}
          >
            {reportLoading ? "Generating..." : "Generate Report"}
          </button>
        </div>
      </div>

      {/* REPORT GENERATION ERROR */}
      {reportError && (
        <div className="report-error-banner">
          ❌ {reportError}
          <button onClick={() => setReportError("")}>✕</button>
        </div>
      )}

      {/* REPORT GENERATION LOADING OVERLAY */}
      {reportLoading && (
        <div className="report-loading-overlay">
          <div className="report-loading-card">
            <div className="spinner large-spinner" />
            <h3>Generating AI Report</h3>
            <p>Analyzing all MRI views with ViT + TDA...</p>
            <p className="loading-sub">This may take 30–60 seconds</p>
            <div className="loading-steps">
              <span>🔬 Feature Extraction</span>
              <span>🧬 Topological Analysis</span>
              <span>🤖 AI Report Writing</span>
            </div>
          </div>
        </div>
      )}

      {/* PATIENT INFO */}
      <div className="pd-info-row">
        <div className="info-card">
          <PersonIcon />
          <div><span>Full Name</span><h4>{patient.name}</h4></div>
        </div>
        <div className="info-card">
          <BadgeIcon />
          <div><span>Patient ID</span><h4>{patient.patientId}</h4></div>
        </div>
        <div className="info-card">
          <CalendarTodayIcon />
          <div><span>Age</span><h4>{patient.age} Years</h4></div>
        </div>
        <div className="info-card">
          <FemaleIcon />
          <div><span>Gender</span><h4>{patient.gender}</h4></div>
        </div>
      </div>

      {/* MPR SECTION */}
      <div className="section-card">
        <div className="section-header">
          <VisibilityIcon />
          <h3>Multi-Planar Reconstruction (MPR)</h3>
        </div>
        <div className="dicom-grid">
          <div className="dicom-box">
            <MRISlider patientId={patient.patientId} plane="axial" />
          </div>
          <div className="dicom-box">
            <MRISlider patientId={patient.patientId} plane="coronal" />
          </div>
          <div className="dicom-box">
            <MRISlider patientId={patient.patientId} plane="sagittal" />
          </div>
        </div>
        <div className="dicom-footer">
          WL 40 • WW 400 • Zoom 1.2x • Interp: Bilinear
        </div>
      </div>

      {/* ANALYSIS ROW */}
      <div className="pd-analysis-row">

        {/* ========== TDA SECTION ========== */}
        <div className="section-card">
          <div className="section-header">
            <HubIcon />
            <h3>TDA Shape Analysis</h3>

            {/* Plane Selector Dropdown */}
            {availablePlanes.length > 0 && (
              <select
                className="plane-select"
                value={selectedPlane}
                onChange={handlePlaneChange}
                disabled={tdaLoading}
              >
                {availablePlanes.map((p) => (
                  <option key={p} value={p}>
                    {p.charAt(0).toUpperCase() + p.slice(1)} Plane
                  </option>
                ))}
              </select>
            )}
          </div>

          {/* Persistence Diagram */}
          <div className="tda-graph-large">
            {tdaLoading ? (
              <div className="tda-loading">
                <div className="spinner" />
                <span>Computing TDA for {selectedPlane || "..."} plane...</span>
              </div>
            ) : tdaError ? (
              <div className="tda-error-msg">{tdaError}</div>
            ) : (
              <img
                src={diagramUrl}
                alt="Persistence Diagram"
                className="tda-diagram-img"
                crossOrigin="anonymous"
              />
            )}
          </div>

          {/* TDA Metrics */}
          <div className="tda-metrics">
            <div>
              <span>H0 Connected Components</span>
              <strong>{tdaLoading ? "..." : tdaMetrics ? tdaMetrics.h0_count : "—"}</strong>
            </div>
            <div>
              <span>H1 Loops (Holes)</span>
              <strong>{tdaLoading ? "..." : tdaMetrics ? tdaMetrics.h1_count : "—"}</strong>
            </div>
            <div>
              <span>Persistence Entropy</span>
              <strong>{tdaLoading ? "..." : tdaMetrics ? `${tdaMetrics.persistence_entropy} bits` : "—"}</strong>
            </div>
            <div>
              <span>Mean H1 Lifetime</span>
              <strong>{tdaLoading ? "..." : tdaMetrics ? tdaMetrics.mean_life_h1.toFixed(2) : "—"}</strong>
            </div>
            <div>
              <span>Max H1 Lifetime</span>
              <strong>{tdaLoading ? "..." : tdaMetrics ? tdaMetrics.max_life_h1.toFixed(2) : "—"}</strong>
            </div>
          </div>

          {/* Status Badge */}
          {tdaMetrics && (
            <div className={`tda-warning ${tdaMetrics.h1_count > 10 ? "" : "tda-normal"}`}>
              {tdaMetrics.h1_count > 10 ? "⚠ Observation Required" : "✓ Normal Topology"}
              <p>
                {tdaMetrics.h1_count > 10
                  ? `${tdaMetrics.h1_count} persistent loops detected — structural anomalies in H1 dimension.`
                  : `${tdaMetrics.h1_count} loops detected — topology appears within normal range.`}
              </p>
              <p className="tda-plane-note">
                {selectedPlane.toUpperCase()} plane • {tdaMetrics.num_points} edge points
              </p>
            </div>
          )}

          {/* Explanation Toggle */}
          <button
            className="explain-toggle"
            onClick={() => setShowExplanation(!showExplanation)}
          >
            <InfoOutlinedIcon fontSize="small" />
            {showExplanation ? "Hide Explanation" : "What does this mean?"}
          </button>

          {showExplanation && (
            <div className="tda-explanation">
              <h4>Understanding TDA (Topological Data Analysis)</h4>
              <p>
                TDA examines the <strong>shape and structure</strong> of MRI data by detecting
                geometric features that persist across multiple scales. It converts MRI edge structures
                into a point cloud, then uses <em>persistent homology</em> to find meaningful patterns.
              </p>

              <div className="explain-item">
                <strong> Persistence Diagram</strong>
                <p>
                  Each dot represents a topological feature. The X-axis (<em>Birth</em>) is when a
                  feature appears; the Y-axis (<em>Death</em>) is when it disappears. Points far
                  from the diagonal are <strong>significant features</strong> — they persist across
                  many scales. Points near the diagonal are noise.
                </p>
              </div>

              <div className="explain-item">
                <strong> H0 — Connected Components</strong>
                <p>
                  Counts distinct connected regions in the MRI edges. A high H0 count means the
                  tissue has many separate structural elements. In knee MRI, this reflects the
                  complexity of bone, cartilage, and ligament boundaries.
                </p>
              </div>

              <div className="explain-item">
                <strong> H1 — Loops (Holes)</strong>
                <p>
                  Detects circular or enclosed structures. In knee MRI, these correspond to cross-sections
                  of ligaments, menisci boundaries, vascular structures, and joint spaces. A higher
                  H1 count may indicate more complex or disrupted tissue geometry — potentially
                  consistent with tears, edema, or structural irregularity.
                </p>
              </div>

              <div className="explain-item">
                <strong> Persistence Entropy</strong>
                <p>
                  Measures the diversity of feature lifetimes. Higher entropy means the topological
                  features have varied significance levels, suggesting complex, heterogeneous tissue
                  structure. Lower entropy indicates more uniform, regular anatomy.
                </p>
              </div>

              <div className="explain-item">
                <strong> Lifetime (Mean / Max)</strong>
                <p>
                  The "lifetime" of a feature (death − birth) indicates its significance. Long-lived
                  features represent robust anatomical structures. Short-lived features are typically
                  noise or fine-grained texture. A high max lifetime suggests a dominant structural pattern.
                </p>
              </div>

              <div className="explain-note">
                <strong>Note:</strong> All findings should be
                correlated with clinical examination and standard radiology interpretation.
              </div>
            </div>
          )}
        </div>

        {/* ========== GRAD-CAM SECTION ========== */}
        <div className="section-card">
          <div className="section-header">
            <LocalFireDepartmentIcon />
            <h3>Grad-CAM Attention Analysis</h3>

            {/* Plane Selector — uses TDA's available planes before Grad-CAM has run */}
            {(gcamAvailablePlanes.length > 0 || availablePlanes.length > 0) && (
              <select
                className="plane-select"
                value={gcamPlane || (availablePlanes.length > 0 ? availablePlanes[0] : "")}
                onChange={(e) => {
                  const p = e.target.value;
                  setGcamPlane(p);
                  fetchGcamInfo(p);
                }}
                disabled={gcamLoading}
              >
                {(gcamAvailablePlanes.length > 0 ? gcamAvailablePlanes : availablePlanes).map((p) => (
                  <option key={p} value={p}>
                    {p.charAt(0).toUpperCase() + p.slice(1)} Plane
                  </option>
                ))}
              </select>
            )}
          </div>

          {/* Grad-CAM 3-panel image */}
          <div className="tda-graph-large">
            {gcamLoading ? (
              <div className="tda-loading">
                <div className="spinner" />
                <span>Running ViT Attention-CAM for {gcamPlane || "..."} plane...</span>
              </div>
            ) : gcamError ? (
              <div className="tda-error-msg">{gcamError}</div>
            ) : gcamInfo ? (
              <img
                key={gcamKey}
                src={`${API_BASE}/grad-cam/${patient.patientId}?plane=${gcamPlane || "axial"}&k=${gcamKey}`}
                alt="Grad-CAM Visualization"
                className="tda-diagram-img"
                crossOrigin="anonymous"
              />
            ) : (
              <div className="tda-loading" style={{ flexDirection: "column", gap: "12px" }}>
                <p style={{ color: "#7dd3c0", fontSize: "14px" }}>
                  Click below to run ViT Attention Analysis
                </p>
                <button
                  className="green-btn"
                  onClick={() => fetchGcamInfo(gcamPlane || (availablePlanes.length > 0 ? availablePlanes[0] : "axial"))}
                  disabled={gcamLoading}
                >
                  Run Attention Analysis
                </button>
              </div>
            )}
          </div>

          {/* CAM Attention Stats */}
          <div className="tda-metrics">
            <div>
              <span>Peak Attention</span>
              <strong>{gcamLoading ? "..." : gcamInfo ? `${(gcamInfo.cam_max * 100).toFixed(1)}%` : "—"}</strong>
            </div>
            <div>
              <span>Mean Attention</span>
              <strong>{gcamLoading ? "..." : gcamInfo ? `${(gcamInfo.cam_mean * 100).toFixed(1)}%` : "—"}</strong>
            </div>
            <div>
              <span>Active Region</span>
              <strong>{gcamLoading ? "..." : gcamInfo ? `${gcamInfo.active_region_pct}%` : "—"}</strong>
            </div>
            <div>
              <span>Attention Threshold</span>
              <strong>{gcamLoading ? "..." : gcamInfo ? gcamInfo.threshold : "—"}</strong>
            </div>
          </div>

          {/* Status Badge */}
          {gcamInfo && !gcamInfo.error && (
            <div className={`tda-warning ${gcamInfo.active_region_pct > 40 ? "" : "tda-normal"}`}>
              {gcamInfo.active_region_pct > 40
                ? "⚠ High Attention Region Detected"
                : "✓ Localized Attention Pattern"}
              <p>
                {gcamInfo.active_region_pct > 40
                  ? `${gcamInfo.active_region_pct}% of the MRI slice has elevated ViT attention — may indicate areas of structural interest.`
                  : `${gcamInfo.active_region_pct}% active region — attention is well-localized on specific anatomical structures.`}
              </p>
              <p className="tda-plane-note">
                {(gcamPlane || "axial").toUpperCase()} plane • slice {gcamInfo.slice_index}
              </p>
            </div>
          )}

          {/* Explanation Toggle */}
          <button
            className="explain-toggle"
            onClick={() => setShowGcamExplanation(!showGcamExplanation)}
          >
            <InfoOutlinedIcon fontSize="small" />
            {showGcamExplanation ? "Hide Explanation" : "What does this mean?"}
          </button>

          {showGcamExplanation && (
            <div className="tda-explanation">
              <h4>Understanding Grad-CAM / Attention-CAM</h4>
              <p>
                Attention-CAM uses the <strong>Vision Transformer (ViT)</strong> model's internal
                attention weights to highlight which regions of the MRI slice the AI focuses on
                when extracting diagnostic features.
              </p>

              <div className="explain-item">
                <strong>Jet Heatmap (center panel)</strong>
                <p>
                  Blue = low attention, green = moderate, red = high. Shows the full
                  continuous attention distribution across the MRI slice.
                </p>
              </div>

              <div className="explain-item">
                <strong>Red Overlay (right panel)</strong>
                <p>
                  Pixels above the attention threshold (0.3) are highlighted in red.
                  These regions are where the ViT model concentrates its features
                  — often corresponding to ligaments, menisci, or regions with tissue changes.
                </p>
              </div>

              <div className="explain-item">
                <strong>Active Region %</strong>
                <p>
                  Percentage of the MRI slice with attention above the threshold.
                  A high value (&gt;40%) may indicate diffuse abnormality or complex anatomy.
                  A low value indicates the model is focused on specific, localized areas.
                </p>
              </div>

              <div className="explain-note">
                <strong>Note:</strong> Attention-CAM highlights are indicative of AI focus, not
                definitive pathology markers. All findings must be correlated clinically.
              </div>
            </div>
          )}
        </div>

      </div>
    </div>
  );
};

export default PatientDetails;