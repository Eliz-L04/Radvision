import React, { useState, useEffect } from "react";
import { useParams, useLocation, useNavigate } from "react-router-dom";
import jsPDF from "jspdf";
import "./RadiologyReport.css";

const API_BASE = "http://127.0.0.1:5000";

const RadiologyReport = () => {
  const { id } = useParams();
  const location = useLocation();
  const navigate = useNavigate();

  const { patient, report } = location.state || {};

  // Editable fields
  const [editTechnique, setEditTechnique] = useState("");
  const [editFindings, setEditFindings] = useState("");
  const [editImpressions, setEditImpressions] = useState("");
  const [editNotes, setEditNotes] = useState("");
  const [isFinalized, setIsFinalized] = useState(false);

  // Signature state (radiologist full name)
  const [radiologistName, setRadiologistName] = useState("");
  const hasSigned = radiologistName.trim().length > 0;

  // Initialize editable fields from report data
  useEffect(() => {
    if (report) {
      setEditTechnique(report.technique || "");
      setEditFindings(
        (report.findings || []).join("\n")
      );
      setEditImpressions(
        (report.impression || []).join("\n")
      );
      // Auto-generate radiologist notes from predictions
      const preds = report.predictions || {};
      let notes = "";
      if (preds.abnormal?.label === "POSITIVE") {
        notes += `AI analysis detected abnormalities with ${(preds.abnormal.probability * 100).toFixed(0)}% confidence. `;
      } else {
        notes += "AI analysis indicates no significant abnormalities. ";
      }
      if (preds.acl?.label === "POSITIVE") {
        notes += `ACL signal abnormality (${(preds.acl.probability * 100).toFixed(0)}%). `;
      } else {
        notes += "ACL intact. ";
      }
      if (preds.meniscus?.label === "POSITIVE") {
        notes += `Meniscal changes (${(preds.meniscus.probability * 100).toFixed(0)}%). `;
      } else {
        notes += "Menisci normal. ";
      }
      notes += "Clinical correlation recommended.";
      setEditNotes(notes);
    }
  }, [report]);

  // ── Finalize Report ──
  const handleFinalize = async () => {
    if (!hasSigned) {
      alert("Please enter the radiologist's full name before finalizing.");
      return;
    }

    try {
      const patientInfo = report?.patient || patient || {};
      await fetch(`${API_BASE}/finalize-report`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          patientId: id,
          radiologistName: radiologistName.trim(),
          name: patientInfo.name,
          age: patientInfo.age,
          gender: patientInfo.gender,
        }),
      });
    } catch (err) {
      console.warn("Failed to save finalization to server:", err);
    }

    setIsFinalized(true);
  };

  // ── Export PDF ──
  const handleExportPDF = () => {
    if (!isFinalized) {
      alert("Please finalize the report before exporting.");
      return;
    }

    const patientInfo = report?.patient || patient || {};
    const doc = new jsPDF("p", "mm", "a4");
    const pageW = doc.internal.pageSize.getWidth();
    const margin = 18;
    const contentW = pageW - margin * 2;
    let y = 20;

    // ── Header ──
    doc.setFillColor(16, 185, 129);
    doc.rect(0, 0, pageW, 14, "F");
    doc.setFont("helvetica", "bold");
    doc.setFontSize(14);
    doc.setTextColor(255, 255, 255);
    doc.text("RadVision — Radiology Report", pageW / 2, 9, { align: "center" });

    y = 24;

    // ── Patient Info Table ──
    doc.setFontSize(9);
    doc.setTextColor(80, 80, 80);
    doc.setFont("helvetica", "normal");

    doc.setDrawColor(200, 200, 200);
    doc.setLineWidth(0.3);
    doc.rect(margin, y, contentW, 24);
    doc.line(margin, y + 12, margin + contentW, y + 12);
    doc.line(margin + contentW / 2, y, margin + contentW / 2, y + 24);

    doc.setFont("helvetica", "bold");
    doc.text("Patient Name:", margin + 3, y + 5);
    doc.setFont("helvetica", "normal");
    doc.text(patientInfo.name || "N/A", margin + 35, y + 5);

    doc.setFont("helvetica", "bold");
    doc.text("Date:", margin + contentW / 2 + 3, y + 5);
    doc.setFont("helvetica", "normal");
    doc.text(new Date().toLocaleDateString(), margin + contentW / 2 + 18, y + 5);

    doc.setFont("helvetica", "bold");
    doc.text("Age / Gender:", margin + 3, y + 17);
    doc.setFont("helvetica", "normal");
    const gender = patientInfo.gender
      ? patientInfo.gender.charAt(0).toUpperCase()
      : "?";
    doc.text(`${patientInfo.age || "N/A"} / ${gender}`, margin + 35, y + 17);

    doc.setFont("helvetica", "bold");
    doc.text("Case ID:", margin + contentW / 2 + 3, y + 17);
    doc.setFont("helvetica", "normal");
    doc.text(`MR-${id}`, margin + contentW / 2 + 23, y + 17);

    y += 32;

    // ── Title ──
    doc.setFont("helvetica", "bold");
    doc.setFontSize(13);
    doc.setTextColor(30, 30, 30);
    doc.text("MRI SCAN OF KNEE JOINT", pageW / 2, y, { align: "center" });
    y += 10;

    // ── Helper: section rendering ──
    const renderSection = (title, content, isBulletList = false) => {
      // Check if we need a new page
      if (y > 260) {
        doc.addPage();
        y = 20;
      }

      doc.setFont("helvetica", "bold");
      doc.setFontSize(10);
      doc.setTextColor(16, 120, 90);
      doc.text(title, margin, y);
      y += 6;

      doc.setFont("helvetica", "normal");
      doc.setFontSize(9);
      doc.setTextColor(50, 50, 50);

      if (isBulletList && Array.isArray(content)) {
        content.forEach((item) => {
          const lines = doc.splitTextToSize(`• ${item}`, contentW - 8);
          lines.forEach((line) => {
            if (y > 275) { doc.addPage(); y = 20; }
            doc.text(line, margin + 4, y);
            y += 4.5;
          });
          y += 1;
        });
      } else {
        const text = typeof content === "string" ? content : (content || []).join("\n");
        const lines = doc.splitTextToSize(text, contentW - 4);
        lines.forEach((line) => {
          if (y > 275) { doc.addPage(); y = 20; }
          doc.text(line, margin + 2, y);
          y += 4.5;
        });
      }

      y += 4;
    };

    // ── Sections ──
    renderSection("TECHNIQUE:", editTechnique);
    renderSection("FINDINGS:", editFindings.split("\n").filter(Boolean), true);
    renderSection("IMPRESSIONS:", editImpressions.split("\n").filter(Boolean), true);
    renderSection("RADIOLOGIST NOTES:", editNotes);

    // ── Divider ──
    y += 4;
    doc.setDrawColor(180, 180, 180);
    doc.setLineWidth(0.3);
    doc.line(margin, y, margin + contentW, y);
    y += 8;

    // ── Digital Signature (Radiologist Name) ──
    if (radiologistName.trim()) {
      if (y > 250) { doc.addPage(); y = 20; }

      doc.setFont("helvetica", "bold");
      doc.setFontSize(9);
      doc.setTextColor(50, 50, 50);
      doc.text("Reporting Radiologist:", margin, y);
      y += 7;

      doc.setFont("helvetica", "italic");
      doc.setFontSize(13);
      doc.setTextColor(30, 30, 30);
      doc.text(radiologistName.trim(), margin + 2, y);
      y += 5;

      doc.setDrawColor(50, 50, 50);
      doc.setLineWidth(0.4);
      doc.line(margin, y, margin + 65, y);
      y += 8;

      doc.setFont("helvetica", "normal");
      doc.setFontSize(8);
      doc.setTextColor(100, 100, 100);
      doc.text(`Finalized on: ${new Date().toLocaleString()}`, margin, y);
      y += 5;
      doc.text("This report has been digitally signed and finalized.", margin, y);
    }

    // ── Footer ──
    const totalPages = doc.internal.getNumberOfPages();
    for (let i = 1; i <= totalPages; i++) {
      doc.setPage(i);
      doc.setFontSize(7);
      doc.setTextColor(140, 140, 140);
      doc.text(
        "RadVision AI | AI-Assisted Radiology Report | CONFIDENTIAL",
        pageW / 2,
        290,
        { align: "center" }
      );
      doc.text(`Page ${i} of ${totalPages}`, pageW - margin, 290, {
        align: "right",
      });
    }

    doc.save(`RadVision_Report_${id}.pdf`);
  };

  // No data — show fallback
  if (!report) {
    return (
      <div className="rad-report-container">
        <div className="rad-report-header">
          <div>
            <div className="case-id">CASE #{id}</div>
            <h1>Report Not Available</h1>
            <p>
              No AI-generated report data found. Please go back and click
              "Generate Report" from the Patient Details page.
            </p>
          </div>
          <div className="header-actions">
            <button className="primary-btn" onClick={() => navigate(-1)}>
              ← Go Back
            </button>
          </div>
        </div>
      </div>
    );
  }

  const {
    predictions = {},
    tda_summary = {},
    tda_per_plane = {},
    generated_at = "",
  } = report;

  const patientInfo = report.patient || patient || {};

  return (
    <div className="rad-report-container">

      {/* HEADER */}
      <div className="rad-report-header">
        <div>
          <div className="case-id">CASE #{id}</div>
          <h1>AI-Generated Radiology Report</h1>
          <p>
            AI-assisted diagnostic report for{" "}
            <strong>{patientInfo.name || "Patient"}</strong> — Knee MRI Study.
            {isFinalized && <span className="finalized-tag"> ✅ FINALIZED</span>}
          </p>
        </div>

        <div className="header-actions">
          <button className="outline-btn" onClick={() => navigate(-1)}>
            ← Back
          </button>
          <button
            className={`outline-btn ${!isFinalized ? "btn-disabled" : ""}`}
            onClick={handleExportPDF}
            disabled={!isFinalized}
            title={!isFinalized ? "Finalize the report first" : "Export as PDF"}
          >
            Export PDF
          </button>
          <button
            className={`primary-btn ${!hasSigned ? "btn-disabled" : ""}`}
            onClick={handleFinalize}
            disabled={!hasSigned || isFinalized}
            title={
              isFinalized
                ? "Already finalized"
                : !hasSigned
                ? "Sign first"
                : "Finalize report"
            }
          >
            {isFinalized ? "✅ Finalized" : "Finalize Report"}
          </button>
        </div>
      </div>

      {/* MAIN CONTENT GRID */}
      <div className="rad-grid">

        {/* LEFT COLUMN */}
        <div className="left-panel">

          {/* Technique — Editable */}
          <div className="card">
            <div className="card-header">
              <h2>Technique</h2>
              <span className="status editable-badge">✏ Editable</span>
            </div>
            <div className="card-content">
              <textarea
                className="editable-textarea"
                value={editTechnique}
                onChange={(e) => setEditTechnique(e.target.value)}
                rows={3}
                disabled={isFinalized}
                placeholder="Enter MRI technique description..."
              />
            </div>
          </div>

          {/* Findings — Editable */}
          <div className="card">
            <div className="card-header">
              <h2>Findings</h2>
              <span className="status editable-badge">✏ Editable</span>
            </div>
            <div className="card-content">
              <textarea
                className="editable-textarea"
                value={editFindings}
                onChange={(e) => setEditFindings(e.target.value)}
                rows={8}
                disabled={isFinalized}
                placeholder="Enter findings (one per line)..."
              />
              <div className="autosave">
                Each line will appear as a separate finding bullet
              </div>
            </div>
          </div>

          {/* Impressions — Editable */}
          <div className="card">
            <div className="card-header">
              <h2>Impressions</h2>
              <span className="status editable-badge">✏ Editable</span>
            </div>
            <div className="card-content">
              <textarea
                className="editable-textarea"
                value={editImpressions}
                onChange={(e) => setEditImpressions(e.target.value)}
                rows={5}
                disabled={isFinalized}
                placeholder="Enter clinical impressions (one per line)..."
              />
              <div className="autosave">
                Each line will appear as a separate impression bullet
              </div>
            </div>
          </div>

          {/* Radiologist Notes — Editable */}
          <div className="card">
            <div className="card-header">
              <h2>Radiologist Notes</h2>
              <span className="status editable-badge">✏ Editable</span>
            </div>
            <div className="card-content">
              <textarea
                className="editable-textarea"
                value={editNotes}
                onChange={(e) => setEditNotes(e.target.value)}
                rows={4}
                disabled={isFinalized}
                placeholder="Enter additional radiologist notes..."
              />
            </div>
          </div>

          {/* Digital Signature — Radiologist Name */}
          <div className="card signature-card">
            <div className="card-header">
              <h2>Digital Signature</h2>
            </div>
            <div className="card-content">
              <p className="sig-instructions">
                Enter the reporting radiologist's full name below. Both Finalize
                and Export PDF require a valid name.
              </p>
              <input
                type="text"
                className="sig-name-input"
                value={radiologistName}
                onChange={(e) => setRadiologistName(e.target.value)}
                disabled={isFinalized}
                placeholder="Dr. Full Name"
              />
              {radiologistName.trim() && (
                <div className="sig-preview">
                  <span className="sig-preview-label">Signed as:</span>
                  <span className="sig-preview-name">{radiologistName}</span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* RIGHT COLUMN */}
        <div className="right-panel">

          {/* AI Confidence Summary */}
          {Object.keys(predictions).length > 0 && (
            <div className="side-card">
              <h3>AI Confidence Summary</h3>
              <div className="prediction-grid-side">
                {Object.entries(predictions).map(([task, info]) => {
                  const isPositive = info.label === "POSITIVE";
                  const prob = (info.probability * 100).toFixed(1);
                  return (
                    <div key={task} className="pred-row-side">
                      <div className="pred-info-side">
                        <span className="pred-task-side">{task.toUpperCase()}</span>
                        <span
                          className={`pred-badge-side ${
                            isPositive ? "pred-pos" : "pred-neg"
                          }`}
                        >
                          {info.label}
                        </span>
                      </div>
                      <div className="pred-bar-wrap-side">
                        <div
                          className={`pred-bar-side ${
                            isPositive ? "bar-pos" : "bar-neg"
                          }`}
                          style={{ width: `${prob}%` }}
                        />
                      </div>
                      <span className="pred-prob-side">{prob}%</span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* TDA Summary */}
          {Object.keys(tda_per_plane).length > 0 && (
            <div className="side-card">
              <h3>TDA Analysis</h3>
              {Object.entries(tda_per_plane).map(([plane, data]) => (
                <div key={plane} className="plane-row">
                  <span className="plane-name">{plane.toUpperCase()}</span>
                  <span>H1: {data.h1_count}</span>
                  <span>Entropy: {data.persistence_entropy}</span>
                </div>
              ))}
            </div>
          )}

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
              <span>Pipeline</span>
              <strong>ViT + TDA + RF</strong>
            </div>
          </div>

          {/* Patient Info */}
          <div className="side-card">
            <h3>Patient Information</h3>
            <div className="info-row">
              <span>Name</span>
              <strong>{patientInfo.name || "N/A"}</strong>
            </div>
            <div className="info-row">
              <span>Age</span>
              <strong>{patientInfo.age || "N/A"} Years</strong>
            </div>
            <div className="info-row">
              <span>Gender</span>
              <strong>
                {patientInfo.gender
                  ? patientInfo.gender.charAt(0).toUpperCase() +
                    patientInfo.gender.slice(1)
                  : "N/A"}
              </strong>
            </div>
          </div>

          {/* Disclaimer */}
          <div className="side-card disclaimer-card">
            <h3>⚠ Disclaimer</h3>
            <p>
              AI-generated report (ViT + TDA + RF + Claude).
              Must be reviewed and confirmed by a qualified radiologist
              before clinical use.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RadiologyReport;
