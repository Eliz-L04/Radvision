import React, { useState } from 'react';
import { useParams } from 'react-router-dom';
import './RadiologyReport.css';

const RadiologyReport = () => {
  const { id } = useParams();
  const [reportData, setReportData] = useState({
    findings: `Chest X-ray (CXR) performed on 2024-07-26 shows clear lung fields without evidence of focal consolidation, pleural effusion, or pneumothorax. The cardiac silhouette is normal in size and contour. Mediastinal and hair structures appear unremarkable. No acute osseous abnormalities are identified within the visualized thorax. Chronic changes may include minimal degenerative changes in the thoracic spine.`,
    notes: `Patient presented with non-specific chest pain. Follow-up with clinical correlation recommended. Consider ECG if symptoms persist.`,
    impressions: `No acute cardiopulmonary pathology. Normal chest X-ray. Findings are consistent with a healthy adult chest.`
  });

  const [actions, setActions] = useState({
    saveDraft: false,
    finalize: false,
    exportPdf: true,
    sendToPacs: false
  });

  const handleInputChange = (field, value) => {
    setReportData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleActionToggle = (action) => {
    setActions(prev => ({
      ...prev,
      [action]: !prev[action]
    }));
  };

  return (
    <div className="report-container">
      <header className="report-header">
        <h1>Draft Radiology Report</h1>
        <div className="report-actions">
          <button className="action-btn save-btn">Save Draft</button>
          <div className="action-checkboxes">
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={actions.finalize}
                onChange={() => handleActionToggle('finalize')}
              />
              Finalize Report
            </label>
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={actions.exportPdf}
                onChange={() => handleActionToggle('exportPdf')}
              />
              Export PDF
            </label>
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={actions.sendToPacs}
                onChange={() => handleActionToggle('sendToPacs')}
              />
              Send to PACS/EHR
            </label>
          </div>
        </div>
      </header>

      <section className="report-section">
        <h2>Findings</h2>
        <textarea
          className="report-textarea findings"
          value={reportData.findings}
          onChange={(e) => handleInputChange('findings', e.target.value)}
          placeholder="Enter findings from the scan..."
        />
      </section>

      <section className="report-section">
        <h2>Radiologist Notes</h2>
        <textarea
          className="report-textarea notes"
          value={reportData.notes}
          onChange={(e) => handleInputChange('notes', e.target.value)}
          placeholder="Add additional notes or observations..."
        />
      </section>

      <section className="report-section">
        <h2>Impressions</h2>
        <textarea
          className="report-textarea impressions"
          value={reportData.impressions}
          onChange={(e) => handleInputChange('impressions', e.target.value)}
          placeholder="Enter clinical impressions and recommendations..."
        />
      </section>

      <div className="report-footer">
        <button className="submit-btn">Submit Final Report</button>
        <div className="footer-info">
          <p>Company Resources</p>
        </div>
      </div>
    </div>
  );
};

export default RadiologyReport;