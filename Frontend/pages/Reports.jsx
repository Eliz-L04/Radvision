import React, { useState } from "react";
import "./Reports.css";

const Reports = () => {
  const [activeTab, setActiveTab] = useState("recent");

  const data = {
    recent: [
      { id: 1, patient: "John Doe", scan: "CT Brain", date: "2026-01-08" },
      { id: 2, patient: "Sarah Lee", scan: "MRI Spine", date: "2026-01-07" }
    ],
    pending: [
      { id: 3, patient: "Mark Smith", scan: "CT Chest", date: "2026-01-06" }
    ],
    completed: [
      { id: 4, patient: "Emily Brown", scan: "MRI Knee", date: "2026-01-05" },
      { id: 5, patient: "Ava Wilson", scan: "CT Abdomen", date: "2026-01-03" }
    ],
  };

  return (
    <div className="reports-page">
      <h1>Reports</h1>
      <p>Browse, review, and manage patient imaging reports.</p>

      <div className="reports-tabs">
        <button className={activeTab === "recent" ? "active" : ""} onClick={() => setActiveTab("recent")}>Recent Uploads</button>
        <button className={activeTab === "pending" ? "active" : ""} onClick={() => setActiveTab("pending")}>Pending Reports</button>
        <button className={activeTab === "completed" ? "active" : ""} onClick={() => setActiveTab("completed")}>Completed Reports</button>
      </div>

      <div className="reports-table">
        <table>
          <thead>
            <tr>
              <th>Patient</th>
              <th>Scan Type</th>
              <th>Date</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {data[activeTab].map(item => (
              <tr key={item.id}>
                <td>{item.patient}</td>
                <td>{item.scan}</td>
                <td>{item.date}</td>
                <td>
                  <button className="view-btn">View Report</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default Reports;
