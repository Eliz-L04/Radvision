import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import "./Reports.css";

/* MUI ICONS */
import FilterListIcon from "@mui/icons-material/FilterList";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import HistoryIcon from "@mui/icons-material/History";
import PendingActionsIcon from "@mui/icons-material/PendingActions";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import SearchIcon from "@mui/icons-material/Search";

const API_BASE = "http://127.0.0.1:5000";

const Reports = () => {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState("recent");
  const [search, setSearch] = useState("");
  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(true);

  // Fetch saved reports from backend
  useEffect(() => {
    setLoading(true);
    fetch(`${API_BASE}/saved-reports`)
      .then((res) => res.json())
      .then((data) => {
        setReports(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Failed to fetch reports:", err);
        setLoading(false);
      });
  }, []);

  // Categorize reports by status
  const categorized = {
    recent: reports,
    pending: reports.filter((r) => r.status === "In Progress" || r.status === "Awaiting Review"),
    completed: reports.filter((r) => r.status === "Reported" || r.status === "Completed"),
  };

  const filteredData = categorized[activeTab].filter(
    (item) =>
      (item.name || "").toLowerCase().includes(search.toLowerCase()) ||
      (item.patient_id || "").toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="reports-page">

      {/* HEADER */}
      <div className="reports-header">
        <h1>Reports</h1>
        <p>
          Browse, review, and manage patient imaging reports for orthopedic knee evaluations.
        </p>
      </div>

      {/* TOPBAR */}
      <div className="reports-topbar">

        {/* TABS */}
        <div className="reports-tabs">
          <button
            className={activeTab === "recent" ? "active" : ""}
            onClick={() => setActiveTab("recent")}
          >
            <HistoryIcon style={{ fontSize: 16, marginRight: 6 }} />
            All Reports ({categorized.recent.length})
          </button>

          <button
            className={activeTab === "pending" ? "active" : ""}
            onClick={() => setActiveTab("pending")}
          >
            <PendingActionsIcon style={{ fontSize: 16, marginRight: 6 }} />
            In Progress ({categorized.pending.length})
          </button>

          <button
            className={activeTab === "completed" ? "active" : ""}
            onClick={() => setActiveTab("completed")}
          >
            <CheckCircleIcon style={{ fontSize: 16, marginRight: 6 }} />
            Completed ({categorized.completed.length})
          </button>
        </div>

        {/* SEARCH + FILTERS */}
        <div className="reports-filters">

          <div className="search-bar">
            <SearchIcon style={{ fontSize: 16 }} />
            <input
              type="text"
              placeholder="Search patient or ID..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </div>

          <button className="filter-btn">
            Knee MRI
            <ExpandMoreIcon style={{ fontSize: 16, marginLeft: 4 }} />
          </button>

          <button className="filter-btn">
            <FilterListIcon style={{ fontSize: 16, marginRight: 6 }} />
            More Filters
          </button>
        </div>
      </div>

      {/* TABLE CARD */}
      <div className="reports-card">
        <table>
          <thead>
            <tr>
              <th>Patient ID</th>
              <th>Name</th>
              <th>Age</th>
              <th>Gender</th>
              <th>Scan</th>
              <th>Date</th>
              <th>Status</th>
              <th>Actions</th>
            </tr>
          </thead>

          <tbody>
            {loading ? (
              <tr>
                <td colSpan="8" style={{ textAlign: "center", padding: "40px", color: "#7dd3c0" }}>
                  Loading reports...
                </td>
              </tr>
            ) : filteredData.length === 0 ? (
              <tr>
                <td colSpan="8" style={{ textAlign: "center", padding: "40px", color: "#7dd3c0" }}>
                  {search ? "No matching reports found" : "No reports yet. Save progress from Patient Details to see reports here."}
                </td>
              </tr>
            ) : (
              filteredData.map((item) => (
                <tr key={item.patient_id}>
                  <td>{item.patient_id}</td>
                  <td>{item.name}</td>
                  <td>{item.age}</td>
                  <td>{item.gender}</td>
                  <td>{item.scan_type || "Knee MRI"}</td>
                  <td>{item.updated_at || item.created_at || "—"}</td>

                  <td>
                    <span className={`status ${(item.status || "").replace(/ /g, "-").toLowerCase()}`}>
                      {item.status}
                    </span>
                  </td>

                  <td>
                    <button
                      className="view-btn"
                      onClick={() =>
                        navigate(`/patient/${item.patient_id}`, {
                          state: {
                            patient: {
                              patientId: item.patient_id,
                              name: item.name,
                              age: item.age,
                              gender: item.gender,
                            },
                          },
                        })
                      }
                    >
                      View
                    </button>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>

        {/* FOOTER */}
        <div className="reports-footer">
          <span>
            {filteredData.length} report{filteredData.length !== 1 ? "s" : ""}
          </span>
        </div>
      </div>

    </div>
  );
};

export default Reports;
