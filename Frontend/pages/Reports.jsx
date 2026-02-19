import React, { useState } from "react";
import "./Reports.css";

/* MUI ICONS */
import FilterListIcon from "@mui/icons-material/FilterList";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import HistoryIcon from "@mui/icons-material/History";
import PendingActionsIcon from "@mui/icons-material/PendingActions";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import SearchIcon from "@mui/icons-material/Search";

const Reports = () => {
  const [activeTab, setActiveTab] = useState("recent");
  const [search, setSearch] = useState("");

  const data = {
    recent: [
      { id: "RAD-2041", name: "John Doe", age: 45, gender: "Male", scan: "Knee MRI (Left)", date: "2026-01-08", status: "Awaiting Review" },
      { id: "RAD-2042", name: "Sarah Lee", age: 38, gender: "Female", scan: "Knee MRI (Right)", date: "2026-01-07", status: "Reported" },
      { id: "RAD-2043", name: "Michael Thompson", age: 52, gender: "Male", scan: "Knee MRI (ACL Protocol)", date: "2026-01-07", status: "In Progress" },
      { id: "RAD-2044", name: "Elena Rodriguez", age: 41, gender: "Female", scan: "Knee MRI (Meniscus Study)", date: "2026-01-06", status: "Reported" },
      { id: "RAD-2045", name: "David Kim", age: 60, gender: "Male", scan: "Knee MRI (Post-Op)", date: "2026-01-06", status: "Urgent" }
    ],
    pending: [],
    completed: []
  };

  const filteredData = data[activeTab].filter(
    (item) =>
      item.name.toLowerCase().includes(search.toLowerCase()) ||
      item.id.toLowerCase().includes(search.toLowerCase())
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
            Recent Uploads
          </button>

          <button
            className={activeTab === "pending" ? "active" : ""}
            onClick={() => setActiveTab("pending")}
          >
            <PendingActionsIcon style={{ fontSize: 16, marginRight: 6 }} />
            Pending Reports
          </button>

          <button
            className={activeTab === "completed" ? "active" : ""}
            onClick={() => setActiveTab("completed")}
          >
            <CheckCircleIcon style={{ fontSize: 16, marginRight: 6 }} />
            Completed Reports
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
            {filteredData.map((item) => (
              <tr key={item.id}>
                <td>{item.id}</td>
                <td>{item.name}</td>
                <td>{item.age}</td>
                <td>{item.gender}</td>
                <td>{item.scan}</td>
                <td>{item.date}</td>

                <td>
                  <span className={`status ${item.status.replace(" ", "-").toLowerCase()}`}>
                    {item.status}
                  </span>
                </td>

                <td>
                  <button className="view-btn">View</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        {/* FOOTER + PAGINATION */}
        <div className="reports-footer">
          <span>Showing 1–5 of 42 orthopedic cases</span>

          <div className="reports-pagination">
            <button className="page-btn">‹</button>
            <button className="page-btn active">1</button>
            <button className="page-btn">2</button>
            <button className="page-btn">3</button>
            <button className="page-btn">›</button>
          </div>
        </div>
      </div>

    </div>
  );
};

export default Reports;
