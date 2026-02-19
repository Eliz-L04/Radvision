// Dashboard.jsx
import React from "react";
import { useNavigate } from "react-router-dom";
import DashboardIcon from "@mui/icons-material/Dashboard";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import DescriptionIcon from "@mui/icons-material/Description";
import PeopleIcon from "@mui/icons-material/People";
import SettingsIcon from "@mui/icons-material/Settings";
import InsertDriveFileIcon from "@mui/icons-material/InsertDriveFile";
import "./Dashboard.css";

const Dashboard = () => {
  const navigate = useNavigate();

  return (
    <div className="dash-layout">
      {/* SIDEBAR */}
      <aside className="dash-sidebar">
        <div>
          <div className="dash-logo">RadVision</div>

          <nav className="dash-nav">
            <button className="active" onClick={() => navigate("/dashboard")}>
              <DashboardIcon fontSize="small" /> Dashboard
            </button>

            <button onClick={() => navigate("/upload")}>
              <CloudUploadIcon fontSize="small" /> Upload Scan
            </button>

            <button onClick={() => navigate("/reports")}>
              <DescriptionIcon fontSize="small" /> Reports
            </button>

            <button onClick={() => navigate("/settings")}>
              <SettingsIcon fontSize="small" /> Settings
            </button>
          </nav>
        </div>

        <div className="dash-user">
          <div className="avatar">👩‍⚕️</div>
          <div>
            <div className="name">Dr. Eli Johnson</div>
            <div className="role">Senior Radiologist</div>
          </div>
        </div>
      </aside>

      {/* MAIN */}
      <main className="dash-main">
        <div className="dash-header">
          <h1>Welcome, {localStorage.getItem("username") || "eli"}</h1>
          <p>
            Here’s a comprehensive overview of your clinical activities and
            diagnostic pipeline.
          </p>
        </div>

        {/* STATS */}
        <div className="dash-stats">
          <div className="stat-card">
            <InsertDriveFileIcon className="stat-icon" />
            <div>
              <div className="stat-value">12</div>
              <div className="stat-label">Recent Uploads</div>
            </div>
          </div>

          <div className="stat-card">
            <DescriptionIcon className="stat-icon" />
            <div>
              <div className="stat-value">03</div>
              <div className="stat-label">Pending Reports</div>
            </div>
          </div>

          <div className="stat-card">
            <DashboardIcon className="stat-icon" />
            <div>
              <div className="stat-value">28</div>
              <div className="stat-label">Completed Reports</div>
            </div>
          </div>
        </div>

        {/* QUICK ACTIONS */}
        <div className="quick-box">
          <div className="quick-title">Quick Actions</div>
          <div className="quick-buttons">
            <button
              className="btn-green"
              onClick={() => navigate("/upload")}
            >
              <CloudUploadIcon fontSize="small" /> Upload Scan
            </button>

            <button
              className="btn-outline"
              onClick={() => navigate("/reports")}
            >
              <DescriptionIcon fontSize="small" /> View Reports
            </button>
          </div>
        </div>

      </main>
    </div>
  );
};

export default Dashboard;








