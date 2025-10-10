import React from "react";
import { useNavigate } from "react-router-dom";
import "./Dashboard.css";
import { Upload, FileText, Users, Settings } from "lucide-react";

const Dashboard = () => {
  const navigate = useNavigate();

  const stats = [
    { title: "Recent Uploads", value: 12, description: "Total scans uploaded recently" },
    { title: "Pending Reports", value: 3, description: "Reports awaiting your review" },
    { title: "Completed Reports", value: 28, description: "Reports finalized this month" },
  ];

  return (
    <div className="dashboard-layout">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <h2 className="logo">RadVision</h2>
        </div>
        <nav className="sidebar-nav">
          <button onClick={() => navigate("/dashboard")}>🏠 Dashboard</button>
          <button onClick={() => navigate("/upload")}>📤 Upload Scan</button>
          <button onClick={() => navigate("/reports")}>📄 Reports</button>
          <button onClick={() => navigate("/patients")}>👥 Patients</button>
          <button onClick={() => navigate("/settings")}>⚙️ Settings</button>
        </nav>
      </aside>

      {/* Main Content */}
      <main className="dashboard-main">
        <header className="dashboard-header">
          <h1>Welcome, Doctor Emily</h1>
          <p>Here's a quick overview of your activities.</p>
        </header>

        <div className="stats-grid">
          {stats.map((stat, index) => (
            <div key={index} className="stat-card">
              <h4>{stat.title}</h4>
              <h3>{stat.value}</h3>
              <p>{stat.description}</p>
            </div>
          ))}
        </div>

        <section className="quick-actions">
          <h3>Quick Actions</h3>
          <div className="action-buttons">
            <button className="primary-btn" onClick={() => navigate("/upload")}>
              <Upload size={18} /> Upload DICOM
            </button>
            <button className="secondary-btn" onClick={() => navigate("/reports")}>
              <FileText size={18} /> View Reports
            </button>
          </div>
        </section>

        <footer className="dashboard-footer">
          <div>
            <a href="#">About</a> • <a href="#">Help</a>
          </div>
          <div className="socials">
            <i className="fab fa-linkedin"></i>
            <i className="fab fa-twitter"></i>
            <i className="fab fa-facebook"></i>
          </div>
        </footer>
      </main>
    </div>
  );
};

export default Dashboard;

