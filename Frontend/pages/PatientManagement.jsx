import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import "./PatientManagement.css";

const PatientManagement = () => {
  const navigate = useNavigate();

  const [search, setSearch] = useState("");
  const [selectedDate, setSelectedDate] = useState("All");
  const [currentPage, setCurrentPage] = useState(1);

  const patientsPerPage = 3;

  const patients = [
    { id: "RAD-2041", name: "Alice Johnson", age: 45, gender: "Female", scan: "Knee MRI", date: "2024-05-12", status: "Reported" },
    { id: "RAD-2042", name: "Robert Williams", age: 62, gender: "Male", scan: "Knee MRI", date: "2024-05-14", status: "Awaiting Review" },
    { id: "RAD-2043", name: "Carol Davis", age: 38, gender: "Female", scan: "Knee MRI", date: "2024-05-14", status: "Priority" },
    { id: "RAD-2044", name: "David Brown", age: 71, gender: "Male", scan: "Knee MRI", date: "2024-05-13", status: "Reported" },
    { id: "RAD-2045", name: "Eve Green", age: 29, gender: "Female", scan: "Knee MRI", date: "2024-05-15", status: "In Progress" },
    { id: "RAD-2046", name: "Frank White", age: 55, gender: "Male", scan: "Knee MRI", date: "2024-05-11", status: "Reported" },
  ];

  // STATUS STYLE
  const getStatusClass = (status) => {
    switch (status) {
      case "Reported": return "status reported";
      case "Awaiting Review": return "status awaiting";
      case "Priority": return "status priority";
      case "In Progress": return "status inprogress";
      default: return "status";
    }
  };

  // 🔎 SEARCH + DATE FILTER
  const filteredPatients = patients.filter((p) => {
    const matchesSearch =
      p.name.toLowerCase().includes(search.toLowerCase()) ||
      p.id.toLowerCase().includes(search.toLowerCase());

    const matchesDate =
      selectedDate === "All" || p.date === selectedDate;

    return matchesSearch && matchesDate;
  });

  // 📄 PAGINATION LOGIC
  const totalPages = Math.ceil(filteredPatients.length / patientsPerPage);
  const indexOfLast = currentPage * patientsPerPage;
  const indexOfFirst = indexOfLast - patientsPerPage;
  const currentPatients = filteredPatients.slice(indexOfFirst, indexOfLast);

  return (
    <div className="patient-page">

      {/* HERO */}
      <div className="patient-hero">
        <div>
          <div className="hero-small">SPECIALIZED DIAGNOSTIC IMAGING</div>
          <h1>Patient Records</h1>
          <h2>Knee MRI Specialization</h2>
          <p>
            Comprehensive database of knee diagnostic scans.
          </p>
        </div>

        <div className="hero-stats">
          <div>
            <span>Total Scans</span>
            <strong>{patients.length}</strong>
          </div>
          <div>
            <span>Pending Review</span>
            <strong>
              {patients.filter(p => p.status === "Awaiting Review").length}
            </strong>
          </div>
        </div>
      </div>

      {/* MAIN CARD */}
      <div className="patient-card">

        {/* HEADER */}
        <div className="queue-header">
          <div>
            <h3>Diagnostic Queue</h3>
            <span>Reviewing {filteredPatients.length} active cases</span>
          </div>

          <div className="queue-controls">
            <input
              className="search-input"
              placeholder="Search patient or ID..."
              value={search}
              onChange={(e) => {
                setSearch(e.target.value);
                setCurrentPage(1);
              }}
            />

            <select>
              <option>Knee MRI</option>
            </select>

            <select
              value={selectedDate}
              onChange={(e) => {
                setSelectedDate(e.target.value);
                setCurrentPage(1);
              }}
            >
              <option value="All">All Dates</option>
              <option value="2024-05-11">May 11</option>
              <option value="2024-05-12">May 12</option>
              <option value="2024-05-13">May 13</option>
              <option value="2024-05-14">May 14</option>
              <option value="2024-05-15">May 15</option>
            </select>

            <button className="export-btn">Export CSV</button>
          </div>
        </div>

        {/* TABLE HEADER */}
        <div className="table-header">
          <div>Patient ID</div>
          <div>Name</div>
          <div>Age</div>
          <div>Gender</div>
          <div>Scan</div>
          <div>Date</div>
          <div>Status</div>
          <div>Actions</div>
        </div>

        {/* ROWS */}
        {currentPatients.map((patient) => (
          <div key={patient.id} className="patient-row">
            <div>{patient.id}</div>
            <div className="patient-name">{patient.name}</div>
            <div>{patient.age}</div>
            <div>{patient.gender}</div>
            <div><span className="scan-badge">{patient.scan}</span></div>
            <div>{patient.date}</div>
            <div>
              <span className={getStatusClass(patient.status)}>
                {patient.status}
              </span>
            </div>
            <div className="action-buttons">
              <button className="view-btn">View</button>
              <button
                className="report-btn"
                onClick={() => navigate(`/patient/${patient.id}`)}
              >
                Report
              </button>
            </div>
          </div>
        ))}

        {/* PAGINATION */}
        <div className="pagination">
          <button
            className="page-btn"
            disabled={currentPage === 1}
            onClick={() => setCurrentPage(currentPage - 1)}
          >
            Previous
          </button>

          {[...Array(totalPages)].map((_, index) => (
            <button
              key={index}
              className={`page-btn ${currentPage === index + 1 ? "active" : ""}`}
              onClick={() => setCurrentPage(index + 1)}
            >
              {index + 1}
            </button>
          ))}

          <button
            className="page-btn"
            disabled={currentPage === totalPages || totalPages === 0}
            onClick={() => setCurrentPage(currentPage + 1)}
          >
            Next
          </button>

          <div className="patient-count">
            Showing {filteredPatients.length} patients
          </div>
        </div>
      </div>
    </div>
  );
};

export default PatientManagement;
