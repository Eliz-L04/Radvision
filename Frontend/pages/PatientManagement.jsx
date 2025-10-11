import React, { useState } from 'react';
import './PatientManagement.css';

const PatientManagement = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [filters, setFilters] = useState({
    scanType: '',
    dateRange: ''
  });

  // Mock patient data
  const patients = [
    { id: 'RAD-001', name: 'Alice Johnson', age: 45, gender: 'Female', scanHistory: ['CT Scan', 'MRI Scan'], lastReport: '2023-10-28' },
    { id: 'RAD-002', name: 'Bob Williams', age: 62, gender: 'Male', scanHistory: ['X-Ray'], lastReport: '2023-11-03' },
    { id: 'RAD-003', name: 'Carol Davis', age: 38, gender: 'Female', scanHistory: ['Ultrasound', 'CT Scan'], lastReport: '2023-09-22' },
    { id: 'RAD-004', name: 'David Brown', age: 71, gender: 'Male', scanHistory: ['MRI Scan', 'X-Ray'], lastReport: '2023-10-12' },
    { id: 'RAD-005', name: 'Eve Green', age: 29, gender: 'Female', scanHistory: ['CT Scan'], lastReport: '2023-11-17' },
    { id: 'RAD-006', name: 'Frank White', age: 55, gender: 'Male', scanHistory: ['Ultrasound', 'MRI Scan'], lastReport: '2023-12-03' }
  ];

  const filteredPatients = patients.filter(patient =>
    patient.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    patient.id.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="patient-management-container">
      <header className="management-header">
        <h1>Efficient Patient Data Management</h1>
        <p>Access, organize, and analyze comprehensive patient records with RadAI's intuitive database.</p>
      </header>

      <section className="search-section">
        <h3>Add New Patient</h3>
        <div className="search-filters">
          <input
            type="text"
            placeholder="Search by Patient ID or Name"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="search-input"
          />
          <select 
            value={filters.scanType}
            onChange={(e) => setFilters(prev => ({ ...prev, scanType: e.target.value }))}
          >
            <option value="">Scan Type</option>
            <option value="ct">CT Scan</option>
            <option value="mri">MRI Scan</option>
            <option value="xray">X-Ray</option>
            <option value="ultrasound">Ultrasound</option>
          </select>
          <select 
            value={filters.dateRange}
            onChange={(e) => setFilters(prev => ({ ...prev, dateRange: e.target.value }))}
          >
            <option value="">Date Range</option>
            <option value="week">Last Week</option>
            <option value="month">Last Month</option>
            <option value="quarter">Last Quarter</option>
          </select>
          <button className="filter-btn">Apply Filters</button>
        </div>
      </section>

      <section className="patient-records">
        <h3>Patient Records</h3>
        <p>Comprehensive list of all patients and their medical history.</p>
        
        <div className="table-container">
          <table className="patients-table">
            <thead>
              <tr>
                <th>Patient ID</th>
                <th>Name</th>
                <th>Age</th>
                <th>Gender</th>
                <th>Scan History</th>
                <th>Last Report Date</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredPatients.map(patient => (
                <tr key={patient.id}>
                  <td>{patient.id}</td>
                  <td>{patient.name}</td>
                  <td>{patient.age}</td>
                  <td>{patient.gender}</td>
                  <td>
                    <div className="scan-history">
                      {patient.scanHistory.map((scan, index) => (
                        <span key={index} className="scan-tag">{scan}</span>
                      ))}
                    </div>
                  </td>
                  <td>{patient.lastReport}</td>
                  <td>
                    <button className="action-btn view-btn">View</button>
                    <button className="action-btn report-btn">Report</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <footer className="management-footer">
        <div>Company Resources</div>
      </footer>
    </div>
  );
};

export default PatientManagement;