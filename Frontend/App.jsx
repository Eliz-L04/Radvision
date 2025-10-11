import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Login from './pages/Login';
import Dashboard from './pages/Dashboard';
import UploadScan from './pages/UploadScan';
import PatientDetails from './pages/PatientDetails';
import RadiologyReport from './pages/RadiologyReport';
import PatientManagement from './pages/PatientManagement';
import './App.css';

function App() {
  return (
    <div className="App">
      <Routes>
        <Route path="/" element={<Login />} />
        <Route path="/login" element={<Login />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/upload" element={<UploadScan />} />
        <Route path="/patient/:id" element={<PatientDetails />} />
        <Route path="/report/:id" element={<RadiologyReport />} />
        <Route path="/patients" element={<PatientManagement />} />
      </Routes>
    </div>
  );
}

export default App;
