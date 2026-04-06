import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Login from './pages/Login';
import Dashboard from './pages/Dashboard';
import UploadScan from './pages/UploadScan';
import PatientDetails from './pages/PatientDetails';
import RadiologyReport from './pages/RadiologyReport';
import Reports from "./pages/Reports";
import "./theme.css";


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
        <Route path="/reports" element={<Reports />} />
      </Routes>
    </div>
  );
}

export default App;
