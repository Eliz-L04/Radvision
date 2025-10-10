import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './Login.css';

const Login = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [userType, setUserType] = useState('radiologist');
  const navigate = useNavigate();

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log('Login attempt:', { email, password, userType });
    navigate('/dashboard');
  };

  return (
    <div className="login-container">
      {/* Floating header */}
      <div className="login-header">
        <h1>AI-Powered Radiology Report Generation</h1>
        <p>Faster, Smarter, More Transparent Diagnostics for enhanced patient care.</p>
      </div>

      {/* Login box */}
      <div className="login-card">
        <h2 className="welcome-text">Welcome Back</h2>
        <p className="sub-text">Sign in to your account</p>

        <form onSubmit={handleSubmit} className="login-form">
          <div className="form-group">
            <input
              type="text"
              placeholder="Email or Username"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </div>

          <div className="form-group">
            <input
              type="password"
              placeholder="Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </div>

          <div className="user-type-selector">
            <button
              type="button"
              className={userType === 'radiologist' ? 'active' : ''}
              onClick={() => setUserType('radiologist')}
            >
              Radiologist
            </button>
            <button
              type="button"
              className={userType === 'admin' ? 'active' : ''}
              onClick={() => setUserType('admin')}
            >
              Admin
            </button>
          </div>

          <button type="submit" className="login-btn">Login</button>
        </form>

        <div className="login-divider"><span>OR</span></div>

        <div className="alternative-login">
          <button className="google-btn">Continue with Google</button>
          <button className="institution-btn">Sign in with Institution</button>
        </div>

        <footer className="login-footer">
          <div>Company • Resources • Legal</div>
        </footer>
      </div>
    </div>
  );
};

export default Login;
