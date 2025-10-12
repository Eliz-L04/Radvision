import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './Login.css';

const Login = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [userType, setUserType] = useState('radiologist');
  const [isSignup, setIsSignup] = useState(false);
  const [signupMessage, setSignupMessage] = useState('');
  const [loginMessage, setLoginMessage] = useState('');
  const navigate = useNavigate();

  // Login handler
  const handleLogin = async (e) => {
    e.preventDefault();
    setLoginMessage('');
    try {
      const res = await fetch('http://127.0.0.1:5000/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      });
      const data = await res.json();
      if (res.ok) {
        navigate('/dashboard');
      } else {
        setLoginMessage(data.message || 'Login failed');
      }
    } catch (err) {
      setLoginMessage('Server error');
    }
  };

  // Signup handler
  const handleSignup = async (e) => {
    e.preventDefault();
    setSignupMessage('');
    try {
      const res = await fetch('http://127.0.0.1:5000/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      });
      const data = await res.json();
      if (res.ok) {
        setSignupMessage('Signup successful! You can now log in.');
        setIsSignup(false);
      } else {
        setSignupMessage(data.message || 'Signup failed');
      }
    } catch (err) {
      setSignupMessage('Server error');
    }
  };

  return (
    <div className="login-container">
      <div className="login-header">
        <h1>AI-Powered Radiology Report Generation</h1>
        <p>Faster, Smarter, More Transparent Diagnostics for enhanced patient care.</p>
      </div>

      <div className="login-card">
        {isSignup ? (
          <>
            <h2 className="welcome-text">Create Account</h2>
            <p className="sub-text">Sign up to get started</p>
            <form onSubmit={handleSignup} className="login-form">
              <div className="form-group">
                <input
                  type="text"
                  placeholder="Email"
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
              <button type="submit" className="login-btn">Sign Up</button>
            </form>
            {signupMessage && <div style={{ color: 'red', marginTop: 10 }}>{signupMessage}</div>}
            <div style={{ marginTop: 18 }}>
              Already have an account?{' '}
              <button type="button" className="secondary-btn" onClick={() => { setIsSignup(false); setSignupMessage(''); }}>
                Login
              </button>
            </div>
          </>
        ) : (
          <>
            <h2 className="welcome-text">Welcome Back</h2>
            <p className="sub-text">Sign in to your account</p>
            <form onSubmit={handleLogin} className="login-form">
              <div className="form-group">
                <input
                  type="text"
                  placeholder="Email"
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
              <button type="submit" className="login-btn">Login</button>
            </form>
            {loginMessage && <div style={{ color: 'red', marginTop: 10 }}>{loginMessage}</div>}
            <div style={{ marginTop: 18 }}>
              New user?{' '}
              <button type="button" className="secondary-btn" onClick={() => { setIsSignup(true); setLoginMessage(''); }}>
                Sign Up
              </button>
            </div>
          </>
        )}
        <footer className="login-footer">
          <div>Company • Resources • Legal</div>
        </footer>
      </div>
    </div>
  );
};

export default Login;
