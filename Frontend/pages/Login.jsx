import React, { useState } from 'react';
import { Eye, EyeOff } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import './Login.css';

const Login = () => {
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [dob, setDob] = useState('');
  const [showPassword, setShowPassword] = useState(false);
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
        localStorage.setItem('username', data.username || 'User');
        navigate('/dashboard');
      } else {
        setLoginMessage(data.message || 'Login failed');
      }
    } catch (err) {
      setLoginMessage('Server error');
    }
  };

  // Password validation
  const isPasswordValid = (pwd) => {
    return pwd.length >= 8 && /\d/.test(pwd);
  };

  // Signup handler
  const handleSignup = async (e) => {
    e.preventDefault();
    setSignupMessage('');
    if (!username || !email || !dob || !password) {
      setSignupMessage('All fields are required.');
      return;
    }
    if (!isPasswordValid(password)) {
      setSignupMessage(
        'Password must be at least 8 characters and contain at least one number.'
      );
      return;
    }
    try {
      const res = await fetch('http://127.0.0.1:5000/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, email, password, dob })
      });
      const data = await res.json();
      if (res.ok) {
        setSignupMessage('Signup successful! You can now log in.');
        setIsSignup(false);
        setUsername('');
        setEmail('');
        setPassword('');
        setDob('');
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
        <h1>RadVision</h1>
        <p>
          AI-Powered Radiology Reporting & Diagnostic Workstation
        </p>
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
                  placeholder="Username"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  required
                />
              </div>

              <div className="form-group">
                <input
                  type="email"
                  placeholder="Email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                />
              </div>

              <div className="form-group">
                <input
                  type="date"
                  placeholder="Date of Birth"
                  value={dob}
                  onChange={(e) => setDob(e.target.value)}
                  required
                />
              </div>

              <div className="form-group">
                <input
                  type={showPassword ? "text" : "password"}
                  placeholder="Password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  style={{ paddingRight: '32px', height: '38px', fontSize: '15px' }}
                />
                <span
                  style={{
                    position: 'absolute',
                    right: '10px',
                    top: '50%',
                    transform: 'translateY(-50%)',
                    cursor: 'pointer'
                  }}
                  onClick={() => setShowPassword(!showPassword)}
                  aria-label={showPassword ? 'Hide password' : 'Show password'}
                >
                  {showPassword ? <EyeOff size={20} /> : <Eye size={20} />}
                </span>
              </div>

              <button type="submit" className="login-btn">Sign Up</button>
            </form>

            {signupMessage && (
              <div style={{ color: 'red', marginTop: 10 }}>
                {signupMessage}
              </div>
            )}

            <div style={{ marginTop: 18 }}>
              Already have an account?{" "}
              <button
                type="button"
                className="secondary-btn"
                onClick={() => {
                  setIsSignup(false);
                  setSignupMessage('');
                }}
              >
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
                  type="email"
                  placeholder="Email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                />
              </div>

              <div className="form-group">
                <input
                  type={showPassword ? "text" : "password"}
                  placeholder="Password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  style={{ paddingRight: '32px' }}
                />
                <span
                  style={{
                    position: 'absolute',
                    right: '10px',
                    top: '50%',
                    transform: 'translateY(-50%)',
                    cursor: 'pointer'
                  }}
                  onClick={() => setShowPassword(!showPassword)}
                  aria-label={showPassword ? 'Hide password' : 'Show password'}
                >
                  {showPassword ? <EyeOff size={20} /> : <Eye size={20} />}
                </span>
              </div>

              <button type="submit" className="login-btn">Login</button>
            </form>

            {loginMessage && (
              <div style={{ color: 'red', marginTop: 10 }}>
                {loginMessage}
              </div>
            )}

            <div style={{ marginTop: 18 }}>
              New user?{" "}
              <button
                type="button"
                className="secondary-btn"
                onClick={() => {
                  setIsSignup(true);
                  setLoginMessage('');
                }}
              >
                Sign Up
              </button>
            </div>
          </>
        )}

        <footer className="login-footer">
          <div>RadVision • Medical AI Platform • Secure Access</div>
        </footer>
      </div>
    </div>
  );
};

export default Login;

