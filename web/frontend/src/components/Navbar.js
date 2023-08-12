import React from 'react';
import  './Navbar.css';
import { Link } from 'react-router-dom';

export default function Navbar() {
  return (
    <div>
      <nav className="navbar navbar-expand-lg">
  <div className="container-fluid">
    <div className="logo">LOGO</div>
    <Link className="navbar-brand" to="/">WISEBUCK.AI</Link>
    <button className="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarSupportedContent" aria-controls="navbarSupportedContent" aria-expanded="false" aria-label="Toggle navigation">
      <span className="navbar-toggler-icon"></span>
    </button>
    <div className="collapse navbar-collapse" id="navbarSupportedContent">
      <ul className="navbar-nav me-auto mb-2 mb-lg-0">
        {/* <li className="nav-item">
          <Link className="nav-link " aria-current="page" to="/">Home</Link>
        </li> */}
        {/* <li className="nav-item">
          <Link className="nav-link" to="/mcq">MCQ</Link>
        </li> */}
        <li className="nav-item dropdown">
          <Link className="nav-link dropdown-toggle" to="/features" role="button" data-bs-toggle="dropdown" aria-expanded="false">
            Features
          </Link>
          <ul className="dropdown-menu">
            <li><Link className="dropdown-item" to="/investing">Investment Challenge</Link></li>
            <li><Link className="dropdown-item" to="/chatbot">WiseBuck Chatbot</Link></li>
            <li><Link className="dropdown-item" to="/paper">Paper Trading</Link></li>
            {/* <li><hr className="dropdown-divider"/></li> */}
            <li><Link className="dropdown-item" to="/competing">Competing against friends</Link></li>
            <li><Link className="dropdown-item" to="/rewards">Reward Based Investing</Link></li>

          </ul>
        </li>
        <li className="nav-item">
          <Link className="nav-link " aria-current="page" to="/about">About Us</Link>
        </li>
        <li className="nav-item">
          <Link className="nav-link " aria-current="page" to="/contact">Contact Us</Link>
        </li>
        <li className="nav-item">
          <Link className="nav-link" to="/mcq">MCQ</Link>
        </li>
        
      </ul>
      <Link to="/login" className=" loginnavbtn">
              LOGIN
            </Link>
        <Link to="/signup" className="btn signupnavbtn">
              SIGN UP
            </Link>
           

    </div>
  </div>
</nav>
    </div>
  )
}
